import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kokoro import KModel, KPipeline


STATIC_ASR_LEN = 198
STATIC_PITCH_LEN = STATIC_ASR_LEN * 2
VOCODER_STATE_LEN = STATIC_PITCH_LEN
HAR_CHANNELS = 22
HAR_FRAMES = 23761
WAVEFORM_LEN = 118800
RESIDUAL_SCALE = 2 ** -0.5


def forward_adain_res_block_for_export(block, x: torch.FloatTensor, timbre: torch.FloatTensor) -> torch.FloatTensor:
    residual = block.norm1(x, timbre)
    residual = block.actv(residual)
    if block.upsample_type == 'none':
        residual = block.pool(residual)
    else:
        residual = F.interpolate(residual, scale_factor=2, mode='nearest')
    residual = block.conv1(block.dropout(residual))
    residual = block.norm2(residual, timbre)
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual))

    shortcut = block.upsample(x)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    return (residual + shortcut) * RESIDUAL_SCALE


class KDecoderFrontForExport(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder

    def forward(
        self,
        asr: torch.FloatTensor,
        f0_pred: torch.FloatTensor,
        n_pred: torch.FloatTensor,
        timbre: torch.FloatTensor,
    ) -> torch.FloatTensor:
        f0 = self.decoder.F0_conv(f0_pred.unsqueeze(1))
        noise = self.decoder.N_conv(n_pred.unsqueeze(1))
        x = torch.cat([asr, f0, noise], axis=1)
        x = self.decoder.encode(x, timbre)
        asr_res = self.decoder.asr_res(asr)
        res = True
        for block in self.decoder.decode:
            if res:
                x = torch.cat([x, asr_res, f0, noise], axis=1)
            x = forward_adain_res_block_for_export(block, x, timbre)
            if block.upsample_type != 'none':
                res = False
        return x


class KVocoderForExport(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator

    def forward(
        self,
        decoder_state: torch.FloatTensor,
        timbre: torch.FloatTensor,
        har: torch.FloatTensor,
    ) -> torch.FloatTensor:
        x = decoder_state
        for index in range(self.generator.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.generator.noise_convs[index](har)
            x_source = self.generator.noise_res[index](x_source, timbre)
            x = self.generator.ups[index](x)
            if index == self.generator.num_upsamples - 1:
                x = self.generator.reflection_pad(x)
            x = x + x_source
            xs = None
            for kernel_index in range(self.generator.num_kernels):
                block = self.generator.resblocks[index * self.generator.num_kernels + kernel_index]
                if xs is None:
                    xs = block(x, timbre)
                else:
                    xs += block(x, timbre)
            x = xs / self.generator.num_kernels

        x = F.leaky_relu(x)
        x = self.generator.conv_post(x)
        spec = torch.exp(x[:, :self.generator.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.generator.post_n_fft // 2 + 1:, :])
        return self.generator.stft.inverse(spec, phase).squeeze()


def resize_last_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = tensor.shape[-1]
    if current_len == target_len:
        return tensor

    resized = tensor.new_zeros(*tensor.shape[:-1], target_len)
    copy_len = min(current_len, target_len)
    resized[..., :copy_len] = tensor[..., :copy_len]
    return resized


def load_input_ids(pipeline: KPipeline, text: str) -> tuple[str, torch.LongTensor]:
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        phonemes = ''
        for _, chunk_phonemes, _ in pipeline.en_tokenize(tokens):
            if chunk_phonemes:
                phonemes = chunk_phonemes
                break
    else:
        phonemes, _ = pipeline.g2p(text)

    if not phonemes:
        raise ValueError('No phonemes were produced from the input text.')

    if len(phonemes) > 510:
        phonemes = phonemes[:510]

    input_ids = [pipeline.model.vocab[p] for p in phonemes if pipeline.model.vocab.get(p) is not None]
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return phonemes, input_ids


def load_timbre(pipeline: KPipeline, voice: str, phoneme_length: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    pack = pipeline.load_voice(voice).to(pipeline.model.device)
    ref_s = pack[phoneme_length - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s, ref_s[:, :128]


def load_texts(text_file: Path, sample_count: int) -> list[str]:
    texts = []
    with text_file.open('r', encoding='utf-8') as handle:
        for line in handle:
            text = line.strip()
            if text:
                texts.append(text)
            if len(texts) == sample_count:
                break

    if len(texts) < sample_count:
        raise ValueError(f'Not enough non-empty lines in {text_file} to generate {sample_count} samples.')
    return texts


def collect_vocoder_tensors(
    model: KModel,
    pipeline: KPipeline,
    decoder_front: KDecoderFrontForExport,
    vocoder: KVocoderForExport,
    text: str,
    voice: str,
    speed: float,
) -> dict[str, torch.Tensor]:
    phonemes, input_ids = load_input_ids(pipeline, text)
    ref_s, timbre = load_timbre(pipeline, voice, len(phonemes))
    generator = model.decoder.generator

    with torch.no_grad():
        d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
        _, _, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=speed)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, ref_s[:, 128:])

        asr_static = resize_last_dim(asr, STATIC_ASR_LEN)
        f0_static = resize_last_dim(f0_pred, STATIC_PITCH_LEN)
        n_static = resize_last_dim(n_pred, STATIC_PITCH_LEN)
        decoder_state = decoder_front(asr_static, f0_static, n_static, timbre)

        f0_upsampled = generator.f0_upsamp(f0_static[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0_upsampled)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        decoder_state_static = resize_last_dim(decoder_state, VOCODER_STATE_LEN)
        har_static = har.new_zeros(har.shape[0], HAR_CHANNELS, HAR_FRAMES)
        copy_channels = min(har.shape[1], HAR_CHANNELS)
        copy_frames = min(har.shape[2], HAR_FRAMES)
        har_static[:, :copy_channels, :copy_frames] = har[:, :copy_channels, :copy_frames]
        waveform = vocoder(decoder_state_static, timbre, har_static)
        waveform_static = resize_last_dim(waveform.unsqueeze(0), WAVEFORM_LEN).squeeze(0)

    return {
        'decoder_state': decoder_state_static.detach().cpu(),
        'timbre': timbre.detach().cpu(),
        'har': har_static.detach().cpu(),
        'waveform': waveform_static.detach().cpu(),
    }


def save_tensors(root_dir: Path, index: int, tensors: dict[str, torch.Tensor]):
    for name, tensor in tensors.items():
        output_dir = root_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{index:03d}.npy', tensor.numpy().astype(np.float32, copy=False))


def main():
    parser = argparse.ArgumentParser('Export fixed-shape vocoder calibration NPY files', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--lang_code', '-l', type=str, default='a', help='pipeline language code')
    parser.add_argument('--voice', '-v', type=str, default='af_heart', help='voice id or .pt path')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='speech speed')
    parser.add_argument('--sample_count', '-n', type=int, default=10, help='number of samples to export')
    parser.add_argument('--text_file', '-t', type=str, default='demo/en.txt', help='text file with one utterance per line')
    parser.add_argument('--output_dir', '-o', type=str, default='vocoder_npy', help='directory where timestamped exports are written')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run on')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = Path(args.output_dir) / timestamp
    texts = load_texts(Path(args.text_file), args.sample_count)

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    model.bert.config._attn_implementation = 'eager'
    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)
    decoder_front = KDecoderFrontForExport(model).eval()
    vocoder = KVocoderForExport(model).eval()

    print(f'export root: {export_root}')
    print(f'samples    : {len(texts)}')
    print(f'voice      : {args.voice}')
    print(f'shapes     : decoder_state=(1,512,{VOCODER_STATE_LEN}), timbre=(1,128), har=(1,{HAR_CHANNELS},{HAR_FRAMES}), waveform=({WAVEFORM_LEN},)')

    for index, text in enumerate(texts):
        tensors = collect_vocoder_tensors(model, pipeline, decoder_front, vocoder, text, args.voice, args.speed)
        save_tensors(export_root, index, tensors)
        print(f'[{index + 1}/{len(texts)}] saved vocoder calibration tensors for: {text[:80]}')


if __name__ == '__main__':
    main()