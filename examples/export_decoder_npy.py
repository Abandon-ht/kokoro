import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from kokoro import KModel, KPipeline


ASR_FRAME_LEN = 198
F0_FRAME_LEN = ASR_FRAME_LEN * 2
RESIDUAL_SCALE = 2 ** -0.5


def forward_adain_res_block_for_export(block, x: torch.FloatTensor, timbre: torch.FloatTensor) -> torch.FloatTensor:
    residual = block.norm1(x, timbre)
    residual = block.actv(residual)
    if block.upsample_type == 'none':
        residual = block.pool(residual)
    else:
        residual = torch.nn.functional.interpolate(residual, scale_factor=2, mode='nearest')
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


def collect_decoder_tensors(
    model: KModel,
    pipeline: KPipeline,
    decoder_front: KDecoderFrontForExport,
    text: str,
    voice: str,
    speed: float,
) -> dict[str, torch.Tensor]:
    phonemes, input_ids = load_input_ids(pipeline, text)
    ref_s, timbre = load_timbre(pipeline, voice, len(phonemes))

    with torch.no_grad():
        d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
        _, _, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=speed)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, ref_s[:, 128:])

        asr_static = resize_last_dim(asr, ASR_FRAME_LEN)
        f0_static = resize_last_dim(f0_pred, F0_FRAME_LEN)
        n_static = resize_last_dim(n_pred, F0_FRAME_LEN)
        decoder_state = decoder_front(asr_static, f0_static, n_static, timbre)

    return {
        'asr': asr_static.detach().cpu(),
        'F0_pred': f0_static.detach().cpu(),
        'N_pred': n_static.detach().cpu(),
        'timbre': timbre.detach().cpu(),
        'decoder_state': decoder_state.detach().cpu(),
    }


def save_tensors(root_dir: Path, index: int, tensors: dict[str, torch.Tensor]):
    for name, tensor in tensors.items():
        output_dir = root_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{index:03d}.npy', tensor.numpy().astype(np.float32, copy=False))


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


def main():
    parser = argparse.ArgumentParser('Export fixed-shape decoder_front calibration NPY files', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--lang_code', '-l', type=str, default='a', help='pipeline language code')
    parser.add_argument('--voice', '-v', type=str, default='af_heart', help='voice id or .pt path')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='speech speed')
    parser.add_argument('--sample_count', '-n', type=int, default=10, help='number of samples to export')
    parser.add_argument('--text_file', '-t', type=str, default='demo/en.txt', help='text file with one utterance per line')
    parser.add_argument('--output_dir', '-o', type=str, default='decoder_npy', help='directory where timestamped exports are written')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run on')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = Path(args.output_dir) / timestamp
    texts = load_texts(Path(args.text_file), args.sample_count)

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    model.bert.config._attn_implementation = 'eager'
    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)
    decoder_front = KDecoderFrontForExport(model).eval()

    print(f'export root: {export_root}')
    print(f'samples    : {len(texts)}')
    print(f'voice      : {args.voice}')
    print(f'shapes     : asr=(1,512,{ASR_FRAME_LEN}), F0_pred=(1,{F0_FRAME_LEN}), N_pred=(1,{F0_FRAME_LEN}), timbre=(1,128), decoder_state=(1,512,{F0_FRAME_LEN})')

    for index, text in enumerate(texts):
        tensors = collect_decoder_tensors(model, pipeline, decoder_front, text, args.voice, args.speed)
        save_tensors(export_root, index, tensors)
        print(f'[{index + 1}/{len(texts)}] saved decoder_front calibration tensors for: {text[:80]}')


if __name__ == '__main__':
    main()