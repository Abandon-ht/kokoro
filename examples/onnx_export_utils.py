from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from kokoro import KModel, KPipeline


DEFAULT_TOKEN_BUCKETS = '128,256,512'
DEFAULT_FRAME_BUCKETS = '198,256,396,512'
DEFAULT_TEXT_ENCODER_BUCKETS = '128:198,256:396,512:512'
DEFAULT_DECODER_FRAME_BUCKET = 198
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SAMPLES_PER_FRAME = 600


def export_module(model, output_path, args, input_names, output_names, dynamic_axes=None):
    export_kwargs = {
        'args': args,
        'f': output_path,
        'export_params': True,
        'input_names': input_names,
        'output_names': output_names,
        'opset_version': 17,
        'do_constant_folding': True,
        'dynamo': False,
    }
    if dynamic_axes is not None:
        export_kwargs['dynamic_axes'] = dynamic_axes

    torch.onnx.export(model, **export_kwargs)
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f'exported {output_path}')


def parse_bucket_list(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(','):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f'bucket values must be positive, got {value}')
        values.append(value)
    if not values:
        raise ValueError('at least one bucket value is required')
    return sorted(set(values))


def parse_text_encoder_buckets(raw_value: str) -> list[tuple[int, int]]:
    buckets = []
    for item in raw_value.split(','):
        item = item.strip()
        if not item:
            continue
        token_part, frame_part = item.split(':', 1)
        token_bucket = int(token_part)
        frame_bucket = int(frame_part)
        if token_bucket <= 0 or frame_bucket <= 0:
            raise ValueError(f'bucket values must be positive, got {item}')
        buckets.append((token_bucket, frame_bucket))
    if not buckets:
        raise ValueError('at least one text encoder bucket is required')
    return sorted(set(buckets))


def validate_token_buckets(kmodel: KModel, token_buckets: list[int], text_encoder_buckets: list[tuple[int, int]]):
    all_token_buckets = set(token_buckets)
    all_token_buckets.update(token_bucket for token_bucket, _ in text_encoder_buckets)
    oversized = sorted(bucket for bucket in all_token_buckets if bucket > kmodel.context_length)
    if oversized:
        raise ValueError(
            f'token buckets exceed model context length {kmodel.context_length}: {oversized}'
        )


def select_bucket(length: int, buckets: list[int], bucket_name: str) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    raise ValueError(f'{bucket_name} length {length} exceeds available buckets {buckets}')


def select_text_encoder_bucket(
    token_length: int,
    frame_length: int,
    text_encoder_buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    for token_bucket, frame_bucket in text_encoder_buckets:
        if token_length <= token_bucket and frame_length <= frame_bucket:
            return token_bucket, frame_bucket
    raise ValueError(
        f'text_encoder sample token_length={token_length}, frame_length={frame_length} '
        f'exceeds available buckets {text_encoder_buckets}'
    )


def build_input_ids(kmodel: KModel, token_bucket: int) -> torch.LongTensor:
    upper_bound = len(kmodel.vocab)
    return torch.randint(1, upper_bound, (1, token_bucket), dtype=torch.long)


def build_ref_s(style_dim: int) -> torch.FloatTensor:
    return torch.randn(1, style_dim * 2, dtype=torch.float32)


def build_input_lengths(length: int) -> torch.LongTensor:
    return torch.tensor([length], dtype=torch.long)


def build_text_mask(length: int, bucket: int) -> torch.BoolTensor:
    positions = torch.arange(bucket, dtype=torch.long).unsqueeze(0)
    return torch.gt(positions + 1, build_input_lengths(length).unsqueeze(1))


def build_alignment(token_bucket: int, frame_bucket: int) -> torch.FloatTensor:
    frame_positions = torch.arange(frame_bucket, dtype=torch.long)
    token_indices = torch.div(frame_positions * token_bucket, frame_bucket, rounding_mode='floor')
    token_indices = token_indices.clamp(max=token_bucket - 1)
    pred_aln_trg = torch.zeros((1, token_bucket, frame_bucket), dtype=torch.float32)
    pred_aln_trg[0, token_indices, frame_positions] = 1.0
    return pred_aln_trg


def build_en(feature_dim: int, frame_bucket: int) -> torch.FloatTensor:
    return torch.randn(1, feature_dim, frame_bucket, dtype=torch.float32)


def resize_last_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = tensor.shape[-1]
    if current_len == target_len:
        return tensor

    resized = tensor.new_zeros(*tensor.shape[:-1], target_len)
    copy_len = min(current_len, target_len)
    resized[..., :copy_len] = tensor[..., :copy_len]
    return resized


def resize_alignment(tensor: torch.Tensor, token_bucket: int, frame_bucket: int) -> torch.Tensor:
    resized = tensor.new_zeros(tensor.shape[0], token_bucket, frame_bucket)
    copy_tokens = min(tensor.shape[1], token_bucket)
    copy_frames = min(tensor.shape[2], frame_bucket)
    resized[:, :copy_tokens, :copy_frames] = tensor[:, :copy_tokens, :copy_frames]
    return resized


def right_pad_last_dim(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.shape[-1] > target_length:
        raise ValueError(f'Cannot pad array with length {array.shape[-1]} to shorter target {target_length}.')
    if array.shape[-1] == target_length:
        return array

    padded = np.zeros((*array.shape[:-1], target_length), dtype=array.dtype)
    padded[..., :array.shape[-1]] = array
    return padded


def right_pad_alignment(alignment: np.ndarray, token_bucket: int, frame_bucket: int) -> np.ndarray:
    if alignment.shape[1] > token_bucket or alignment.shape[2] > frame_bucket:
        raise ValueError(
            f'Alignment shape {alignment.shape} exceeds static bucket {(token_bucket, frame_bucket)}.'
        )

    padded = np.zeros((alignment.shape[0], token_bucket, frame_bucket), dtype=alignment.dtype)
    padded[:, :alignment.shape[1], :alignment.shape[2]] = alignment
    return padded


def build_padded_text_mask(token_length: int, token_bucket: int) -> np.ndarray:
    positions = np.arange(token_bucket, dtype=np.int64)[None, :]
    return positions >= token_length


def load_texts(text_file: Path, sample_count: int | None = None) -> list[str]:
    texts = []
    with text_file.open('r', encoding='utf-8') as handle:
        for line in handle:
            text = line.strip()
            if text:
                texts.append(text)
            if sample_count is not None and len(texts) == sample_count:
                break

    if sample_count is not None and len(texts) < sample_count:
        raise ValueError(f'Not enough non-empty lines in {text_file} to generate {sample_count} samples.')
    return texts


def load_phonemes_and_input_ids_torch(pipeline: KPipeline, text: str) -> tuple[str, torch.LongTensor]:
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


def load_phonemes_and_input_ids_numpy(pipeline: KPipeline, text: str) -> tuple[str, np.ndarray]:
    phonemes, input_ids = load_phonemes_and_input_ids_torch(pipeline, text)
    return phonemes, input_ids.cpu().numpy().astype(np.int64, copy=False)


def load_reference_style_torch(pipeline: KPipeline, voice: str, phoneme_length: int) -> torch.FloatTensor:
    pack = pipeline.load_voice(voice).to(pipeline.model.device)
    ref_s = pack[phoneme_length - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s


def load_reference_style_numpy(pipeline: KPipeline, voice: str, phoneme_length: int) -> np.ndarray:
    return load_reference_style_torch(pipeline, voice, phoneme_length).cpu().numpy().astype(np.float32)


def create_session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(model_path.as_posix(), providers=providers)


def build_har(generator, f0_pred: np.ndarray) -> np.ndarray:
    f0_tensor = torch.from_numpy(f0_pred).float()
    with torch.no_grad():
        f0 = generator.f0_upsamp(f0_tensor[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
    return har.cpu().contiguous().numpy().astype(np.float32)


RESIDUAL_SCALE = 2 ** -0.5


def forward_adain_res_block_for_export(block, x: torch.FloatTensor, timbre: torch.FloatTensor) -> torch.FloatTensor:
    residual = block.norm1(x, timbre)
    residual = block.actv(residual)
    residual = block.pool(residual)
    residual = block.conv1(block.dropout(residual))
    residual = block.norm2(residual, timbre)
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual))

    shortcut = block.upsample(x)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    return (residual + shortcut) * RESIDUAL_SCALE


class KDecoderFrontForONNX(torch.nn.Module):
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


def run_vocoder_core_for_export(
    generator,
    decoder_state: torch.FloatTensor,
    timbre: torch.FloatTensor,
    har: torch.FloatTensor,
) -> torch.FloatTensor:
    x = decoder_state
    for index in range(generator.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = generator.noise_convs[index](har)
        x_source = generator.noise_res[index](x_source, timbre)
        x = generator.ups[index](x)
        x = x + x_source
        xs = None
        for kernel_index in range(generator.num_kernels):
            block = generator.resblocks[index * generator.num_kernels + kernel_index]
            if xs is None:
                xs = block(x, timbre)
            else:
                xs += block(x, timbre)
        x = xs / generator.num_kernels
    return x


class KVocoderCoreForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator

    def forward(
        self,
        decoder_state: torch.FloatTensor,
        timbre: torch.FloatTensor,
        har: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return run_vocoder_core_for_export(self.generator, decoder_state, timbre, har)


class KVocoderTailForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator

    def forward(self, vocoder_hidden: torch.FloatTensor) -> torch.FloatTensor:
        x = F.leaky_relu(vocoder_hidden)
        x = self.generator.conv_post(x)
        spec = torch.exp(x[:, :self.generator.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.generator.post_n_fft // 2 + 1:, :])
        return self.generator.stft.inverse(spec, phase).squeeze()


class KVocoderForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator

    def forward(
        self,
        decoder_state: torch.FloatTensor,
        timbre: torch.FloatTensor,
        har: torch.FloatTensor,
    ) -> torch.FloatTensor:
        x = run_vocoder_core_for_export(self.generator, decoder_state, timbre, har)
        x = F.leaky_relu(x)
        x = self.generator.conv_post(x)
        spec = torch.exp(x[:, :self.generator.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.generator.post_n_fft // 2 + 1:, :])
        return self.generator.stft.inverse(spec, phase).squeeze()


def build_dynamic_sample_inputs(kmodel: KModel) -> dict[str, torch.Tensor | int]:
    seq_len = min(50, kmodel.context_length)
    input_ids = torch.randint(1, len(kmodel.vocab), (1, seq_len), dtype=torch.long)
    ref_s = torch.randn(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        d_en, input_lengths, text_mask = kmodel._encode_linguistic_tokens(input_ids)
        _, pred_dur, pred_aln_trg, en = kmodel._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed)
        f0_pred, n_pred = kmodel.predictor.F0Ntrain(en, ref_s[:, 128:])
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

    return {
        'input_ids': input_ids,
        'ref_s': ref_s,
        'speed': speed,
        'd_en': d_en,
        'input_lengths': input_lengths,
        'text_mask': text_mask,
        'pred_dur': pred_dur,
        'pred_aln_trg': pred_aln_trg,
        'en': en,
        'F0_pred': f0_pred,
        'N_pred': n_pred,
        't_en': t_en,
        'asr': asr,
        'frame_len': int(pred_aln_trg.shape[-1]),
    }


def build_static_backend_samples(kmodel: KModel, frame_bucket: int) -> dict[str, torch.Tensor]:
    pitch_bucket = frame_bucket * 2
    asr = torch.randn(1, kmodel.bert_encoder.out_features, frame_bucket, dtype=torch.float32)
    f0_pred = torch.randn(1, pitch_bucket, dtype=torch.float32)
    n_pred = torch.randn(1, pitch_bucket, dtype=torch.float32)
    timbre = torch.randn(1, 128, dtype=torch.float32)

    decoder_front = KDecoderFrontForONNX(kmodel).eval()
    vocoder_core = KVocoderCoreForONNX(kmodel).eval()
    vocoder_tail = KVocoderTailForONNX(kmodel).eval()
    with torch.no_grad():
        decoder_state = decoder_front(asr, f0_pred, n_pred, timbre)
        har = torch.from_numpy(build_har(kmodel.decoder.generator, f0_pred.numpy()))
        vocoder_hidden = vocoder_core(decoder_state, timbre, har)
        waveform = vocoder_tail(vocoder_hidden)

    return {
        'asr': asr,
        'F0_pred': f0_pred,
        'N_pred': n_pred,
        'timbre': timbre,
        'decoder_state': decoder_state,
        'har': har,
        'vocoder_hidden': vocoder_hidden,
        'waveform': waveform,
    }


def save_numpy_tensors(root_dir: Path, model_name: str, sample_index: int, tensors: dict[str, np.ndarray]):
    for tensor_name, tensor in tensors.items():
        output_dir = root_dir / model_name / tensor_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{sample_index:03d}.npy', tensor)


def write_metadata(metadata_path: Path, records: list[dict]):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)