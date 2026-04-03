import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from kokoro import KModel, KPipeline
from kokoro.model import KEncoderForONNX, KF0NHeadForONNX, KF0NSharedForONNX, KTextEncoderForONNX


class KEncoderFeaturesForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.encoder = KEncoderForONNX(kmodel)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        d_en, _, _ = self.encoder(input_ids)
        return d_en


DEFAULT_TOKEN_BUCKETS = '128,256,512'
DEFAULT_FRAME_BUCKETS = '198,256,396,512'
DEFAULT_TEXT_ENCODER_BUCKETS = '128:198,256:396,512:512'


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


def load_ref_s(pipeline: KPipeline, voice: str, phoneme_length: int) -> torch.FloatTensor:
    pack = pipeline.load_voice(voice).to(pipeline.model.device)
    ref_s = pack[phoneme_length - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def save_tensors(root_dir: Path, index: int, tensors: dict[str, torch.Tensor]):
    for name, tensor in tensors.items():
        output_dir = root_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{index:03d}.npy', tensor_to_numpy(tensor))


def collect_static_frontend_tensors(
    model: KModel,
    pipeline: KPipeline,
    encoder: KEncoderFeaturesForONNX,
    text_encoder: KTextEncoderForONNX,
    f0n_shared: KF0NSharedForONNX,
    f0n_head: KF0NHeadForONNX,
    text: str,
    voice: str,
    speed: float,
    token_buckets: list[int],
    frame_buckets: list[int],
    text_encoder_buckets: list[tuple[int, int]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, int]]:
    phonemes, input_ids = load_input_ids(pipeline, text)
    ref_s = load_ref_s(pipeline, voice, len(phonemes))

    with torch.no_grad():
        d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
        _, _, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=speed)

        token_length = input_ids.shape[-1]
        frame_length = pred_aln_trg.shape[-1]

        encoder_bucket = select_bucket(token_length, token_buckets, 'encoder token')
        f0n_bucket = select_bucket(frame_length, frame_buckets, 'f0n frame')
        text_token_bucket, text_frame_bucket = select_text_encoder_bucket(
            token_length,
            frame_length,
            text_encoder_buckets,
        )

        encoder_input_ids = resize_last_dim(input_ids, encoder_bucket)
        encoder_d_en = encoder(encoder_input_ids)

        text_encoder_input_ids = resize_last_dim(input_ids, text_token_bucket)
        text_encoder_pred_aln_trg = resize_alignment(pred_aln_trg, text_token_bucket, text_frame_bucket)
        text_encoder_t_en, text_encoder_asr = text_encoder(text_encoder_input_ids, text_encoder_pred_aln_trg)

        f0n_en = resize_last_dim(en, f0n_bucket)
        f0n_shared_out = f0n_shared(f0n_en)
        f0n_f0_pred, f0n_n_pred = f0n_head(f0n_shared_out, ref_s)

    tensors = {
        f'encoder_token_{encoder_bucket}': {
            'input_ids': encoder_input_ids,
            'd_en': encoder_d_en,
        },
        f'text_encoder_token_{text_token_bucket}_frame_{text_frame_bucket}': {
            'input_ids': text_encoder_input_ids,
            'pred_aln_trg': text_encoder_pred_aln_trg,
            't_en': text_encoder_t_en,
            'asr': text_encoder_asr,
        },
        f'f0n_shared_frame_{f0n_bucket}': {
            'en': f0n_en,
            'shared': f0n_shared_out,
        },
        f'f0n_head_frame_{f0n_bucket}': {
            'shared': f0n_shared_out,
            'ref_s': ref_s,
            'F0_pred': f0n_f0_pred,
            'N_pred': f0n_n_pred,
        },
    }
    lengths = {
        'token_length': token_length,
        'frame_length': frame_length,
        'encoder_bucket': encoder_bucket,
        'text_token_bucket': text_token_bucket,
        'text_frame_bucket': text_frame_bucket,
        'f0n_bucket': f0n_bucket,
    }
    return tensors, lengths


def main():
    parser = argparse.ArgumentParser('Export fixed-bucket frontend calibration NPY files', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--lang_code', '-l', type=str, default='a', help='pipeline language code')
    parser.add_argument('--voice', '-v', type=str, default='af_heart', help='voice id or .pt path')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='speech speed')
    parser.add_argument('--sample_count', '-n', type=int, default=10, help='number of samples to export')
    parser.add_argument('--text_file', '-t', type=str, default='demo/en.txt', help='text file with one utterance per line')
    parser.add_argument('--output_dir', '-o', type=str, default='static_frontend_npy', help='directory where timestamped exports are written')
    parser.add_argument('--token_buckets', type=str, default=DEFAULT_TOKEN_BUCKETS, help='comma-separated token buckets for encoder export')
    parser.add_argument('--frame_buckets', type=str, default=DEFAULT_FRAME_BUCKETS, help='comma-separated frame buckets for f0n export')
    parser.add_argument('--text_encoder_buckets', type=str, default=DEFAULT_TEXT_ENCODER_BUCKETS, help='comma-separated token:frame buckets for text_encoder export')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run on')
    args = parser.parse_args()

    token_buckets = parse_bucket_list(args.token_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = Path(args.output_dir) / timestamp
    texts = load_texts(Path(args.text_file), args.sample_count)

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    model.bert.config._attn_implementation = 'eager'
    validate_token_buckets(model, token_buckets, text_encoder_buckets)

    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)
    encoder = KEncoderFeaturesForONNX(model).eval()
    text_encoder = KTextEncoderForONNX(model).eval()
    f0n_shared = KF0NSharedForONNX(model).eval()
    f0n_head = KF0NHeadForONNX(model).eval()
    bucket_counts = defaultdict(int)

    print(f'export root          : {export_root}')
    print(f'samples              : {len(texts)}')
    print(f'voice                : {args.voice}')
    print(f'encoder buckets      : {token_buckets}')
    print(f'f0n frame buckets    : {frame_buckets}')
    print(f'text_encoder buckets : {text_encoder_buckets}')
    print('note                 : bucketed encoder/text_encoder inputs are right-padded to match static ONNX graph shapes')

    for text in texts:
        tensors_by_bucket, lengths = collect_static_frontend_tensors(
            model,
            pipeline,
            encoder,
            text_encoder,
            f0n_shared,
            f0n_head,
            text,
            args.voice,
            args.speed,
            token_buckets,
            frame_buckets,
            text_encoder_buckets,
        )

        for bucket_name, tensors in tensors_by_bucket.items():
            save_index = bucket_counts[bucket_name]
            save_tensors(export_root / bucket_name, save_index, tensors)
            bucket_counts[bucket_name] += 1

        print(
            f'saved token_len={lengths["token_length"]}, frame_len={lengths["frame_length"]} '
            f'-> encoder={lengths["encoder_bucket"]}, '
            f'text_encoder=({lengths["text_token_bucket"]},{lengths["text_frame_bucket"]}), '
            f'f0n={lengths["f0n_bucket"]} for: {text[:80]}'
        )

    print('bucket sample counts:')
    for bucket_name in sorted(bucket_counts):
        print(f'  {bucket_name}: {bucket_counts[bucket_name]}')


if __name__ == '__main__':
    main()