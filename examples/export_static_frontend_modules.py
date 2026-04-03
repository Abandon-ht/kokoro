import argparse
import os

import onnx
import torch

from kokoro import KModel
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


def export_module(model, output_path, args, input_names, output_names):
    torch.onnx.export(
        model,
        args=args,
        f=output_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f'exported {output_path}')


def build_input_ids(kmodel: KModel, token_bucket: int) -> torch.LongTensor:
    upper_bound = len(kmodel.vocab)
    return torch.randint(1, upper_bound, (1, token_bucket), dtype=torch.long)


def build_ref_s(style_dim: int) -> torch.FloatTensor:
    return torch.randn(1, style_dim * 2, dtype=torch.float32)


def build_alignment(token_bucket: int, frame_bucket: int) -> torch.FloatTensor:
    frame_positions = torch.arange(frame_bucket, dtype=torch.long)
    token_indices = torch.div(frame_positions * token_bucket, frame_bucket, rounding_mode='floor')
    token_indices = token_indices.clamp(max=token_bucket - 1)
    pred_aln_trg = torch.zeros((1, token_bucket, frame_bucket), dtype=torch.float32)
    pred_aln_trg[0, token_indices, frame_positions] = 1.0
    return pred_aln_trg


def build_en(feature_dim: int, frame_bucket: int) -> torch.FloatTensor:
    return torch.randn(1, feature_dim, frame_bucket, dtype=torch.float32)


def export_encoder_buckets(kmodel: KModel, output_dir: str, token_buckets: list[int]):
    model = KEncoderFeaturesForONNX(kmodel).eval()
    for token_bucket in token_buckets:
        input_ids = build_input_ids(kmodel, token_bucket)
        output_path = os.path.join(output_dir, f'encoder_token_{token_bucket}.onnx')
        export_module(
            model,
            output_path,
            args=(input_ids,),
            input_names=['input_ids'],
            output_names=['d_en'],
        )


def export_text_encoder_buckets(kmodel: KModel, output_dir: str, text_encoder_buckets: list[tuple[int, int]]):
    model = KTextEncoderForONNX(kmodel).eval()
    for token_bucket, frame_bucket in text_encoder_buckets:
        input_ids = build_input_ids(kmodel, token_bucket)
        pred_aln_trg = build_alignment(token_bucket, frame_bucket)
        output_path = os.path.join(output_dir, f'text_encoder_token_{token_bucket}_frame_{frame_bucket}.onnx')
        export_module(
            model,
            output_path,
            args=(input_ids, pred_aln_trg),
            input_names=['input_ids', 'pred_aln_trg'],
            output_names=['t_en', 'asr'],
        )


def export_f0n_buckets(kmodel: KModel, output_dir: str, frame_buckets: list[int]):
    shared_model = KF0NSharedForONNX(kmodel).eval()
    head_model = KF0NHeadForONNX(kmodel).eval()
    feature_dim = kmodel.predictor.shared.input_size
    ref_s = build_ref_s(kmodel.predictor.text_encoder.sty_dim)
    for frame_bucket in frame_buckets:
        en = build_en(feature_dim, frame_bucket)
        with torch.no_grad():
            shared = shared_model(en)

        shared_output_path = os.path.join(output_dir, f'f0n_shared_frame_{frame_bucket}.onnx')
        export_module(
            shared_model,
            shared_output_path,
            args=(en,),
            input_names=['en'],
            output_names=['shared'],
        )

        head_output_path = os.path.join(output_dir, f'f0n_head_frame_{frame_bucket}.onnx')
        export_module(
            head_model,
            head_output_path,
            args=(shared, ref_s),
            input_names=['shared', 'ref_s'],
            output_names=['F0_pred', 'N_pred'],
        )


def main():
    parser = argparse.ArgumentParser('Export fixed-bucket Kokoro frontend modules to ONNX', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='onnx_modules_static_frontend', help='output directory')
    parser.add_argument('--token_buckets', type=str, default=DEFAULT_TOKEN_BUCKETS, help='comma-separated token buckets for encoder export')
    parser.add_argument('--frame_buckets', type=str, default=DEFAULT_FRAME_BUCKETS, help='comma-separated frame buckets for f0n export')
    parser.add_argument('--text_encoder_buckets', type=str, default=DEFAULT_TEXT_ENCODER_BUCKETS, help='comma-separated token:frame buckets for text_encoder export')
    args = parser.parse_args()

    token_buckets = parse_bucket_list(args.token_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)

    os.makedirs(args.output_dir, exist_ok=True)

    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).eval()
    kmodel.bert.config._attn_implementation = 'eager'
    validate_token_buckets(kmodel, token_buckets, text_encoder_buckets)

    export_encoder_buckets(kmodel, args.output_dir, token_buckets)
    export_text_encoder_buckets(kmodel, args.output_dir, text_encoder_buckets)
    export_f0n_buckets(kmodel, args.output_dir, frame_buckets)


if __name__ == '__main__':
    main()