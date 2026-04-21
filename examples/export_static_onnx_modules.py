import argparse
import os

import torch

from kokoro import KModel
from kokoro.model import KStaticEncoderForONNX, KStaticF0NSharedForONNX, KStaticTextEncoderForONNX, KF0NHeadForONNX
from onnx_export_utils import (
    DEFAULT_DECODER_FRAME_BUCKET,
    DEFAULT_FRAME_BUCKETS,
    DEFAULT_TEXT_ENCODER_BUCKETS,
    DEFAULT_TOKEN_BUCKETS,
    KDecoderFrontForONNX,
    KVocoderCoreForONNX,
    KVocoderTailForONNX,
    build_alignment,
    build_en,
    build_input_ids,
    build_input_lengths,
    build_ref_s,
    build_static_backend_samples,
    build_text_mask,
    export_module,
    parse_bucket_list,
    parse_text_encoder_buckets,
    validate_token_buckets,
)


def export_static_encoder_buckets(kmodel: KModel, output_dir: str, token_buckets: list[int]):
    model = KStaticEncoderForONNX(kmodel).eval()
    for token_bucket in token_buckets:
        input_ids = build_input_ids(kmodel, token_bucket)
        input_lengths = build_input_lengths(token_bucket)
        text_mask = build_text_mask(token_bucket, token_bucket)
        output_path = os.path.join(output_dir, f'encoder_token_{token_bucket}.onnx')
        export_module(
            model,
            output_path,
            args=(input_ids, input_lengths, text_mask),
            input_names=['input_ids', 'input_lengths', 'text_mask'],
            output_names=['d_en'],
        )


def export_static_text_encoder_buckets(kmodel: KModel, output_dir: str, text_encoder_buckets: list[tuple[int, int]]):
    model = KStaticTextEncoderForONNX(kmodel).eval()
    for token_bucket, frame_bucket in text_encoder_buckets:
        input_ids = build_input_ids(kmodel, token_bucket)
        pred_aln_trg = build_alignment(token_bucket, frame_bucket)
        text_mask = build_text_mask(token_bucket, token_bucket)
        output_path = os.path.join(output_dir, f'text_encoder_token_{token_bucket}_frame_{frame_bucket}.onnx')
        export_module(
            model,
            output_path,
            args=(input_ids, pred_aln_trg, text_mask),
            input_names=['input_ids', 'pred_aln_trg', 'text_mask'],
            output_names=['t_en', 'asr'],
        )


def export_static_f0n_buckets(kmodel: KModel, output_dir: str, frame_buckets: list[int]):
    shared_model = KStaticF0NSharedForONNX(kmodel).eval()
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


def export_static_backend(kmodel: KModel, output_dir: str, decoder_frame_bucket: int):
    samples = build_static_backend_samples(kmodel, decoder_frame_bucket)

    export_module(
        KDecoderFrontForONNX(kmodel).eval(),
        os.path.join(output_dir, 'decoder_front.onnx'),
        args=(samples['asr'], samples['F0_pred'], samples['N_pred'], samples['timbre']),
        input_names=['asr', 'F0_pred', 'N_pred', 'timbre'],
        output_names=['decoder_state'],
    )

    export_module(
        KVocoderCoreForONNX(kmodel).eval(),
        os.path.join(output_dir, 'vocoder_core.onnx'),
        args=(samples['decoder_state'], samples['timbre'], samples['har']),
        input_names=['decoder_state', 'timbre', 'har'],
        output_names=['vocoder_hidden'],
    )

    export_module(
        KVocoderTailForONNX(kmodel).eval(),
        os.path.join(output_dir, 'vocoder_tail.onnx'),
        args=(samples['vocoder_hidden'],),
        input_names=['vocoder_hidden'],
        output_names=['waveform'],
    )


def main():
    parser = argparse.ArgumentParser('Export fixed-bucket static Kokoro ONNX modules', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_1-zh.pth', help='path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='onnx_modules_static_frontend', help='static ONNX output directory')
    parser.add_argument('--token_buckets', type=str, default=DEFAULT_TOKEN_BUCKETS, help='comma-separated token buckets for encoder export')
    parser.add_argument('--frame_buckets', type=str, default=DEFAULT_FRAME_BUCKETS, help='comma-separated frame buckets for f0n export')
    parser.add_argument('--text_encoder_buckets', type=str, default=DEFAULT_TEXT_ENCODER_BUCKETS, help='comma-separated token:frame buckets for text_encoder export')
    parser.add_argument('--decoder_frame_bucket', type=int, default=DEFAULT_DECODER_FRAME_BUCKET, help='frame bucket for static decoder and vocoder export')
    args = parser.parse_args()

    token_buckets = parse_bucket_list(args.token_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)

    os.makedirs(args.output_dir, exist_ok=True)

    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).eval()
    kmodel.bert.config._attn_implementation = 'eager'
    validate_token_buckets(kmodel, token_buckets, text_encoder_buckets)

    export_static_encoder_buckets(kmodel, args.output_dir, token_buckets)
    export_static_text_encoder_buckets(kmodel, args.output_dir, text_encoder_buckets)
    export_static_f0n_buckets(kmodel, args.output_dir, frame_buckets)
    export_static_backend(kmodel, args.output_dir, args.decoder_frame_bucket)


if __name__ == '__main__':
    main()