import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from kokoro import KModel, KPipeline
from kokoro.model import KStaticEncoderForONNX, KStaticF0NSharedForONNX, KStaticTextEncoderForONNX, KF0NHeadForONNX
from onnx_export_utils import (
    DEFAULT_DECODER_FRAME_BUCKET,
    DEFAULT_FRAME_BUCKETS,
    DEFAULT_TEXT_ENCODER_BUCKETS,
    DEFAULT_TOKEN_BUCKETS,
    build_har,
    build_padded_text_mask,
    create_session,
    load_phonemes_and_input_ids_torch,
    load_reference_style_torch,
    load_texts,
    parse_bucket_list,
    parse_text_encoder_buckets,
    right_pad_alignment,
    right_pad_last_dim,
    save_numpy_tensors,
    select_bucket,
    select_text_encoder_bucket,
    validate_token_buckets,
    write_metadata,
)


def should_export_frontend(modules: set[str]) -> bool:
    return bool(modules & {'all', 'frontend'})


def should_export_decoder(modules: set[str]) -> bool:
    return bool(modules & {'all', 'backend', 'decoder'})


def should_export_vocoder(modules: set[str]) -> bool:
    return bool(modules & {'all', 'backend', 'vocoder'})


def resolve_backend_padding_policy(args: argparse.Namespace) -> bool:
    if args.allow_backend_padding and args.require_exact_backend_bucket:
        raise ValueError('cannot set both --allow_backend_padding and --require_exact_backend_bucket')
    if args.require_exact_backend_bucket:
        return False
    return True


def run_static_npy_export(args: argparse.Namespace):
    token_buckets = parse_bucket_list(args.token_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)
    modules = {item.strip() for item in args.modules.split(',') if item.strip()}
    if not modules:
        raise ValueError('at least one module group must be selected')
    allow_backend_padding = resolve_backend_padding_policy(args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = Path(args.output_dir) / timestamp
    all_texts = load_texts(Path(args.text_file), None)

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    model.bert.config._attn_implementation = 'eager'
    validate_token_buckets(model, token_buckets, text_encoder_buckets)

    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]
    static_root = Path(args.static_onnx_dir)

    encoder_module = KStaticEncoderForONNX(model).eval()
    text_encoder_module = KStaticTextEncoderForONNX(model).eval()
    f0n_shared_module = KStaticF0NSharedForONNX(model).eval()
    f0n_head_module = KF0NHeadForONNX(model).eval()

    decoder_session = None
    vocoder_session = None
    if should_export_decoder(modules):
        decoder_session = create_session(static_root / 'decoder_front.onnx', providers)
    if should_export_vocoder(modules):
        vocoder_session = create_session(static_root / 'vocoder.onnx', providers)

    bucket_counts = defaultdict(int)
    metadata_records = []
    backend_requested = should_export_decoder(modules) or should_export_vocoder(modules)
    frontend_requested = should_export_frontend(modules)
    frontend_samples_written = 0

    for text in all_texts:
        need_frontend = frontend_requested and frontend_samples_written < args.sample_count
        need_decoder = should_export_decoder(modules) and bucket_counts['decoder_front'] == 0
        need_vocoder = should_export_vocoder(modules) and bucket_counts['vocoder'] == 0
        need_backend = need_decoder or need_vocoder
        if not need_frontend and not need_backend:
            break

        phonemes, input_ids = load_phonemes_and_input_ids_torch(pipeline, text)
        ref_s = load_reference_style_torch(pipeline, args.voice, len(phonemes))

        with torch.no_grad():
            d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
            _, _, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=args.speed)

        token_length = int(input_ids.shape[-1])
        frame_length = int(pred_aln_trg.shape[-1])
        try:
            encoder_bucket = select_bucket(token_length, token_buckets, 'encoder token')
            f0n_bucket = select_bucket(frame_length, frame_buckets, 'f0n frame')
            text_token_bucket, text_frame_bucket = select_text_encoder_bucket(token_length, frame_length, text_encoder_buckets)
        except ValueError as exc:
            print(f'skipped token_len={token_length}, frame_len={frame_length}: {exc}')
            continue

        compatible_encoder_buckets = [bucket for bucket in token_buckets if token_length <= bucket]
        compatible_f0n_buckets = [bucket for bucket in frame_buckets if frame_length <= bucket]
        compatible_text_encoder_buckets = [
            (token_bucket, frame_bucket)
            for token_bucket, frame_bucket in text_encoder_buckets
            if token_length <= token_bucket and frame_length <= frame_bucket
        ]

        input_lengths_np = np.array([token_length], dtype=np.int64)
        input_ids_np = input_ids.detach().cpu().numpy()
        pred_aln_trg_np = pred_aln_trg.detach().cpu().numpy().astype(np.float32)
        en_np = en.detach().cpu().numpy().astype(np.float32)
        ref_s_np = ref_s.detach().cpu().numpy().astype(np.float32)
        frame_lengths_np = np.array([frame_length], dtype=np.int64)
        padded_input_ids = np.zeros((1, encoder_bucket), dtype=np.int64)
        padded_input_ids[:, :token_length] = input_ids_np
        padded_text_mask = build_padded_text_mask(token_length, encoder_bucket)
        with torch.no_grad():
            encoder_d_en = encoder_module(
                torch.from_numpy(padded_input_ids).to(model.device),
                torch.from_numpy(input_lengths_np).to(model.device),
                torch.from_numpy(padded_text_mask).to(model.device),
            ).detach().cpu().numpy().astype(np.float32)

        padded_alignment = right_pad_alignment(pred_aln_trg_np, text_token_bucket, text_frame_bucket)
        padded_text_input_ids = np.zeros((1, text_token_bucket), dtype=np.int64)
        padded_text_input_ids[:, :token_length] = input_ids_np
        padded_text_encoder_mask = build_padded_text_mask(token_length, text_token_bucket)
        with torch.no_grad():
            t_en, asr = text_encoder_module(
                torch.from_numpy(padded_text_input_ids).to(model.device),
                torch.from_numpy(padded_alignment).to(model.device),
                torch.from_numpy(padded_text_encoder_mask).to(model.device),
            )
        t_en = t_en.detach().cpu().numpy().astype(np.float32)
        asr = asr.detach().cpu().numpy().astype(np.float32)

        padded_en = np.zeros((1, en.shape[1], f0n_bucket), dtype=np.float32)
        padded_en[:, :, :frame_length] = en_np
        with torch.no_grad():
            shared = f0n_shared_module(torch.from_numpy(padded_en).to(model.device))
            f0_pred, n_pred = f0n_head_module(shared, ref_s)
        shared = shared.detach().cpu().numpy().astype(np.float32)
        f0_pred = f0_pred.detach().cpu().numpy().astype(np.float32)
        n_pred = n_pred.detach().cpu().numpy().astype(np.float32)

        if need_frontend:
            for compatible_encoder_bucket in compatible_encoder_buckets:
                compatible_input_ids = np.zeros((1, compatible_encoder_bucket), dtype=np.int64)
                compatible_input_ids[:, :token_length] = input_ids_np
                compatible_text_mask = build_padded_text_mask(token_length, compatible_encoder_bucket)
                with torch.no_grad():
                    compatible_d_en = encoder_module(
                        torch.from_numpy(compatible_input_ids).to(model.device),
                        torch.from_numpy(input_lengths_np).to(model.device),
                        torch.from_numpy(compatible_text_mask).to(model.device),
                    ).detach().cpu().numpy().astype(np.float32)
                save_numpy_tensors(
                    export_root,
                    f'encoder_token_{compatible_encoder_bucket}',
                    bucket_counts[f'encoder_token_{compatible_encoder_bucket}'],
                    {
                        'input_ids': compatible_input_ids,
                        'text_mask': compatible_text_mask,
                        'd_en': compatible_d_en,
                    },
                )

            for compatible_text_token_bucket, compatible_text_frame_bucket in compatible_text_encoder_buckets:
                compatible_alignment = right_pad_alignment(pred_aln_trg_np, compatible_text_token_bucket, compatible_text_frame_bucket)
                compatible_text_input_ids = np.zeros((1, compatible_text_token_bucket), dtype=np.int64)
                compatible_text_input_ids[:, :token_length] = input_ids_np
                compatible_text_mask = build_padded_text_mask(token_length, compatible_text_token_bucket)
                with torch.no_grad():
                    compatible_t_en, compatible_asr = text_encoder_module(
                        torch.from_numpy(compatible_text_input_ids).to(model.device),
                        torch.from_numpy(compatible_alignment).to(model.device),
                        torch.from_numpy(compatible_text_mask).to(model.device),
                    )
                save_numpy_tensors(
                    export_root,
                    f'text_encoder_token_{compatible_text_token_bucket}_frame_{compatible_text_frame_bucket}',
                    bucket_counts[f'text_encoder_token_{compatible_text_token_bucket}_frame_{compatible_text_frame_bucket}'],
                    {
                        'input_ids': compatible_text_input_ids,
                        'pred_aln_trg': compatible_alignment,
                        'text_mask': compatible_text_mask,
                        't_en': compatible_t_en.detach().cpu().numpy().astype(np.float32),
                        'asr': compatible_asr.detach().cpu().numpy().astype(np.float32),
                    },
                )

            for compatible_f0n_bucket in compatible_f0n_buckets:
                compatible_en = np.zeros((1, en.shape[1], compatible_f0n_bucket), dtype=np.float32)
                compatible_en[:, :, :frame_length] = en_np
                with torch.no_grad():
                    compatible_shared = f0n_shared_module(torch.from_numpy(compatible_en).to(model.device))
                    compatible_f0_pred, compatible_n_pred = f0n_head_module(compatible_shared, ref_s)
                compatible_shared_np = compatible_shared.detach().cpu().numpy().astype(np.float32)
                save_numpy_tensors(
                    export_root,
                    f'f0n_shared_frame_{compatible_f0n_bucket}',
                    bucket_counts[f'f0n_shared_frame_{compatible_f0n_bucket}'],
                    {
                        'en': compatible_en,
                        'shared': compatible_shared_np,
                    },
                )
                save_numpy_tensors(
                    export_root,
                    f'f0n_head_frame_{compatible_f0n_bucket}',
                    bucket_counts[f'f0n_head_frame_{compatible_f0n_bucket}'],
                    {
                        'shared': compatible_shared_np,
                        'ref_s': ref_s_np,
                        'F0_pred': compatible_f0_pred.detach().cpu().numpy().astype(np.float32),
                        'N_pred': compatible_n_pred.detach().cpu().numpy().astype(np.float32),
                    },
                )
            frontend_samples_written += 1

        backend_exported = False
        backend_padding_used = False
        if need_backend and frame_length <= args.decoder_frame_bucket and (allow_backend_padding or frame_length == args.decoder_frame_bucket):
            timbre = ref_s[:, :128].detach().cpu().numpy().astype(np.float32)
            padded_asr = right_pad_last_dim(asr, args.decoder_frame_bucket)
            padded_f0 = right_pad_last_dim(f0_pred, args.decoder_frame_bucket * 2)
            padded_n = right_pad_last_dim(n_pred, args.decoder_frame_bucket * 2)
            backend_exported = True
            backend_padding_used = frame_length != args.decoder_frame_bucket

            if need_decoder:
                decoder_state = decoder_session.run(
                    None,
                    {
                        'asr': padded_asr,
                        'F0_pred': padded_f0,
                        'N_pred': padded_n,
                        'timbre': timbre,
                    },
                )[0].astype(np.float32)
                save_numpy_tensors(
                    export_root,
                    'decoder_front',
                    bucket_counts['decoder_front'],
                    {
                        'asr': padded_asr,
                        'F0_pred': padded_f0,
                        'N_pred': padded_n,
                        'timbre': timbre,
                        'decoder_state': decoder_state,
                    },
                )
            else:
                decoder_state = None

            if need_vocoder:
                if decoder_state is None:
                    decoder_state = decoder_session.run(
                        None,
                        {
                            'asr': padded_asr,
                            'F0_pred': padded_f0,
                            'N_pred': padded_n,
                            'timbre': timbre,
                        },
                    )[0].astype(np.float32)
                har = build_har(model.decoder.generator, padded_f0)
                waveform = vocoder_session.run(
                    None,
                    {
                        'decoder_state': decoder_state,
                        'timbre': timbre,
                        'har': har,
                    },
                )[0].astype(np.float32)
                save_numpy_tensors(
                    export_root,
                    'vocoder',
                    bucket_counts['vocoder'],
                    {
                        'decoder_state': decoder_state,
                        'timbre': timbre,
                        'har': har,
                        'waveform': waveform,
                    },
                )

        metadata_records.append(
            {
                'text': text,
                'phonemes': phonemes,
                'token_length': token_length,
                'frame_length': frame_length,
                'encoder_bucket': encoder_bucket,
                'compatible_encoder_buckets': compatible_encoder_buckets,
                'text_encoder_bucket': [text_token_bucket, text_frame_bucket],
                'compatible_text_encoder_buckets': [list(bucket) for bucket in compatible_text_encoder_buckets],
                'f0n_bucket': f0n_bucket,
                'compatible_f0n_buckets': compatible_f0n_buckets,
                'backend_exported': backend_exported,
                'backend_padding_used': backend_padding_used,
            }
        )

        if need_frontend:
            for compatible_encoder_bucket in compatible_encoder_buckets:
                bucket_counts[f'encoder_token_{compatible_encoder_bucket}'] += 1
            for compatible_text_token_bucket, compatible_text_frame_bucket in compatible_text_encoder_buckets:
                bucket_counts[f'text_encoder_token_{compatible_text_token_bucket}_frame_{compatible_text_frame_bucket}'] += 1
            for compatible_f0n_bucket in compatible_f0n_buckets:
                bucket_counts[f'f0n_shared_frame_{compatible_f0n_bucket}'] += 1
                bucket_counts[f'f0n_head_frame_{compatible_f0n_bucket}'] += 1
        if backend_exported and should_export_decoder(modules):
            bucket_counts['decoder_front'] += 1
        if backend_exported and should_export_vocoder(modules):
            bucket_counts['vocoder'] += 1

        print(
            f'saved token_len={token_length}, frame_len={frame_length} '
            f'-> encoder={encoder_bucket}, text_encoder=({text_token_bucket},{text_frame_bucket}), '
            f'f0n={f0n_bucket}, backend={backend_exported}, backend_padding={backend_padding_used} '
            f'for: {text[:80]}'
        )

    if backend_requested:
        exported_decoder = bucket_counts['decoder_front']
        exported_vocoder = bucket_counts['vocoder']
        if should_export_decoder(modules) and exported_decoder == 0:
            raise ValueError(
                f'no decoder calibration samples were exported for decoder_frame_bucket={args.decoder_frame_bucket}. '
                'Provide shorter text, increase the bucket, or allow padded backend export.'
            )

    if frontend_requested and frontend_samples_written < args.sample_count:
        raise ValueError(
            f'only wrote {frontend_samples_written} frontend samples, but sample_count={args.sample_count}. '
            f'Provide more non-empty lines in {args.text_file}.'
        )
        if should_export_vocoder(modules) and exported_vocoder == 0:
            raise ValueError(
                f'no vocoder calibration samples were exported for decoder_frame_bucket={args.decoder_frame_bucket}. '
                'Provide shorter text, increase the bucket, or allow padded backend export.'
            )

    write_metadata(export_root / 'metadata.json', metadata_records)
    print(f'export root: {export_root}')


def main():
    parser = argparse.ArgumentParser('Export static ONNX representative NPY tensors for NPU quantization', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--lang_code', '-l', type=str, default='a', help='pipeline language code')
    parser.add_argument('--voice', '-v', type=str, default='af_heart', help='voice id or .pt path')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='speech speed')
    parser.add_argument('--sample_count', '-n', type=int, default=10, help='number of samples to export')
    parser.add_argument('--text_file', '-t', type=str, default='demo/en.txt', help='text file with one utterance per line')
    parser.add_argument('--output_dir', '-o', type=str, default='static_onnx_npy', help='directory where timestamped exports are written')
    parser.add_argument('--static_onnx_dir', type=str, default='onnx_modules_static_frontend', help='directory with static ONNX modules')
    parser.add_argument('--token_buckets', type=str, default=DEFAULT_TOKEN_BUCKETS, help='comma-separated token buckets for encoder export')
    parser.add_argument('--frame_buckets', type=str, default=DEFAULT_FRAME_BUCKETS, help='comma-separated frame buckets for f0n export')
    parser.add_argument('--text_encoder_buckets', type=str, default=DEFAULT_TEXT_ENCODER_BUCKETS, help='comma-separated token:frame buckets for text_encoder export')
    parser.add_argument('--decoder_frame_bucket', type=int, default=DEFAULT_DECODER_FRAME_BUCKET, help='frame bucket for decoder_front and vocoder calibration data')
    parser.add_argument('--modules', type=str, default='all', help='comma-separated groups: all, frontend, backend, decoder, vocoder')
    parser.add_argument('--allow_backend_padding', action='store_true', help='deprecated compatibility flag; padded backend export is enabled by default unless strict mode is requested')
    parser.add_argument('--require_exact_backend_bucket', action='store_true', help='export decoder_front and vocoder only when frame_length matches decoder_frame_bucket exactly')
    parser.add_argument('--providers', type=str, default='CPUExecutionProvider', help='comma-separated ONNX Runtime providers in priority order')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run the PyTorch reference pipeline on')
    args = parser.parse_args()
    run_static_npy_export(args)


if __name__ == '__main__':
    main()