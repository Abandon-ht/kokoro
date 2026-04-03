import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from kokoro import KModel, KPipeline
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


def run_static_npy_export(args: argparse.Namespace):
    token_buckets = parse_bucket_list(args.token_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)
    modules = {item.strip() for item in args.modules.split(',') if item.strip()}
    if not modules:
        raise ValueError('at least one module group must be selected')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = Path(args.output_dir) / timestamp
    texts = load_texts(Path(args.text_file), args.sample_count)

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    model.bert.config._attn_implementation = 'eager'
    validate_token_buckets(model, token_buckets, text_encoder_buckets)

    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]
    static_root = Path(args.static_onnx_dir)

    encoder_sessions = {
        bucket: create_session(static_root / f'encoder_token_{bucket}.onnx', providers)
        for bucket in token_buckets
    }
    text_encoder_sessions = {
        bucket_pair: create_session(static_root / f'text_encoder_token_{bucket_pair[0]}_frame_{bucket_pair[1]}.onnx', providers)
        for bucket_pair in text_encoder_buckets
    }
    f0n_shared_sessions = {
        bucket: create_session(static_root / f'f0n_shared_frame_{bucket}.onnx', providers)
        for bucket in frame_buckets
    }
    f0n_head_sessions = {
        bucket: create_session(static_root / f'f0n_head_frame_{bucket}.onnx', providers)
        for bucket in frame_buckets
    }

    decoder_session = None
    vocoder_session = None
    if should_export_decoder(modules):
        decoder_session = create_session(static_root / 'decoder_front.onnx', providers)
    if should_export_vocoder(modules):
        vocoder_session = create_session(static_root / 'vocoder.onnx', providers)

    bucket_counts = defaultdict(int)
    metadata_records = []

    for text in texts:
        phonemes, input_ids = load_phonemes_and_input_ids_torch(pipeline, text)
        ref_s = load_reference_style_torch(pipeline, args.voice, len(phonemes))

        with torch.no_grad():
            d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
            _, _, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=args.speed)

        token_length = int(input_ids.shape[-1])
        frame_length = int(pred_aln_trg.shape[-1])
        encoder_bucket = select_bucket(token_length, token_buckets, 'encoder token')
        f0n_bucket = select_bucket(frame_length, frame_buckets, 'f0n frame')
        text_token_bucket, text_frame_bucket = select_text_encoder_bucket(token_length, frame_length, text_encoder_buckets)

        padded_input_ids = np.zeros((1, encoder_bucket), dtype=np.int64)
        padded_input_ids[:, :token_length] = input_ids.detach().cpu().numpy()
        padded_text_mask = build_padded_text_mask(token_length, encoder_bucket)
        input_lengths_np = np.array([token_length], dtype=np.int64)
        encoder_outputs = encoder_sessions[encoder_bucket].run(
            None,
            {
                'input_ids': padded_input_ids,
                'input_lengths': input_lengths_np,
                'text_mask': padded_text_mask,
            },
        )
        encoder_d_en = encoder_outputs[0].astype(np.float32)

        padded_alignment = right_pad_alignment(
            pred_aln_trg.detach().cpu().numpy().astype(np.float32),
            text_token_bucket,
            text_frame_bucket,
        )
        padded_text_input_ids = np.zeros((1, text_token_bucket), dtype=np.int64)
        padded_text_input_ids[:, :token_length] = input_ids.detach().cpu().numpy()
        padded_text_encoder_mask = build_padded_text_mask(token_length, text_token_bucket)
        text_encoder_outputs = text_encoder_sessions[(text_token_bucket, text_frame_bucket)].run(
            None,
            {
                'input_ids': padded_text_input_ids,
                'pred_aln_trg': padded_alignment,
                'input_lengths': input_lengths_np,
                'text_mask': padded_text_encoder_mask,
            },
        )
        t_en = text_encoder_outputs[0].astype(np.float32)
        asr = text_encoder_outputs[1].astype(np.float32)

        padded_en = np.zeros((1, en.shape[1], f0n_bucket), dtype=np.float32)
        padded_en[:, :, :frame_length] = en.detach().cpu().numpy().astype(np.float32)
        frame_lengths_np = np.array([frame_length], dtype=np.int64)
        shared = f0n_shared_sessions[f0n_bucket].run(None, {'en': padded_en, 'frame_lengths': frame_lengths_np})[0].astype(np.float32)
        f0_pred, n_pred = f0n_head_sessions[f0n_bucket].run(
            None,
            {
                'shared': shared,
                'ref_s': ref_s.detach().cpu().numpy().astype(np.float32),
            },
        )
        f0_pred = f0_pred.astype(np.float32)
        n_pred = n_pred.astype(np.float32)

        sample_index = bucket_counts[f'encoder_token_{encoder_bucket}']

        if should_export_frontend(modules):
            save_numpy_tensors(
                export_root,
                f'encoder_token_{encoder_bucket}',
                sample_index,
                {
                    'input_ids': padded_input_ids,
                    'input_lengths': input_lengths_np,
                    'text_mask': padded_text_mask,
                    'd_en': encoder_d_en,
                },
            )
            save_numpy_tensors(
                export_root,
                f'text_encoder_token_{text_token_bucket}_frame_{text_frame_bucket}',
                bucket_counts[f'text_encoder_token_{text_token_bucket}_frame_{text_frame_bucket}'],
                {
                    'input_ids': padded_text_input_ids,
                    'pred_aln_trg': padded_alignment,
                    'input_lengths': input_lengths_np,
                    'text_mask': padded_text_encoder_mask,
                    't_en': t_en,
                    'asr': asr,
                },
            )
            save_numpy_tensors(
                export_root,
                f'f0n_shared_frame_{f0n_bucket}',
                bucket_counts[f'f0n_shared_frame_{f0n_bucket}'],
                {
                    'en': padded_en,
                    'frame_lengths': frame_lengths_np,
                    'shared': shared,
                },
            )
            save_numpy_tensors(
                export_root,
                f'f0n_head_frame_{f0n_bucket}',
                bucket_counts[f'f0n_head_frame_{f0n_bucket}'],
                {
                    'shared': shared,
                    'ref_s': ref_s.detach().cpu().numpy().astype(np.float32),
                    'F0_pred': f0_pred,
                    'N_pred': n_pred,
                },
            )

        backend_exported = False
        if frame_length <= args.decoder_frame_bucket and (args.allow_backend_padding or frame_length == args.decoder_frame_bucket):
            timbre = ref_s[:, :128].detach().cpu().numpy().astype(np.float32)
            padded_asr = right_pad_last_dim(asr, args.decoder_frame_bucket)
            padded_f0 = right_pad_last_dim(f0_pred, args.decoder_frame_bucket * 2)
            padded_n = right_pad_last_dim(n_pred, args.decoder_frame_bucket * 2)
            backend_exported = True

            if should_export_decoder(modules):
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

            if should_export_vocoder(modules):
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
                'text_encoder_bucket': [text_token_bucket, text_frame_bucket],
                'f0n_bucket': f0n_bucket,
                'backend_exported': backend_exported,
            }
        )

        bucket_counts[f'encoder_token_{encoder_bucket}'] += 1
        bucket_counts[f'text_encoder_token_{text_token_bucket}_frame_{text_frame_bucket}'] += 1
        bucket_counts[f'f0n_shared_frame_{f0n_bucket}'] += 1
        bucket_counts[f'f0n_head_frame_{f0n_bucket}'] += 1
        if backend_exported and should_export_decoder(modules):
            bucket_counts['decoder_front'] += 1
        if backend_exported and should_export_vocoder(modules):
            bucket_counts['vocoder'] += 1

        print(
            f'saved token_len={token_length}, frame_len={frame_length} '
            f'-> encoder={encoder_bucket}, text_encoder=({text_token_bucket},{text_frame_bucket}), '
            f'f0n={f0n_bucket}, backend={backend_exported} for: {text[:80]}'
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
    parser.add_argument('--allow_backend_padding', action='store_true', help='allow padded short samples to be exported for decoder_front and vocoder')
    parser.add_argument('--providers', type=str, default='CPUExecutionProvider', help='comma-separated ONNX Runtime providers in priority order')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run the PyTorch reference pipeline on')
    args = parser.parse_args()
    run_static_npy_export(args)


if __name__ == '__main__':
    main()