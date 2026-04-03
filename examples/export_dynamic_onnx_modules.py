import argparse
import os

from kokoro import KModel
from kokoro.model import (
    KDurationPredictorForONNX,
    KEncoderForONNX,
    KF0NPredictorForONNX,
    KModelForONNX,
    KTextEncoderForONNX,
)
from onnx_export_utils import build_dynamic_sample_inputs, export_module


def export_dynamic_modules(kmodel: KModel, output_dir: str, include_full_model: bool):
    os.makedirs(output_dir, exist_ok=True)
    samples = build_dynamic_sample_inputs(kmodel)

    export_module(
        KEncoderForONNX(kmodel).eval(),
        os.path.join(output_dir, 'encoder.onnx'),
        args=(samples['input_ids'],),
        input_names=['input_ids'],
        output_names=['d_en', 'input_lengths', 'text_mask'],
        dynamic_axes={
            'input_ids': {1: 'token_len'},
            'd_en': {2: 'token_len'},
            'input_lengths': {0: 'batch_size'},
            'text_mask': {1: 'token_len'},
        },
    )

    export_module(
        KDurationPredictorForONNX(kmodel).eval(),
        os.path.join(output_dir, 'duration_predictor.onnx'),
        args=(samples['d_en'], samples['ref_s'], samples['input_lengths'], samples['text_mask'], samples['speed']),
        input_names=['d_en', 'ref_s', 'input_lengths', 'text_mask', 'speed'],
        output_names=['d', 'pred_dur', 'pred_aln_trg', 'en'],
        dynamic_axes={
            'd_en': {2: 'token_len'},
            'input_lengths': {0: 'batch_size'},
            'text_mask': {1: 'token_len'},
            'd': {1: 'token_len'},
            'pred_dur': {0: 'token_len'},
            'pred_aln_trg': {1: 'token_len', 2: 'frame_len'},
            'en': {2: 'frame_len'},
        },
    )

    export_module(
        KF0NPredictorForONNX(kmodel).eval(),
        os.path.join(output_dir, 'f0n_predictor.onnx'),
        args=(samples['en'], samples['ref_s']),
        input_names=['en', 'ref_s'],
        output_names=['F0_pred', 'N_pred'],
        dynamic_axes={
            'en': {2: 'frame_len'},
            'F0_pred': {1: 'frame_len'},
            'N_pred': {1: 'frame_len'},
        },
    )

    export_module(
        KTextEncoderForONNX(kmodel).eval(),
        os.path.join(output_dir, 'text_encoder.onnx'),
        args=(samples['input_ids'], samples['pred_aln_trg']),
        input_names=['input_ids', 'pred_aln_trg'],
        output_names=['t_en', 'asr'],
        dynamic_axes={
            'input_ids': {1: 'token_len'},
            'pred_aln_trg': {1: 'token_len', 2: 'frame_len'},
            't_en': {2: 'token_len'},
            'asr': {2: 'frame_len'},
        },
    )

    if include_full_model:
        export_module(
            KModelForONNX(kmodel).eval(),
            os.path.join(output_dir, 'kokoro.onnx'),
            args=(samples['input_ids'], samples['ref_s'], samples['speed']),
            input_names=['input_ids', 'ref_s', 'speed'],
            output_names=['waveform', 'duration'],
            dynamic_axes={
                'input_ids': {1: 'token_len'},
                'waveform': {1: 'sample_len'},
                'duration': {0: 'token_len'},
            },
        )


def main():
    parser = argparse.ArgumentParser('Export dynamic Kokoro ONNX modules', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='onnx_modules', help='dynamic ONNX output directory')
    parser.add_argument('--skip_full_model', action='store_true', help='skip exporting kokoro.onnx')
    args = parser.parse_args()

    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).eval()
    kmodel.bert.config._attn_implementation = 'eager'
    export_dynamic_modules(kmodel, args.output_dir, include_full_model=not args.skip_full_model)


if __name__ == '__main__':
    main()