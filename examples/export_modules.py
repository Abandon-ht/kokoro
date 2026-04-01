import argparse
import os

import onnx
import torch

from kokoro import KModel
from kokoro.model import (
    KDecoderForONNX,
    KDurationPredictorForONNX,
    KEncoderForONNX,
    KF0NPredictorForONNX,
    KModelForONNX,
    KTextEncoderForONNX,
)


def export_module(model, output_path, args, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        model,
        args=args,
        f=output_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"exported {output_path}")


def build_sample_inputs(kmodel: KModel):
    seq_len = min(50, kmodel.context_length)
    hidden_dim = kmodel.bert_encoder.out_features
    input_ids = torch.randint(1, len(kmodel.vocab), (1, seq_len), dtype=torch.long)
    ref_s = torch.randn(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        d_en, input_lengths, text_mask = kmodel._encode_linguistic_tokens(input_ids)
        _, pred_dur, pred_aln_trg, en = kmodel._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed)
        F0_pred, N_pred = kmodel.predictor.F0Ntrain(en, ref_s[:, 128:])
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

    return {
        'hidden_dim': hidden_dim,
        'input_ids': input_ids,
        'ref_s': ref_s,
        'speed': speed,
        'd_en': d_en,
        'input_lengths': input_lengths,
        'text_mask': text_mask,
        'pred_aln_trg': pred_aln_trg,
        'en': en,
        'F0_pred': F0_pred,
        'N_pred': N_pred,
        't_en': t_en,
        'asr': asr,
        'frame_len': pred_aln_trg.shape[-1],
    }


def export_all_modules(kmodel: KModel, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    samples = build_sample_inputs(kmodel)

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

    export_module(
        KDecoderForONNX(kmodel).eval(),
        os.path.join(output_dir, 'decoder.onnx'),
        args=(samples['asr'], samples['F0_pred'], samples['N_pred'], samples['ref_s']),
        input_names=['asr', 'F0_pred', 'N_pred', 'ref_s'],
        output_names=['waveform'],
        dynamic_axes={
            'asr': {2: 'frame_len'},
            'F0_pred': {1: 'frame_len'},
            'N_pred': {1: 'frame_len'},
            'waveform': {1: 'sample_len'},
        },
    )

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
    parser = argparse.ArgumentParser('Export Kokoro modules to ONNX', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='onnx_modules', help='output directory')
    args = parser.parse_args()

    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).eval()
    kmodel.bert.config._attn_implementation = 'eager'
    export_all_modules(kmodel, args.output_dir)


if __name__ == '__main__':
    main()