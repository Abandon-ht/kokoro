import argparse
import os

import onnx
import torch
import torch.nn.functional as F

from kokoro import KModel
from kokoro.model import (
    KDurationPredictorForONNX,
    KEncoderForONNX,
    KF0NPredictorForONNX,
    KModelForONNX,
    KTextEncoderForONNX,
)


STATIC_ASR_LEN = 198
STATIC_PITCH_LEN = STATIC_ASR_LEN * 2


class KDecoderFrontForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder

    def forward(
        self,
        asr: torch.FloatTensor,
        F0_pred: torch.FloatTensor,
        N_pred: torch.FloatTensor,
        timbre: torch.FloatTensor,
    ) -> torch.FloatTensor:
        F0 = self.decoder.F0_conv(F0_pred.unsqueeze(1))
        N = self.decoder.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.decoder.encode(x, timbre)
        asr_res = self.decoder.asr_res(asr)
        res = True
        for block in self.decoder.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, timbre)
            if block.upsample_type != 'none':
                res = False
        return x


class KVocoderForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator

    def forward(
        self,
        x: torch.FloatTensor,
        timbre: torch.FloatTensor,
        har: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
    print(f"exported {output_path}")


def resize_last_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = tensor.shape[-1]
    if current_len == target_len:
        return tensor

    resized = tensor.new_zeros(*tensor.shape[:-1], target_len)
    copy_len = min(current_len, target_len)
    resized[..., :copy_len] = tensor[..., :copy_len]
    return resized


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
        timbre = ref_s[:, :128]

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
        'timbre': timbre,
        'frame_len': pred_aln_trg.shape[-1],
    }


def build_static_decoder_samples(kmodel: KModel, samples):
    asr = resize_last_dim(samples['asr'], STATIC_ASR_LEN)
    F0_pred = resize_last_dim(samples['F0_pred'], STATIC_PITCH_LEN)
    N_pred = resize_last_dim(samples['N_pred'], STATIC_PITCH_LEN)
    timbre = samples['timbre']

    decoder_front = KDecoderFrontForONNX(kmodel).eval()
    generator = kmodel.decoder.generator

    with torch.no_grad():
        decoder_state = decoder_front(asr, F0_pred, N_pred, timbre)
        f0 = generator.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

    return {
        'asr': asr,
        'F0_pred': F0_pred,
        'N_pred': N_pred,
        'timbre': timbre,
        'decoder_state': decoder_state,
        'har': har,
    }


def export_all_modules(kmodel: KModel, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    samples = build_sample_inputs(kmodel)
    static_decoder_samples = build_static_decoder_samples(kmodel, samples)

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
        KDecoderFrontForONNX(kmodel).eval(),
        os.path.join(output_dir, 'decoder_front.onnx'),
        args=(
            static_decoder_samples['asr'],
            static_decoder_samples['F0_pred'],
            static_decoder_samples['N_pred'],
            static_decoder_samples['timbre'],
        ),
        input_names=['asr', 'F0_pred', 'N_pred', 'timbre'],
        output_names=['decoder_state'],
    )

    export_module(
        KVocoderForONNX(kmodel).eval(),
        os.path.join(output_dir, 'vocoder.onnx'),
        args=(
            static_decoder_samples['decoder_state'],
            static_decoder_samples['timbre'],
            static_decoder_samples['har'],
        ),
        input_names=['decoder_state', 'timbre', 'har'],
        output_names=['waveform'],
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