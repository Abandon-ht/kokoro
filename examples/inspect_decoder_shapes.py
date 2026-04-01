import argparse
import math

import torch

from kokoro import KModel, KPipeline


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


def inspect_decoder(
    model: KModel,
    asr: torch.FloatTensor,
    f0_pred: torch.FloatTensor,
    n_pred: torch.FloatTensor,
    ref_s: torch.FloatTensor,
):
    decoder = model.decoder
    timbre = ref_s[:, :128]

    print('\n[decoder] input tensors')
    print(f'  asr      : {tuple(asr.shape)}')
    print(f'  F0_pred  : {tuple(f0_pred.shape)}')
    print(f'  N_pred   : {tuple(n_pred.shape)}')
    print(f'  ref_s    : {tuple(ref_s.shape)}')

    f0 = decoder.F0_conv(f0_pred.unsqueeze(1))
    n = decoder.N_conv(n_pred.unsqueeze(1))
    print('\n[decoder] after stride-2 conv')
    print(f'  F0_conv  : {tuple(f0.shape)}')
    print(f'  N_conv   : {tuple(n.shape)}')

    x = torch.cat([asr, f0, n], axis=1)
    print(f'  concat   : {tuple(x.shape)}')

    x = decoder.encode(x, timbre)
    print(f'  encode   : {tuple(x.shape)}')

    asr_res = decoder.asr_res(asr)
    print(f'  asr_res  : {tuple(asr_res.shape)}')

    res = True
    for index, block in enumerate(decoder.decode):
        if res:
            x = torch.cat([x, asr_res, f0, n], axis=1)
            print(f'  block{index}_in : {tuple(x.shape)}')
        x = block(x, timbre)
        print(f'  block{index}_out: {tuple(x.shape)} upsample={block.upsample_type}')
        if block.upsample_type != 'none':
            res = False

    generator = decoder.generator
    upsample_product = math.prod(layer.stride[0] for layer in generator.ups)
    hop_length = generator.stft.hop_length
    print('\n[generator] static factors')
    print(f'  convtranspose product : {upsample_product}')
    print(f'  istft hop_length      : {hop_length}')
    print(f'  source upsample scale : {generator.m_source.l_sin_gen.upsample_scale}')

    waveform = generator(x, timbre, f0_pred)
    print('\n[generator] output')
    print(f'  waveform : {tuple(waveform.shape)}')

    asr_len = asr.shape[-1]
    f0_len = f0_pred.shape[-1]
    waveform_len = waveform.shape[-1]
    print('\n[length ratios]')
    print(f'  F0_pred / asr      = {f0_len} / {asr_len} = {f0_len / asr_len:.4f}')
    print(f'  waveform / asr     = {waveform_len} / {asr_len} = {waveform_len / asr_len:.4f}')
    print(f'  waveform / F0_pred = {waveform_len} / {f0_len} = {waveform_len / f0_len:.4f}')


def main():
    parser = argparse.ArgumentParser('Inspect Kokoro decoder shape relationships', add_help=True)
    parser.add_argument('--config_file', '-c', type=str, default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', '-p', type=str, default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--lang_code', '-l', type=str, default='a', help='pipeline language code')
    parser.add_argument('--voice', '-v', type=str, default='af_heart', help='voice id or .pt path')
    parser.add_argument('--text', '-t', type=str, default='The sky above the port was the color of television, tuned to a dead channel.', help='text to inspect')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device to run on')
    args = parser.parse_args()

    model = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True).to(args.device).eval()
    pipeline = KPipeline(lang_code=args.lang_code, model=model, device=args.device)

    phonemes, input_ids = load_input_ids(pipeline, args.text)
    ref_s = load_ref_s(pipeline, args.voice, len(phonemes))

    print('[inputs]')
    print(f'  text           : {args.text}')
    print(f'  phonemes       : {phonemes}')
    print(f'  phoneme length : {len(phonemes)}')
    print(f'  input_ids      : {tuple(input_ids.shape)}')
    print(f'  ref_s          : {tuple(ref_s.shape)}')

    with torch.no_grad():
        d_en, input_lengths, text_mask = model._encode_linguistic_tokens(input_ids)
        d, pred_dur, pred_aln_trg, en = model._predict_alignment(d_en, ref_s, input_lengths, text_mask, speed=1.0)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(en, ref_s[:, 128:])
        waveform = model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()

    print('\n[front-end outputs]')
    print(f'  d_en           : {tuple(d_en.shape)}')
    print(f'  input_lengths  : {tuple(input_lengths.shape)} value={input_lengths.tolist()}')
    print(f'  text_mask      : {tuple(text_mask.shape)}')
    print(f'  d              : {tuple(d.shape)}')
    print(f'  pred_dur       : {tuple(pred_dur.shape)} sum={int(pred_dur.sum().item())}')
    print(f'  pred_aln_trg   : {tuple(pred_aln_trg.shape)}')
    print(f'  en             : {tuple(en.shape)}')
    print(f'  t_en           : {tuple(t_en.shape)}')
    print(f'  asr            : {tuple(asr.shape)}')
    print(f'  F0_pred        : {tuple(f0_pred.shape)}')
    print(f'  N_pred         : {tuple(n_pred.shape)}')
    print(f'  waveform       : {tuple(waveform.shape)}')

    inspect_decoder(model, asr, f0_pred, n_pred, ref_s)


if __name__ == '__main__':
    main()
