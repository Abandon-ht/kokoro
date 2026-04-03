import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from kokoro import KModel, KPipeline


FRAME_BUCKET = 198
PITCH_BUCKET = FRAME_BUCKET * 2
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 600


def load_phonemes_and_input_ids(pipeline: KPipeline, text: str) -> tuple[str, np.ndarray]:
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

    input_id_list = [0]
    for phoneme_char in phonemes:
        token_id = pipeline.model.vocab.get(phoneme_char)
        if token_id is not None:
            input_id_list.append(token_id)
    input_id_list.append(0)

    input_ids = np.asarray([input_id_list], dtype=np.int64)
    return phonemes, input_ids


def load_reference_style(pipeline: KPipeline, voice: str, phoneme_length: int) -> np.ndarray:
    pack = pipeline.load_voice(voice).cpu()
    ref_s = pack[phoneme_length - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s.numpy().astype(np.float32)


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


def synthesize_with_dynamic_frontend_static_backend(
    text: str,
    voice: str,
    lang_code: str,
    speed: float,
    repo_id: str,
    config_file: str,
    checkpoint_path: str,
    onnx_dir: str,
    static_onnx_dir: str,
    output_path: str,
    providers: list[str],
) -> dict[str, int | str]:
    model = KModel(
        repo_id=repo_id,
        config=config_file,
        model=checkpoint_path,
        disable_complex=True,
    ).eval()
    model.bert.config._attn_implementation = 'eager'
    pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id, model=model, device='cpu')

    phonemes, input_ids = load_phonemes_and_input_ids(pipeline, text)
    ref_s = load_reference_style(pipeline, voice, len(phonemes))

    onnx_root = Path(onnx_dir)
    static_root = Path(static_onnx_dir)

    encoder_session = create_session(onnx_root / 'encoder.onnx', providers)
    duration_session = create_session(onnx_root / 'duration_predictor.onnx', providers)
    text_encoder_session = create_session(onnx_root / 'text_encoder.onnx', providers)
    f0n_session = create_session(onnx_root / 'f0n_predictor.onnx', providers)
    decoder_session = create_session(static_root / 'decoder_front.onnx', providers)
    vocoder_session = create_session(static_root / 'vocoder.onnx', providers)

    d_en, input_lengths, text_mask = encoder_session.run(None, {'input_ids': input_ids})
    duration_outputs = duration_session.run(
        None,
        {
            'd_en': d_en,
            'ref_s': ref_s,
            'input_lengths': input_lengths,
            'text_mask': text_mask,
            'speed': np.asarray([speed], dtype=np.float32),
        },
    )
    pred_dur = duration_outputs[1]
    pred_aln_trg = duration_outputs[2].astype(np.float32)
    en = duration_outputs[3].astype(np.float32)
    frame_length = int(pred_aln_trg.shape[-1])

    if frame_length != FRAME_BUCKET:
        raise ValueError(
            f'Frame length {frame_length} does not match the static decoder bucket {FRAME_BUCKET}. '
            'The exported static decoder/vocoder use instance normalization over time, so padding shorter '
            'utterances changes the entire signal. Use text that lands exactly on this bucket or export '
            'matching static decoder/vocoder buckets.'
        )

    text_encoder_outputs = text_encoder_session.run(
        None,
        {
            'input_ids': input_ids,
            'pred_aln_trg': pred_aln_trg,
        },
    )
    asr = text_encoder_outputs[1].astype(np.float32)

    f0_pred, n_pred = f0n_session.run(
        None,
        {
            'en': en,
            'ref_s': ref_s,
        },
    )
    f0_pred = f0_pred.astype(np.float32)
    n_pred = n_pred.astype(np.float32)

    pitch_length = int(f0_pred.shape[-1])
    if pitch_length != PITCH_BUCKET:
        raise ValueError(
            f'Pitch length {pitch_length} does not match the static decoder bucket {PITCH_BUCKET}. '
            'Use text that lands exactly on this bucket or export matching static decoder/vocoder buckets.'
        )

    timbre = ref_s[:, :128].astype(np.float32)
    decoder_state = decoder_session.run(
        None,
        {
            'asr': asr,
            'F0_pred': f0_pred,
            'N_pred': n_pred,
            'timbre': timbre,
        },
    )[0].astype(np.float32)

    har = build_har(model.decoder.generator, f0_pred)
    waveform = vocoder_session.run(
        None,
        {
            'decoder_state': decoder_state,
            'timbre': timbre,
            'har': har,
        },
    )[0].astype(np.float32)

    sample_length = frame_length * SAMPLES_PER_FRAME
    trimmed_waveform = waveform[:sample_length]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, trimmed_waveform, SAMPLE_RATE)

    return {
        'phonemes': phonemes,
        'token_length': int(input_ids.shape[-1]),
        'frame_length': frame_length,
        'pitch_length': pitch_length,
        'sample_length': int(trimmed_waveform.shape[-1]),
        'output_path': output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Run Kokoro dynamic frontend ONNX inference with static decoder and vocoder',
        add_help=True,
    )
    parser.add_argument('--text', required=True, help='input text to synthesize')
    parser.add_argument('--voice', default='af_heart', help='voice id or .pt path')
    parser.add_argument('--lang_code', default='a', help='pipeline language code')
    parser.add_argument('--speed', type=float, default=1.0, help='speech speed')
    parser.add_argument('--repo_id', default='hexgrad/Kokoro-82M', help='voice/model repository id used by KPipeline/KModel')
    parser.add_argument('--config_file', default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--onnx_dir', default='onnx_modules', help='directory with dynamic ONNX modules: encoder, duration_predictor, text_encoder, and f0n_predictor')
    parser.add_argument('--static_onnx_dir', default='onnx_modules_static_frontend', help='directory with static ONNX modules: decoder_front.onnx and vocoder.onnx')
    parser.add_argument('--output', default='hybrid_onnx_output.wav', help='output wav path')
    parser.add_argument(
        '--providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers in priority order',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]

    result = synthesize_with_dynamic_frontend_static_backend(
        text=args.text,
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        repo_id=args.repo_id,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        onnx_dir=args.onnx_dir,
        static_onnx_dir=args.static_onnx_dir,
        output_path=args.output,
        providers=providers,
    )

    print(f'phonemes     : {result["phonemes"]}')
    print(f'token_length : {result["token_length"]}')
    print(f'frame_length : {result["frame_length"]}/{FRAME_BUCKET}')
    print(f'pitch_length : {result["pitch_length"]}/{PITCH_BUCKET}')
    print(f'sample_length: {result["sample_length"]}')
    print(f'output       : {result["output_path"]}')


if __name__ == '__main__':
    main()