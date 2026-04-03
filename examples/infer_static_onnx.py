import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from kokoro import KModel, KPipeline


TOKEN_BUCKET = 128
FRAME_BUCKET = 198
PITCH_BUCKET = FRAME_BUCKET * 2
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 600


def load_phonemes_and_input_ids(pipeline: KPipeline, text: str) -> tuple[str, torch.LongTensor]:
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

    input_ids = torch.tensor([input_id_list], dtype=torch.long)
    return phonemes, input_ids


def load_reference_style(pipeline: KPipeline, voice: str, phoneme_length: int) -> torch.FloatTensor:
    pack = pipeline.load_voice(voice).cpu()
    ref_s = pack[phoneme_length - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s


def right_pad_last_dim(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.shape[-1] > target_length:
        raise ValueError(f'Cannot pad array with length {array.shape[-1]} to shorter target {target_length}.')
    if array.shape[-1] == target_length:
        return array

    padded = np.zeros((*array.shape[:-1], target_length), dtype=array.dtype)
    padded[..., :array.shape[-1]] = array
    return padded


def right_pad_alignment(alignment: np.ndarray, token_bucket: int, frame_bucket: int) -> np.ndarray:
    if alignment.shape[1] > token_bucket or alignment.shape[2] > frame_bucket:
        raise ValueError(
            f'Alignment shape {alignment.shape} exceeds static bucket {(token_bucket, frame_bucket)}.'
        )

    padded = np.zeros((alignment.shape[0], token_bucket, frame_bucket), dtype=alignment.dtype)
    padded[:, :alignment.shape[1], :alignment.shape[2]] = alignment
    return padded


def build_padded_text_mask(token_length: int, token_bucket: int) -> np.ndarray:
    positions = np.arange(token_bucket, dtype=np.int64)[None, :]
    return positions >= token_length


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


def synthesize_with_static_frontend(
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

    token_length = input_ids.shape[-1]
    if token_length > TOKEN_BUCKET:
        raise ValueError(
            f'Token length {token_length} exceeds the first static token bucket {TOKEN_BUCKET}. '
            'Use shorter text or export a larger static bucket.'
        )

    onnx_root = Path(onnx_dir)
    static_root = Path(static_onnx_dir)

    encoder_session = create_session(static_root / f'encoder_token_{TOKEN_BUCKET}.onnx', providers)
    duration_session = create_session(onnx_root / 'duration_predictor.onnx', providers)
    text_encoder_session = create_session(
        static_root / f'text_encoder_token_{TOKEN_BUCKET}_frame_{FRAME_BUCKET}.onnx',
        providers,
    )
    f0n_shared_session = create_session(static_root / f'f0n_shared_frame_{FRAME_BUCKET}.onnx', providers)
    f0n_head_session = create_session(static_root / f'f0n_head_frame_{FRAME_BUCKET}.onnx', providers)
    decoder_session = create_session(static_root / 'decoder_front.onnx', providers)
    vocoder_session = create_session(static_root / 'vocoder.onnx', providers)

    padded_input_ids = np.zeros((1, TOKEN_BUCKET), dtype=np.int64)
    padded_input_ids[:, :token_length] = input_ids.numpy()
    padded_text_mask = build_padded_text_mask(token_length, TOKEN_BUCKET)
    padded_d_en = encoder_session.run(
        None,
        {
            'input_ids': padded_input_ids,
            'text_mask': padded_text_mask,
        },
    )[0]
    d_en = padded_d_en[:, :, :token_length]

    text_mask = np.zeros((1, token_length), dtype=bool)
    duration_outputs = duration_session.run(
        None,
        {
            'd_en': d_en,
            'ref_s': ref_s.numpy().astype(np.float32),
            'input_lengths': np.array([token_length], dtype=np.int64),
            'text_mask': text_mask,
            'speed': np.array([speed], dtype=np.float32),
        },
    )
    pred_dur = duration_outputs[1]
    pred_aln_trg = duration_outputs[2].astype(np.float32)
    en = duration_outputs[3].astype(np.float32)
    frame_length = pred_aln_trg.shape[-1]

    if frame_length > FRAME_BUCKET:
        raise ValueError(
            f'Frame length {frame_length} exceeds the first static frame bucket {FRAME_BUCKET}. '
            'Use shorter text, increase speed, or export a larger static bucket.'
        )

    padded_alignment = right_pad_alignment(pred_aln_trg, TOKEN_BUCKET, FRAME_BUCKET)
    text_encoder_outputs = text_encoder_session.run(
        None,
        {
            'input_ids': padded_input_ids,
            'pred_aln_trg': padded_alignment,
            'text_mask': padded_text_mask,
        },
    )
    asr = text_encoder_outputs[1].astype(np.float32)

    padded_en = right_pad_last_dim(en, FRAME_BUCKET)
    shared = f0n_shared_session.run(
        None,
        {
            'en': padded_en,
        },
    )[0].astype(np.float32)
    f0_pred, n_pred = f0n_head_session.run(
        None,
        {
            'shared': shared,
            'ref_s': ref_s.numpy().astype(np.float32),
        },
    )
    f0_pred = f0_pred.astype(np.float32)
    n_pred = n_pred.astype(np.float32)

    timbre = ref_s[:, :128].numpy().astype(np.float32)
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
        'token_length': token_length,
        'frame_length': frame_length,
        'sample_length': sample_length,
        'output_path': output_path,
        'pitch_length': PITCH_BUCKET,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Run Kokoro static ONNX inference with the first exported bucket',
        add_help=True,
    )
    parser.add_argument('--text', required=True, help='input text to synthesize')
    parser.add_argument('--voice', default='af_heart', help='voice id or .pt path')
    parser.add_argument('--lang_code', default='a', help='pipeline language code')
    parser.add_argument('--speed', type=float, default=1.0, help='speech speed')
    parser.add_argument('--repo_id', default='hexgrad/Kokoro-82M', help='voice/model repository id used by KPipeline/KModel')
    parser.add_argument('--config_file', default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--onnx_dir', default='onnx_modules', help='directory with dynamic ONNX modules; only duration_predictor.onnx is used')
    parser.add_argument('--static_onnx_dir', default='onnx_modules_static_frontend', help='directory with static ONNX modules, including frontend, decoder_front.onnx, and vocoder.onnx')
    parser.add_argument('--output', default='static_onnx_output.wav', help='output wav path')
    parser.add_argument(
        '--providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers in priority order',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]

    result = synthesize_with_static_frontend(
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
    print(f'token_length : {result["token_length"]}/{TOKEN_BUCKET}')
    print(f'frame_length : {result["frame_length"]}/{FRAME_BUCKET}')
    print(f'sample_length: {result["sample_length"]}')
    print(f'pitch_length : {result["pitch_length"]}')
    print(f'output       : {result["output_path"]}')


if __name__ == '__main__':
    main()