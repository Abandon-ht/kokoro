import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import axengine as art
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
    return (positions >= token_length).astype(np.uint8)


def create_ort_session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(model_path.as_posix(), providers=providers)


def create_ax_session(model_path: Path) -> art.InferenceSession:
    return art.InferenceSession(model_path.as_posix())


def _cast_array_for_dtype(array: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    if array.dtype == target_dtype:
        return array
    return array.astype(target_dtype, copy=False)


def _normalize_feed_dict(session: Any, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not hasattr(session, 'get_inputs'):
        return feed_dict

    normalized = dict(feed_dict)
    for node_arg in session.get_inputs():
        name = getattr(node_arg, 'name', None)
        target_dtype = getattr(node_arg, 'dtype', None)
        if name is None or target_dtype is None or name not in normalized:
            continue
        normalized[name] = _cast_array_for_dtype(normalized[name], target_dtype)
    return normalized


def run_session(session: Any, feed_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
    return session.run(None, _normalize_feed_dict(session, feed_dict))


def build_har(generator, f0_pred: np.ndarray) -> np.ndarray:
    f0_tensor = torch.from_numpy(f0_pred).float()
    with torch.no_grad():
        f0 = generator.f0_upsamp(f0_tensor[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
    return har.cpu().contiguous().numpy().astype(np.float32)


def synthesize_with_static_axmodel(
    text: str,
    voice: str,
    lang_code: str,
    speed: float,
    repo_id: str,
    config_file: str,
    checkpoint_path: str,
    onnx_dir: str,
    axmodel_dir: str,
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
    axmodel_root = Path(axmodel_dir)

    encoder_session = create_ax_session(axmodel_root / f'encoder_token_{TOKEN_BUCKET}.axmodel')
    duration_session = create_ort_session(onnx_root / 'duration_predictor.onnx', providers)
    text_encoder_session = create_ax_session(
        axmodel_root / f'text_encoder_token_{TOKEN_BUCKET}_frame_{FRAME_BUCKET}.axmodel',
    )
    f0n_shared_session = create_ax_session(axmodel_root / f'f0n_shared_frame_{FRAME_BUCKET}.axmodel')
    f0n_head_session = create_ax_session(axmodel_root / f'f0n_head_frame_{FRAME_BUCKET}.axmodel')
    decoder_session = create_ax_session(axmodel_root / f'decoder_front_{FRAME_BUCKET}.axmodel')
    vocoder_session = create_ax_session(axmodel_root / f'vocoder_{PITCH_BUCKET}.axmodel')

    padded_input_ids = np.zeros((1, TOKEN_BUCKET), dtype=np.int32)
    padded_input_ids[:, :token_length] = input_ids.numpy().astype(np.int32)
    padded_text_mask = build_padded_text_mask(token_length, TOKEN_BUCKET)
    padded_d_en = run_session(
        encoder_session,
        {
            'input_ids': padded_input_ids,
            'text_mask': padded_text_mask,
        },
    )[0]
    d_en = padded_d_en[:, :, :token_length]

    text_mask = np.zeros((1, token_length), dtype=bool)
    duration_outputs = run_session(
        duration_session,
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
    text_encoder_outputs = run_session(
        text_encoder_session,
        {
            'input_ids': padded_input_ids,
            'pred_aln_trg': padded_alignment,
            'text_mask': padded_text_mask,
        },
    )
    asr = text_encoder_outputs[1].astype(np.float32)

    padded_en = right_pad_last_dim(en, FRAME_BUCKET)
    shared = run_session(
        f0n_shared_session,
        {
            'en': padded_en,
        },
    )[0].astype(np.float32)
    f0_pred, n_pred = run_session(
        f0n_head_session,
        {
            'shared': shared,
            'ref_s': ref_s.numpy().astype(np.float32),
        },
    )
    f0_pred = f0_pred.astype(np.float32)
    n_pred = n_pred.astype(np.float32)

    timbre = ref_s[:, :128].numpy().astype(np.float32)
    decoder_state = run_session(
        decoder_session,
        {
            'asr': asr,
            'F0_pred': f0_pred,
            'N_pred': n_pred,
            'timbre': timbre,
        },
    )[0].astype(np.float32)

    har = build_har(model.decoder.generator, f0_pred)
    waveform = run_session(
        vocoder_session,
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
        'Run Kokoro static AXModel inference with the first exported bucket',
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
    parser.add_argument('--axmodel_dir', default='kokoro-axmodel', help='directory with static AXERA models for encoder, text_encoder, f0n, decoder_front, and vocoder')
    parser.add_argument('--output', default='static_axmodel_output.wav', help='output wav path')
    parser.add_argument(
        '--providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers in priority order for duration_predictor.onnx only',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]

    result = synthesize_with_static_axmodel(
        text=args.text,
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        repo_id=args.repo_id,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        onnx_dir=args.onnx_dir,
        axmodel_dir=args.axmodel_dir,
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