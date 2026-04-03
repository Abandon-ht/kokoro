import argparse
from pathlib import Path
from typing import Any

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
MODULE_NAMES = ('encoder', 'text_encoder', 'f0n_shared', 'f0n_head', 'decoder', 'vocoder')


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


def build_padded_text_mask(token_length: int, token_bucket: int, backend: str) -> np.ndarray:
    positions = np.arange(token_bucket, dtype=np.int64)[None, :]
    mask = positions >= token_length
    if backend == 'axmodel':
        return mask.astype(np.uint8)
    return mask


def build_padded_input_ids(input_ids: torch.LongTensor, token_length: int, backend: str) -> np.ndarray:
    dtype = np.int32 if backend == 'axmodel' else np.int64
    padded = np.zeros((1, TOKEN_BUCKET), dtype=dtype)
    padded[:, :token_length] = input_ids.numpy().astype(dtype)
    return padded


def create_ort_session(model_path: Path, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(model_path.as_posix(), providers=providers)


def create_ax_session(model_path: Path) -> Any:
    try:
        import axengine as art
    except ImportError as exc:
        raise RuntimeError(
            'AXModel backend requested but axengine is not installed in the active environment.'
        ) from exc
    return art.InferenceSession(model_path.as_posix())


def run_session(session: Any, feed_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
    return session.run(None, feed_dict)


def build_har(generator: Any, f0_pred: np.ndarray) -> np.ndarray:
    f0_tensor = torch.from_numpy(f0_pred).float()
    with torch.no_grad():
        f0 = generator.f0_upsamp(f0_tensor[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
    return har.cpu().contiguous().numpy().astype(np.float32)


def resolve_backend_map(args: argparse.Namespace) -> dict[str, str]:
    backend_map = {module_name: args.default_backend for module_name in MODULE_NAMES}
    for module_name in MODULE_NAMES:
        override = getattr(args, f'{module_name}_backend')
        if override is not None:
            backend_map[module_name] = override
    return backend_map


def resolve_model_path(root: Path, module_name: str, backend: str) -> Path:
    if backend == 'onnx':
        candidates = {
            'encoder': [f'encoder_token_{TOKEN_BUCKET}.onnx'],
            'text_encoder': [f'text_encoder_token_{TOKEN_BUCKET}_frame_{FRAME_BUCKET}.onnx'],
            'f0n_shared': [f'f0n_shared_frame_{FRAME_BUCKET}.onnx'],
            'f0n_head': [f'f0n_head_frame_{FRAME_BUCKET}.onnx'],
            'decoder': ['decoder_front.onnx', f'decoder_front_{FRAME_BUCKET}.onnx'],
            'vocoder': ['vocoder.onnx', f'vocoder_{PITCH_BUCKET}.onnx'],
        }
    elif backend == 'axmodel':
        candidates = {
            'encoder': [f'encoder_token_{TOKEN_BUCKET}.axmodel'],
            'text_encoder': [f'text_encoder_token_{TOKEN_BUCKET}_frame_{FRAME_BUCKET}.axmodel'],
            'f0n_shared': [f'f0n_shared_frame_{FRAME_BUCKET}.axmodel'],
            'f0n_head': [f'f0n_head_frame_{FRAME_BUCKET}.axmodel'],
            'decoder': [f'decoder_front_{FRAME_BUCKET}.axmodel', 'decoder_front.axmodel'],
            'vocoder': [f'vocoder_{PITCH_BUCKET}.axmodel', 'vocoder.axmodel'],
        }
    else:
        raise ValueError(f'Unsupported backend {backend!r} for module {module_name!r}.')

    for relative_path in candidates[module_name]:
        model_path = root / relative_path
        if model_path.exists():
            return model_path

    candidate_list = ', '.join(candidates[module_name])
    raise FileNotFoundError(
        f'Unable to find a {backend} model for {module_name!r} under {root}. Tried: {candidate_list}.'
    )


def create_stage_session(
    module_name: str,
    backend: str,
    static_onnx_root: Path,
    axmodel_root: Path,
    static_onnx_providers: list[str],
) -> tuple[Any, Path]:
    if backend == 'onnx':
        model_path = resolve_model_path(static_onnx_root, module_name, backend)
        return create_ort_session(model_path, static_onnx_providers), model_path
    if backend == 'axmodel':
        model_path = resolve_model_path(axmodel_root, module_name, backend)
        return create_ax_session(model_path), model_path
    raise ValueError(f'Unsupported backend {backend!r} for module {module_name!r}.')


def synthesize_with_mixed_static_backends(
    text: str,
    voice: str,
    lang_code: str,
    speed: float,
    repo_id: str,
    config_file: str,
    checkpoint_path: str,
    dynamic_onnx_path: str,
    static_onnx_dir: str,
    axmodel_dir: str,
    output_path: str,
    backend_map: dict[str, str],
    dynamic_providers: list[str],
    static_onnx_providers: list[str],
) -> dict[str, Any]:
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

    static_onnx_root = Path(static_onnx_dir)
    axmodel_root = Path(axmodel_dir)

    stage_sessions: dict[str, Any] = {}
    stage_model_paths: dict[str, str] = {}
    for module_name in MODULE_NAMES:
        stage_session, model_path = create_stage_session(
            module_name=module_name,
            backend=backend_map[module_name],
            static_onnx_root=static_onnx_root,
            axmodel_root=axmodel_root,
            static_onnx_providers=static_onnx_providers,
        )
        stage_sessions[module_name] = stage_session
        stage_model_paths[module_name] = model_path.as_posix()

    duration_model_path = Path(dynamic_onnx_path)
    duration_session = create_ort_session(duration_model_path, dynamic_providers)

    padded_input_ids = build_padded_input_ids(input_ids, token_length, backend_map['encoder'])
    padded_text_mask = build_padded_text_mask(token_length, TOKEN_BUCKET, backend_map['encoder'])
    padded_d_en = run_session(
        stage_sessions['encoder'],
        {
            'input_ids': padded_input_ids,
            'text_mask': padded_text_mask,
        },
    )[0]
    d_en = padded_d_en[:, :, :token_length].astype(np.float32)

    duration_outputs = run_session(
        duration_session,
        {
            'd_en': d_en,
            'ref_s': ref_s.numpy().astype(np.float32),
            'input_lengths': np.array([token_length], dtype=np.int64),
            'text_mask': np.zeros((1, token_length), dtype=bool),
            'speed': np.array([speed], dtype=np.float32),
        },
    )
    pred_aln_trg = duration_outputs[2].astype(np.float32)
    en = duration_outputs[3].astype(np.float32)
    frame_length = pred_aln_trg.shape[-1]

    if frame_length > FRAME_BUCKET:
        raise ValueError(
            f'Frame length {frame_length} exceeds the first static frame bucket {FRAME_BUCKET}. '
            'Use shorter text, increase speed, or export a larger static bucket.'
        )

    text_encoder_input_ids = build_padded_input_ids(input_ids, token_length, backend_map['text_encoder'])
    text_encoder_mask = build_padded_text_mask(token_length, TOKEN_BUCKET, backend_map['text_encoder'])
    padded_alignment = right_pad_alignment(pred_aln_trg, TOKEN_BUCKET, FRAME_BUCKET)
    text_encoder_outputs = run_session(
        stage_sessions['text_encoder'],
        {
            'input_ids': text_encoder_input_ids,
            'pred_aln_trg': padded_alignment,
            'text_mask': text_encoder_mask,
        },
    )
    asr = text_encoder_outputs[1].astype(np.float32)

    padded_en = right_pad_last_dim(en, FRAME_BUCKET)
    shared = run_session(
        stage_sessions['f0n_shared'],
        {
            'en': padded_en,
        },
    )[0].astype(np.float32)
    f0_pred, n_pred = run_session(
        stage_sessions['f0n_head'],
        {
            'shared': shared,
            'ref_s': ref_s.numpy().astype(np.float32),
        },
    )
    f0_pred = f0_pred.astype(np.float32)
    n_pred = n_pred.astype(np.float32)

    timbre = ref_s[:, :128].numpy().astype(np.float32)
    decoder_state = run_session(
        stage_sessions['decoder'],
        {
            'asr': asr,
            'F0_pred': f0_pred,
            'N_pred': n_pred,
            'timbre': timbre,
        },
    )[0].astype(np.float32)

    har = build_har(model.decoder.generator, f0_pred)
    waveform = run_session(
        stage_sessions['vocoder'],
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
        'backend_map': backend_map,
        'stage_model_paths': stage_model_paths,
        'duration_model_path': duration_model_path.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Run Kokoro mixed static backend inference for backend ablation experiments',
        add_help=True,
    )
    parser.add_argument('--text', required=True, help='input text to synthesize')
    parser.add_argument('--voice', default='af_heart', help='voice id or .pt path')
    parser.add_argument('--lang_code', default='a', help='pipeline language code')
    parser.add_argument('--speed', type=float, default=1.0, help='speech speed')
    parser.add_argument('--repo_id', default='hexgrad/Kokoro-82M', help='voice/model repository id used by KPipeline/KModel')
    parser.add_argument('--config_file', default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--dynamic_onnx_path', default='onnx_modules/duration_predictor.onnx', help='path to the dynamic duration_predictor.onnx model')
    parser.add_argument('--static_onnx_dir', default='onnx_modules_static_frontend', help='directory with static ONNX frontend modules')
    parser.add_argument('--axmodel_dir', default='kokoro-axmodel', help='directory with converted AXModel modules')
    parser.add_argument('--output', default='mixed_static_backend_output.wav', help='output wav path')
    parser.add_argument('--default_backend', choices=('onnx', 'axmodel'), default='axmodel', help='default backend used by all static modules before per-module overrides')
    parser.add_argument('--encoder_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for encoder')
    parser.add_argument('--text_encoder_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for text_encoder')
    parser.add_argument('--f0n_shared_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for f0n_shared')
    parser.add_argument('--f0n_head_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for f0n_head')
    parser.add_argument('--decoder_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for decoder')
    parser.add_argument('--vocoder_backend', choices=('onnx', 'axmodel'), default=None, help='override backend for vocoder')
    parser.add_argument(
        '--dynamic_providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers for dynamic duration_predictor.onnx',
    )
    parser.add_argument(
        '--static_onnx_providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers for static ONNX modules when a module backend is onnx',
    )
    return parser.parse_args()


def parse_provider_list(provider_string: str) -> list[str]:
    return [provider.strip() for provider in provider_string.split(',') if provider.strip()]


def main() -> None:
    args = parse_args()
    backend_map = resolve_backend_map(args)
    result = synthesize_with_mixed_static_backends(
        text=args.text,
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        repo_id=args.repo_id,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        dynamic_onnx_path=args.dynamic_onnx_path,
        static_onnx_dir=args.static_onnx_dir,
        axmodel_dir=args.axmodel_dir,
        output_path=args.output,
        backend_map=backend_map,
        dynamic_providers=parse_provider_list(args.dynamic_providers),
        static_onnx_providers=parse_provider_list(args.static_onnx_providers),
    )

    print(f'phonemes     : {result["phonemes"]}')
    print(f'token_length : {result["token_length"]}/{TOKEN_BUCKET}')
    print(f'frame_length : {result["frame_length"]}/{FRAME_BUCKET}')
    print(f'sample_length: {result["sample_length"]}')
    print(f'pitch_length : {result["pitch_length"]}')
    print(f'duration     : dynamic_onnx -> {result["duration_model_path"]}')
    for module_name in MODULE_NAMES:
        print(f'{module_name:13}: {result["backend_map"][module_name]:7} -> {result["stage_model_paths"][module_name]}')
    print(f'output       : {result["output_path"]}')


if __name__ == '__main__':
    main()