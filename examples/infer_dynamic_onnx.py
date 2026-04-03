import argparse
from pathlib import Path

import onnxruntime as ort
import soundfile as sf
import numpy as np

from kokoro import KModel, KPipeline


SAMPLE_RATE = 24000


def load_phonemes_and_input_ids(pipeline: KPipeline, text: str):
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


def synthesize_with_full_onnx(
    text: str,
    voice: str,
    lang_code: str,
    speed: float,
    repo_id: str,
    config_file: str,
    checkpoint_path: str,
    onnx_path: str,
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

    session = create_session(Path(onnx_path), providers)
    waveform, duration = session.run(
        None,
        {
            'input_ids': input_ids,
            'ref_s': ref_s,
            'speed': np.asarray([speed], dtype=np.float32),
        },
    )

    waveform = waveform.astype(np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, SAMPLE_RATE)

    return {
        'phonemes': phonemes,
        'token_length': int(input_ids.shape[-1]),
        'duration_length': int(duration.shape[-1]),
        'sample_length': int(waveform.shape[-1]),
        'output_path': output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Run dynamic Kokoro full-model ONNX inference', add_help=True)
    parser.add_argument('--text', required=True, help='input text to synthesize')
    parser.add_argument('--voice', default='af_heart', help='voice id or .pt path')
    parser.add_argument('--lang_code', default='a', help='pipeline language code')
    parser.add_argument('--speed', type=float, default=1.0, help='speech speed')
    parser.add_argument('--repo_id', default='hexgrad/Kokoro-82M', help='voice/model repository id used by KPipeline/KModel')
    parser.add_argument('--config_file', default='checkpoints/config.json', help='path to model config file')
    parser.add_argument('--checkpoint_path', default='checkpoints/kokoro-v1_0.pth', help='path to model checkpoint')
    parser.add_argument('--onnx_path', default='onnx_modules/kokoro.onnx', help='path to the dynamic full-model ONNX file')
    parser.add_argument('--output', default='full_onnx_output.wav', help='output wav path')
    parser.add_argument(
        '--providers',
        default='CPUExecutionProvider',
        help='comma-separated ONNX Runtime providers in priority order',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]

    result = synthesize_with_full_onnx(
        text=args.text,
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        repo_id=args.repo_id,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        output_path=args.output,
        providers=providers,
    )

    print(f'phonemes       : {result["phonemes"]}')
    print(f'token_length   : {result["token_length"]}')
    print(f'duration_length: {result["duration_length"]}')
    print(f'sample_length  : {result["sample_length"]}')
    print(f'output         : {result["output_path"]}')


if __name__ == '__main__':
    main()