import argparse
import shutil
import tarfile
from pathlib import Path


def parse_bucket_list(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError('at least one bucket value is required')
    return sorted(set(values))


def parse_text_encoder_buckets(raw_value: str) -> list[tuple[int, int]]:
    buckets = []
    for item in raw_value.split(','):
        item = item.strip()
        if not item:
            continue
        token_part, frame_part = item.split(':', 1)
        buckets.append((int(token_part), int(frame_part)))
    if not buckets:
        raise ValueError('at least one text encoder bucket is required')
    return sorted(set(buckets))


def pack_tensor_dir(source_dir: Path, archive_path: Path):
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=source_dir.name)


def stage_bucket(onnx_root: Path, npy_root: Path, axmodel_root: Path, bucket_name: str):
    source_onnx = onnx_root / f'{bucket_name}.onnx'
    source_npy_bucket = npy_root / bucket_name
    if not source_onnx.is_file():
        raise FileNotFoundError(f'missing ONNX file: {source_onnx}')
    if not source_npy_bucket.is_dir():
        raise FileNotFoundError(f'missing NPY bucket directory: {source_npy_bucket}')

    dest_dir = axmodel_root / bucket_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_onnx, dest_dir / source_onnx.name)

    for archive_path in dest_dir.glob('*.tar.gz'):
        archive_path.unlink()

    for tensor_dir in sorted(path for path in source_npy_bucket.iterdir() if path.is_dir()):
        pack_tensor_dir(tensor_dir, dest_dir / f'{tensor_dir.name}.tar.gz')

    print(f'staged {bucket_name} -> {dest_dir}')


def main():
    parser = argparse.ArgumentParser('Stage static split ONNX and calibration assets into axmodel directories', add_help=True)
    parser.add_argument('--onnx_dir', type=str, default='onnx_modules_static_frontend', help='directory containing exported split ONNX models')
    parser.add_argument('--npy_dir', type=str, required=True, help='timestamped static_frontend_npy export directory')
    parser.add_argument('--axmodel_dir', type=str, default='axmodel', help='destination axmodel directory')
    parser.add_argument('--token_buckets', type=str, default='128,256,512', help='comma-separated encoder token buckets to stage')
    parser.add_argument('--text_encoder_buckets', type=str, default='128:198,256:396,512:512', help='comma-separated token:frame buckets to stage for text_encoder')
    parser.add_argument('--frame_buckets', type=str, default='198,256,396,512', help='comma-separated f0n frame buckets to stage')
    parser.add_argument('--include_head', action='store_true', help='also stage f0n_head assets for experimental/manual builds')
    args = parser.parse_args()

    onnx_root = Path(args.onnx_dir)
    npy_root = Path(args.npy_dir)
    axmodel_root = Path(args.axmodel_dir)
    token_buckets = parse_bucket_list(args.token_buckets)
    text_encoder_buckets = parse_text_encoder_buckets(args.text_encoder_buckets)
    frame_buckets = parse_bucket_list(args.frame_buckets)

    for token_bucket in token_buckets:
        stage_bucket(onnx_root, npy_root, axmodel_root, f'encoder_token_{token_bucket}')

    for token_bucket, frame_bucket in text_encoder_buckets:
        stage_bucket(onnx_root, npy_root, axmodel_root, f'text_encoder_token_{token_bucket}_frame_{frame_bucket}')

    for frame_bucket in frame_buckets:
        stage_bucket(onnx_root, npy_root, axmodel_root, f'f0n_shared_frame_{frame_bucket}')
        if args.include_head:
            stage_bucket(onnx_root, npy_root, axmodel_root, f'f0n_head_frame_{frame_bucket}')


if __name__ == '__main__':
    main()