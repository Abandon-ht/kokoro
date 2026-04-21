Examples directory layout and recommended workflows.

Current entrypoints:
- `python examples/export_dynamic_onnx_modules.py`
- `python examples/export_static_onnx_modules.py`
- `python examples/export_static_onnx_npy.py`
- `python examples/infer_dynamic_onnx.py`
- `python examples/infer_static_onnx.py`
- `python examples/infer_static_axmodel.py`

Export workflow:

1. Export dynamic ONNX modules into `onnx_modules`:

```bash
python examples/export_dynamic_onnx_modules.py \
  --config_file checkpoints/config.json \
  --checkpoint_path checkpoints/kokoro-v1_0.pth \
  --output_dir onnx_modules
```

2. Export static ONNX modules into `onnx_modules_static_frontend`:

```bash
python examples/export_static_onnx_modules.py \
  --config_file checkpoints/config.json \
  --checkpoint_path checkpoints/kokoro-v1_0.pth \
  --output_dir onnx_modules_static_frontend
```

3. Export representative NPY inputs for quantization:

```bash
python examples/export_static_onnx_npy.py \
  --config_file checkpoints/config.json \
  --checkpoint_path checkpoints/kokoro-v1_0.pth
```

The static backend export now splits the vocoder into:
- `vocoder_core.onnx`: quantized backend that runs on NPU
- `vocoder_tail.onnx`: floating-point tail with `conv_post + exp/sin + iSTFT`

The calibration export only writes NPY datasets for `decoder_front` and `vocoder_core`, because `vocoder_tail` is intended to stay in floating point.

Inference workflow:

1. Dynamic full-model ONNX inference:

```bash
python examples/infer_dynamic_onnx.py \
  --text 'Hello from Kokoro.' \
  --onnx_path onnx_modules/kokoro.onnx \
  --output full_onnx_output.wav
```

2. Static ONNX inference using static frontend plus static backend buckets:

```bash
python examples/infer_static_onnx.py \
  --text 'Hello from Kokoro.' \
  --onnx_dir onnx_modules \
  --static_onnx_dir onnx_modules_static_frontend \
  --output static_onnx_output.wav
```

3. Static AXModel inference for the quantized deployment path:

```bash
python examples/infer_static_axmodel.py \
  --text 'Hello from Kokoro.' \
  --onnx_dir onnx_modules \
  --static_onnx_dir onnx_modules_static_frontend \
  --axmodel_dir kokoro-axmodel \
  --output static_axmodel_output.wav
```

Notes:
- `infer_dynamic_onnx.py` is the reference ONNX path when you want the fewest bucket constraints.
- `infer_static_onnx.py` is the recommended static-bucket ONNX path after the frontend length/mask fix; it automatically chains `vocoder_core.onnx` and `vocoder_tail.onnx` when both are present.
- `infer_static_axmodel.py` is the deployment-oriented path that mixes ONNX Runtime duration prediction with AXERA static models and an ONNX floating-point `vocoder_tail.onnx`.
- `examples/legacy` contains older names, wrappers, and one-off debugging scripts.
- `examples/legacy/hybrid_onnx_infer.py` is kept only for historical debugging; it is not a recommended production path because static decoder/vocoder buckets are unsafe for padded short utterances.