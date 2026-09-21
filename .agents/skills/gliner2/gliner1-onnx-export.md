# GLiNER (v1): ONNX Runtime & OpenVINO Export

Export a PyTorch checkpoint to ONNX or OpenVINO IR and run it through the same
`predict_entities`/`inference` API used for PyTorch models. Condensed from
[ONNX Runtime and OpenVINO](https://urchade.github.io/GLiNER/convert_to_onnx.html).

| Runtime | Artifact | Use |
|---|---|---|
| PyTorch | `model.safetensors` / `pytorch_model.bin` | Training, standard inference |
| ONNX Runtime | `.onnx` | Portable CPU or CUDA inference |
| OpenVINO | `.xml` + `.bin` (or `.onnx`) | CPU/GPU/NPU/`AUTO` via OpenVINO |

## Install

```bash
pip install "gliner[onnx]" onnx   # CPU ONNX Runtime + exporter
pip install "gliner[gpu]"         # CUDA ONNX Runtime (pick onnx OR gpu, never both — same module)
pip install "gliner[openvino]"    # direct OpenVINO export/inference, no intermediate ONNX needed
```

## Export

A runtime-backed model **cannot be exported again** — keep the original PyTorch checkpoint if you
need multiple deployment formats.

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")   # load PyTorch first

paths = model.export_to_onnx(save_dir="exports/onnx", onnx_filename="model.onnx", opset=19)
print(paths["onnx_path"])

# Add a dynamically-quantized copy (best-effort — warns and returns None if unavailable/fails,
# regular ONNX model still produced)
paths = model.export_to_onnx(
    save_dir="exports/onnx", quantized_filename="model_int8.onnx", quantize=True, opset=19,
)
```

```python
paths = model.export_to_openvino(save_dir="exports/openvino", compress_to_fp16=False)
print(paths["openvino_path"], paths["weights_path"])
```

CLI equivalents:

```bash
python scripts/convert_to_onnx.py --model_path urchade/gliner_small-v2.1 \
    --save_path exports/onnx --file_name model.onnx --quantized_file_name model_int8.onnx

python scripts/convert_to_openvino.py --model_path urchade/gliner_small-v2.1 \
    --save_path exports/openvino --file_name model.xml
```

Export writes `gliner_config.json` and tokenizer files alongside the artifact automatically. Keep
an OpenVINO `.xml` beside its matching `.bin`.

## Run an exported model

```python
# ONNX Runtime, CPU (default)
model = GLiNER.from_pretrained(
    "exports/onnx", runtime="onnxruntime", runtime_model_file="model.onnx", local_files_only=True,
)

# ONNX Runtime, CUDA — install gliner[gpu]; include CPU as fallback for unsupported ops
model = GLiNER.from_pretrained(
    "exports/onnx", runtime="onnxruntime", runtime_model_file="model.onnx",
    runtime_options={"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
)

# OpenVINO IR
model = GLiNER.from_pretrained(
    "exports/openvino", runtime="openvino", runtime_model_file="model.xml",
    runtime_options={"device_name": "CPU", "config": {}},   # "CPU"|"GPU"|"NPU"|"AUTO"
)

# OpenVINO compiling an ONNX artifact directly (no IR conversion needed)
model = GLiNER.from_pretrained(
    "exports/onnx", runtime="openvino", runtime_model_file="model.onnx",
    runtime_options={"device_name": "AUTO"},
)

entities = model.predict_entities("Apple was founded by Steve Jobs.", ["organization", "person"])
```

Runtime aliases: `onnx`/`ort` → `onnxruntime`; `ov` → `openvino`. `runtime_options` also accepts a
pre-built `session_options`/`session` (ONNX Runtime) or `core`/`compiled_model` (OpenVINO) when the
application owns runtime lifecycle/caching.

## Supported architectures

| Architecture | ONNX | OpenVINO |
|---|---|---|
| UniEncoderSpan / UniEncoderToken | ✅ | ✅ |
| BiEncoderSpan / BiEncoderToken | ✅ | ✅ |
| Relation extraction (span/token) | ✅ | ✅ |
| Generative decoder (`*Decoder`) | ❌ (iterative generation, no static graph) | ❌ |
| StreamingSpan | ❌ (requires runtime state/cache updates) | ❌ |

Use the PyTorch runtime for unsupported architectures.

## PyTorch-only options are rejected on a runtime graph

`variant`, `dtype`, `quantize=`, `compile_torch_model`, and `low_cpu_mem_usage`
([gliner1-performance.md](gliner1-performance.md)) configure PyTorch loading and cannot be applied
while loading an exported ONNX/OpenVINO graph — choose precision at export time instead.

## Validate an export

```bash
python test_onnx.py exports/onnx/model.onnx
python test_onnx.py exports/onnx/model.onnx --runtime openvino
python test_onnx.py exports/openvino/model.xml --runtime openvino
```

Compare entity text/offsets/labels on a representative dataset with a score **tolerance**, not
exact equality — floating-point results differ slightly across backends.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Runtime artifact not found | `runtime_model_file` must name the file inside the directory passed as the first arg; keep OpenVINO `.xml`/`.bin` together |
| `OpenVINO is not installed` | `pip install "gliner[openvino]"` |
| ONNX export says `onnx` missing | `pip install onnx` |
| Architecture can't export/load | Decoder and StreamingSpan models require the PyTorch runtime |
| PyTorch-only option rejected | Set precision/quantization at export time, not on the runtime-graph load |
