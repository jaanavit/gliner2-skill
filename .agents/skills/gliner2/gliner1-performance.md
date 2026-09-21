# GLiNER (v1): Performance Tuning

Load-time and inference-time levers for latency, memory, and cost — precision, quantization,
compilation, FlashDeBERTa, sequence packing, and prompt-embedding precomputation. Condensed from
[Advanced Usage](https://urchade.github.io/GLiNER/usage.html). For the GLiNER2 equivalents, see
[performance-tuning.md](performance-tuning.md).

## Reduced-precision loading (`dtype`)

Loads weights directly at the target precision — no intermediate fp32 copy, no post-load cast:

```python
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", dtype="bf16", map_location="cuda")
# or dtype=torch.bfloat16
```

Accepts `"fp16"`/`"float16"`/`"half"`, `"bf16"`/`"bfloat16"`, `"fp32"`/`"float32"`/`"float"`, or a
`torch.dtype`. `dtype=` (not `quantize=`) is the only way to get half precision now — passing
`quantize=True`/`"fp16"`/`"bf16"` raises with a migration message. Matters most for cold starts on
serverless/autoscaled deployments: shorter load, lower peak memory, faster first inference.

### Skip the random-init shell (`low_cpu_mem_usage`)

`dtype=` alone still builds a full fp32 random-initialized shell before overwriting it.
`low_cpu_mem_usage=True` builds the graph under `torch.device("meta")` (no allocation, no random
init) and assigns loaded tensors directly:

```python
model = GLiNER.from_pretrained(
    "urchade/gliner_medium-v2.1", dtype="bf16", low_cpu_mem_usage=True, map_location="cuda",
)
```

Measured on `gliner_medium-v2.1` (RTX 5090): ~2x faster load, 23-89% lower peak host RSS depending
on dtype. Bit-identical loaded parameters. Off by default (path is newer); stacks with `dtype=`,
independent of `quantize=`/`compile_torch_model=`.

### Selective download (`variant`)

`dtype=` casts in memory but still downloads the full fp32 file. If the publisher uploaded a
half-precision file (`model.fp16.safetensors` / `model.bf16.safetensors`):

```python
model = GLiNER.from_pretrained("org/gliner_bf16-v1", variant="bf16")
# ~745MB -> ~370MB download for gliner_medium-v2.1-sized checkpoints, when published
```

Best-effort: if the variant file isn't published, falls back to the fp32 file + in-memory `dtype=`
cast with a `UserWarning` — no error, just no bandwidth savings. `dtype=` is inferred from
`variant=`; passing both with mismatched precision raises.

## Quantization vs. dtype

| | `dtype="fp16"/"bf16"` | `quantize="int8"` |
|---|---|---|
| What | Plain precision downcast at load | Real int8 quantization |
| CPU | — | FBGEMM kernels, ~1.6x speedup |
| GPU | Half-precision inference | torchao int8 weight-only, ~50% memory, no speed gain |
| When | Always safe | Best for checkpoints fine-tuned with quantization-aware training — stock DeBERTa checkpoints lose accuracy |

`quantize=` now accepts only `"int8"` or `None`.

## `torch.compile`

```python
model = GLiNER.from_pretrained(
    "urchade/gliner_medium-v2.1", map_location="cuda", dtype="fp16", compile_torch_model=True,
)
# or after the fact:
model.to(torch.float16)
model.compile()
```

Combine `dtype="fp16"` + compile for ~1.9x GPU speedup at zero F1 loss (CoNLL-2003, RTX 5090:
0.8107 F1 baseline → 0.8107 F1 at 1.94x). Best for **short sequences** (eager overhead is
proportionally larger there); for long sequences, FlashDeBERTa scales better. Linux/WSL only — not
native Windows or macOS. First call after compiling is slower (JIT); subsequent calls benefit.

## FlashDeBERTa

Most GLiNER checkpoints use a DeBERTa backbone, which historically had no flash-attention kernel
for its disentangled attention. FlashDeBERTa fixes that — up to 3x speedup, no accuracy loss,
larger gains for longer sequences.

```bash
pip install flashdeberta -U   # requires transformers>=4.51.3
```

```python
import os
os.environ["USE_FLASHDEBERTA"] = "1"   # or: export USE_FLASHDEBERTA=1
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# To force eager attention instead:
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", _attn_implementation="eager")
```

## Sequence packing

Packs multiple short requests into one transformer pass with a block-diagonal attention mask,
cutting padding-token overhead:

```python
from gliner import GLiNER, InferencePackingConfig

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", map_location="cuda")
packing_cfg = InferencePackingConfig(
    max_length=512,   # backbone token IDs, NOT splitter tokens — see the warning below
    sep_token_id=model.data_processor.transformer_tokenizer.sep_token_id,
    streams_per_batch=1,
)
model.configure_inference_packing(packing_cfg)   # applies to every subsequent inference call

predictions = model.inference(texts, labels, batch_size=16)
```

Override or disable per call with `packing_config=<cfg>` / `packing_config=None`.

> **Truncation warning:** `InferencePackingConfig.max_length` is a *separate* limit from
> `config.max_len` ([gliner1-usage.md](gliner1-usage.md)) and is measured in already-tokenized
> backbone IDs. An over-length packed request is silently truncated to `max_length` IDs with **no**
> `config.max_len` warning. Size it for the full encoded request (including any uni-encoder label
> prompt), or disable packing for requests that might exceed it.

## Prompt compression (precomputed prompt embeddings)

For uni-encoder models with a **fixed** label set: precompute per-label prompt embeddings once so
the encoder only sees the text at inference (shorter sequence, less attention cost, small accuracy
trade-off vs. re-encoding prompts every call).

```python
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

calibration_texts = [...]  # 100-1000 diverse in-domain sentences; no labels needed, just context
labels = ["person", "organization", "location", "date"]
model.compress_prompt_embeddings(calibration_texts, labels, batch_size=16)

# Inference now uses the stored embeddings — pass the same label set (order-insensitive)
entities = model.predict_entities("Tim Cook visited Berlin last Tuesday.", labels, threshold=0.5)
model.save_pretrained("./gliner-compressed")   # embeddings travel with state_dict
```

For relex models, also pass `rel_labels=[...]` so relation prompts get compressed too.

Compression alone can lose context-specific signal. Recover it with `distill=True` — the raw model
generates pseudo-labels over `texts`, then the compressed model is fine-tuned on them in the same
call:

```python
model.compress_prompt_embeddings(
    texts=calibration_texts, labels=labels, batch_size=16,
    distill=True, distill_threshold=0.3, distill_epochs=3, distill_lr=1e-5,
    distill_output_dir="./distill_ckpt",
)
```

Not supported together with label-description dicts ([gliner1-usage.md](gliner1-usage.md)).

## Quick recipe by goal

| Goal | Do |
|---|---|
| Fastest cold start / lowest memory | `dtype="bf16"` + `low_cpu_mem_usage=True`; `variant="bf16"` if published |
| Fastest GPU steady-state, short texts | `dtype="fp16"` + `compile_torch_model=True` |
| Fastest GPU steady-state, long texts | FlashDeBERTa |
| Many small concurrent requests | Sequence packing |
| Fixed label set, called constantly | `compress_prompt_embeddings` |
| Many entity types (50-200+) | BiEncoder + `encode_labels()` once, reuse embeddings |
