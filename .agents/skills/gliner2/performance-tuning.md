# Performance Tuning: Quantization, Compilation, Word Splitters, FlashDeBERTa

Runtime knobs that change *how fast* or *on what hardware* a model runs, without changing which
extraction method or schema you use. None of these require re-downloading a checkpoint or
retraining.

## CPU / no-GPU deployments

The knobs below are framed for GPU (`quantize`/`compile` examples use `map_location="cuda"`,
FlashDeBERTa is NVIDIA-only). For CPU-only or edge deployment, the lever that actually matters is
**checkpoint size**, not a runtime flag: use `fastino/gliner2.5-small-v1` (74M params,
DeBERTa-v3-xsmall) instead of `-base-v1` (194M) or `-multi-v1` (287M) — see
[SKILL.md](SKILL.md)'s model catalog.

Measured on one machine (10-core Apple Silicon, CPU only, `batch_size=1`, 5-label entity
extraction on short sentences — a directional anchor for capacity planning, not a universal
benchmark; re-measure on your own hardware):

| Checkpoint | Throughput | 50,000 docs, single-threaded |
|---|---|---|
| `gliner2.5-small-v1` | ~81 docs/sec | ~10 min |
| `gliner2.5-base-v1` | ~33 docs/sec | ~25 min |

Scale further by batching across processes/workers, not by reaching for `quantize`: verified
`quantize=True` (`from_pretrained`'s boolean flag casts to fp16 by default, same as calling
`model.quantize("fp16")`) makes CPU throughput **2.7x worse** (81 → 30 docs/sec on `small-v1`) —
most CPUs have no native fp16 compute and emulate it, and `quantize()` only supports fp16/bf16
casts, not true int8. It is a GPU-only lever; skip it entirely without a GPU.

## Quantization and `torch.compile`

Both are opt-in at load time (or after loading) and need no extra dependencies:

```python
from gliner2 import AutoExtractor

# fp16
model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda", quantize=True)

# torch.compile (fused GPU kernels; first call triggers tracing, so it is slower)
model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", map_location="cuda", compile=True)

# Both
model = AutoExtractor.from_pretrained(
    "fastino/gliner2.5-base-v1", map_location="cuda", quantize=True, compile=True
)

# Or apply after loading
model.quantize()
model.compile()
```

`compile=True`'s tracing cost means the first `extract_*` call after load is markedly slower —
warm the model with a throwaway call before timing or serving real traffic.

## Custom word splitters

GLiNER2 first splits text into word tokens, then runs the model's subword tokenizer over those.
The default `"whitespace"` splitter is what public checkpoints were trained with — changing it on
a pretrained model can hurt quality unless the model was itself trained with the new splitter.

Use the character-level splitter for languages without whitespace-delimited words (e.g. Chinese,
Japanese) — **necessary, but not sufficient, for Japanese quality.** Verified empirically on
`gliner2.5-multi-v1`: with the default whitespace splitter, a Japanese sentence with a person and
an organization came back with **both entirely missed** — one garbage 18-character span
mislabeled as a location instead. Switching to `word_splitter="char"` recovered the person
(0.99 confidence) and the location (0.99 confidence), but the organization was still weak and
duplicated into the location bucket too (`organization: 0.51` vs. an incorrect `location: 0.85`
hit on the same span) — real residual cross-label ambiguity the splitter doesn't fix. Add
descriptions to disambiguate Japanese labels (see the "descriptions beat bare label lists" rule
in [SKILL.md](SKILL.md)) rather than assuming the splitter alone gets you English/Spanish-level
quality:

```python
model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1", word_splitter="char")

# Or after loading
model.set_word_splitter("char")
```

| Name | Class | Use when |
|---|---|---|
| `"whitespace"` (default) | `WhitespaceTokenSplitter` | Space-delimited languages; matches public checkpoints |
| `"char"` | `CharLevelSplitter` | Chinese and similar; keeps Latin words/emails intact, splits other non-space characters |

A custom splitter is any callable yielding `(token, start, end)` tuples with exclusive-end
offsets into the *original* text:

```python
from gliner2.processor import CharLevelSplitter

model.set_word_splitter(CharLevelSplitter())
```

The choice is runtime-only — a saved checkpoint reloads with `"whitespace"` unless you pass
`word_splitter` again on that later `from_pretrained` call.

## FlashDeBERTa (optional GPU acceleration)

For DeBERTaV2-based checkpoints (all public GLiNER2 models), [FlashDeBERTa](https://github.com/fastino-ai/flashdeberta)
accelerates inference on NVIDIA GPUs via flash-attention kernels — works for both `span` and
`boundary` architectures.

```bash
pip install flashdeberta
```

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained(
    "fastino/gliner2-base-v1",
    use_flashdeberta=True,
    map_location="cuda",
)
model.half().eval()  # FP16/BF16 on CUDA is required to realize the speedup

result = model.extract_entities(
    "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"],
)
```

It only activates when both the encoder is DeBERTaV2 *and* the `flashdeberta` package is
installed; otherwise it silently falls back to the standard Hugging Face encoder. For backward
compatibility, `USE_FLASHDEBERTA=1` enables it when `use_flashdeberta` is omitted; passing
`use_flashdeberta=False` explicitly always overrides the environment variable.

## Best practices

- These are independent knobs — quantize, compile, a custom word splitter, and FlashDeBERTa can
  all be combined on the same model.
- None of them change output shape or schema semantics; they are pure runtime/latency levers.
- Benchmark on your own hardware and checkpoint before committing to `compile=True` or
  FlashDeBERTa in production — the first-call tracing cost and GPU-only applicability mean they
  are not free wins for CPU or low-QPS deployments.
