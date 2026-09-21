# GLiNER (v1): Introduction & Installation

**This is a different package from the rest of this skill.** `gliner` (this file and the other
`gliner1-*.md` files) is the original, single-task NER library by the same author (Urchade
Zaratiana) that GLiNER2 is built on top of. See [SKILL.md](SKILL.md)'s "GLiNER vs GLiNER2" section
for when to reach for this instead of GLiNER2. Mirrors the
[GLiNER docs](https://urchade.github.io/GLiNER/intro.html).

GLiNER identifies **any entity type** using a bidirectional transformer encoder (BERT-like) —
zero-shot NER without being limited to a predefined label set, and without the cost/latency of an
LLM.

## Installation

> **Installing alongside `gliner2`?** The two packages version their dependencies
> independently and `gliner`'s base install is torch-mandatory (unlike `gliner2`'s optional-torch
> design) — see [gliner1-repo-and-dependencies.md](gliner1-repo-and-dependencies.md) for the exact
> version pins and why separate virtual environments are usually the safer default.

```bash
pip install gliner                    # base — PyTorch inference and training

pip install "gliner[onnx]"            # + ONNX Runtime CPU inference
pip install "gliner[gpu]"             # + ONNX Runtime CUDA inference (pick onnx OR gpu, not both)
pip install onnx                      # additionally required to *export* to ONNX
pip install "gliner[openvino]"        # + OpenVINO IR export/inference

pip install "gliner[tokenizers]"      # + tokenizers for non-English multilingual models
pip install "gliner[stanza]"          # + Stanza tokenizer support
pip install "gliner[training]"        # + training dependencies
pip install "gliner[serve]"           # + Ray Serve production deployment
```

Conda: `conda install -c conda-forge gliner`.

## Basic usage

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

text = """
Cristiano Ronaldo dos Santos Aveiro (born 5 February 1985) is a Portuguese professional
footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and
the Portugal national team.
"""

labels = ["Person", "Award", "Date", "Competitions", "Teams"]
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
# Cristiano Ronaldo dos Santos Aveiro => person
# 5 February 1985 => date
# Al Nassr => teams
# Portugal national team => teams
```

Each result dict is `{'start': int, 'end': int, 'text': str, 'label': str, 'score': float}` —
character offsets, half-open `[start, end)`. See [gliner1-quickstart.md](gliner1-quickstart.md)
for one more minimal worked example, or jump straight to full API detail (batching, descriptions,
thresholds, multi-label, truncation) in [gliner1-usage.md](gliner1-usage.md).

## Model catalog

| Goal | Checkpoint | License |
|---|---|---|
| English, general purpose (small) | `urchade/gliner_small-v2.1` | Apache 2.0 |
| English, general purpose (medium) | `urchade/gliner_medium-v2.1` | Apache 2.0 |
| English, general purpose (large) | `urchade/gliner_large-v2.1` | Apache 2.0 |
| Multilingual | `urchade/gliner_multi-v2.1` | Apache 2.0 |
| Multilingual PII (6 languages) | `urchade/gliner_multi_pii-v1` | Apache 2.0 |
| Biomedical | `urchade/gliner_large_bio-v0.1` | Apache 2.0 |

These are the maintainer's own (`urchade/*`) checkpoints, covering the general-purpose,
multilingual, PII, and biomedical cases most tasks need. The community has published further
fine-tunes (other languages, news domain, alternate architectures) under other namespaces
(`numind/`, `knowledgator/`, etc.) — same `GLiNER.from_pretrained(...)` API applies to any of
them; see the [GLiNER intro page](https://urchade.github.io/GLiNER/intro.html) for the full list
if none of the above fits.

## Known research gaps (from the maintainers)

- A span cannot yet have multiple labels in the base decoding strategy (use `multi_label=True` —
  see [gliner1-usage.md](gliner1-usage.md) — as the current workaround).
- No dynamic thresholding: the model under-predicts on domains/entity-types poorly represented in
  training data; lowering `threshold` is a manual mitigation.

## Where to go next

| Need | File |
|---|---|
| Repo layout, exact dependency pins, coexisting with `gliner2` | [gliner1-repo-and-dependencies.md](gliner1-repo-and-dependencies.md) |
| Batching, descriptions, thresholds, truncation limits | [gliner1-usage.md](gliner1-usage.md) |
| Which architecture (UniEncoder/BiEncoder/Decoder/Relex/Streaming) and its config params | [gliner1-configs-architectures.md](gliner1-configs-architectures.md) |
| Speed/memory: dtype, quantization, `torch.compile`, FlashDeBERTa | [gliner1-performance.md](gliner1-performance.md) |
| Incremental text (live transcripts, chat, log streams) | [gliner1-streaming.md](gliner1-streaming.md) |
| Classification/QA/relation/summarization pipelines, spaCy, UTCA | [gliner1-pipelines.md](gliner1-pipelines.md) |
| Fine-tuning on your own data | [gliner1-training.md](gliner1-training.md) |
| ONNX / OpenVINO export | [gliner1-onnx-export.md](gliner1-onnx-export.md) |
| Production HTTP serving (Ray Serve) | [gliner1-serving.md](gliner1-serving.md) |
| Implementing a new architecture variant | [gliner1-custom-architectures.md](gliner1-custom-architectures.md) |
