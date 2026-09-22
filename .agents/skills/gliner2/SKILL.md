---
name: gliner2
description: Build GLiNER2 (gliner2 PyPI package / fastino-ai/GLiNER2) schemas and pick the right extraction method for a use case — text classification, entity extraction, structured/JSON extraction, relation extraction, combined multi-task schemas, regex validators, span attributes, constrained classification, joint entity-relation extraction, long-context/chunked extraction, LoRA training, adapter switching, and Pioneer's hosted inference API. Also covers the original GLiNER (v1, `gliner` PyPI package / urchade/GLiNER) — single-task NER, architectures (UniEncoder/BiEncoder/Decoder/Relex/StreamingSpan), training, ONNX/OpenVINO export, and Ray Serve deployment. Use whenever the user mentions GLiNER2, gliner2, AutoExtractor, fastino/gliner2, GLiNER2.5, GLiNER, gliner, urchade, predict_entities, schema-based NER/classification/extraction, or asks which GLiNER method/package fits their use case.
---

# GLiNER2

This file walks the GLiNER2 pipeline in order: **set up the environment → pick a model → route
by task → build and run a schema → fine-tune locally (optional) → inference locally.** If you
actually need the original GLiNER (v1) package, skip to the
[appendix](#appendix-gliner-vs-gliner2--which-package) at the bottom instead.

## 1. Set up the environment — do this before anything else

Run these four checks **in order**. Stop and fix at the first failure; do not write schema code
until all four pass. Skipping this and going straight to `pip show gliner2` / picking a model is
the single most common way agents burn time on this skill.

1. **Python version first.** `python3 --version` must be **3.10+**. `gliner2`'s type hints use
   `X | None` syntax, which raises `TypeError: unsupported operand type(s) for |` on import under
   3.9 — a confusing, load-time failure that looks unrelated to Python version. A venv alone does
   not fix this; the interpreter itself must be 3.10+.
2. **Install into a clean venv on that interpreter — don't trust an existing install.**
   `pip show gliner2` reporting "installed" does **not** mean it's usable: it may be a stale
   version, or one installed against a pre-3.10 interpreter. Likewise `pip index versions
   gliner2` can under-report the true latest release. Create a fresh venv and install there
   rather than debugging an existing site-packages install.
3. **Install the extra(s) your task needs — the base package alone is not enough for most work:**
   ```bash
   pip install gliner2                                   # Schema, RegexValidator, InputExample/TrainingDataset -- no torch required
   pip install "gliner2[local]" protobuf sentencepiece    # + local model inference (AutoExtractor, LoRA)
   pip install "gliner2[train]"                           # + training (ExtractorTrainer, TrainingConfig)
   ```
   `gliner2[local]` alone is not sufficient for local inference — loading a checkpoint's
   tokenizer via `AutoExtractor.from_pretrained(...)` raises `ImportError: ... requires the
   protobuf library` without `protobuf` and `sentencepiece` also installed.
4. **Smoke-test with a real import, not `pip show`:**
   ```bash
   python -c "from gliner2 import AutoExtractor; print(AutoExtractor)"
   ```
   Only move on once this succeeds with no traceback.

Loading a `fastino/gliner2.5-*-v1` checkpoint (e.g. via `AutoExtractor.from_pretrained(...)`)
prints three warnings every time — none are errors, all are safe to ignore:
`UserWarning: Checkpoint uses legacy list-valued extra_special_tokens metadata...`,
`FutureWarning: torch.jit.script is deprecated...`, and `RuntimeWarning: Encoder rejected
attn_implementation='sdpa'; falling back to 'eager'...`. They look alarming on a first run but
don't affect correctness or require any action.

## 2. Pick and download a model

GLiNER2 is a schema-conditioned encoder for **NER, text classification, structured/JSON
extraction, relation extraction, and span attributes** in a single forward pass — CPU-capable,
no LLM required. Two architectures share one public API, both loaded via `AutoExtractor`, which
dispatches on the checkpoint's saved `architecture` field:

- **`boundary` (GLiNER2.5)** — sparse start/end pairing, any span length within the encoded
  window. **Default choice for new work.** Dropping explicit span enumeration is also what lets
  it train on and process sequences up to ~4,096 words in a single forward pass — most
  reports/contracts/transcripts fit without chunking; see [long-context.md](long-context.md) for
  when you still need it.
- **`span` (GLiNER2 / legacy)** — fixed-width span grid. Only needed for legacy checkpoints and
  the specialty safety/PII fine-tunes.

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")   # English, default
# model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1") # Multilingual
# model = AutoExtractor.from_pretrained("fastino/gliner2.5-small-v1") # Fast / CPU / edge
```

`GLiNER2.from_pretrained(...)` remains span-only and will not load GLiNER2.5 boundary
checkpoints — use `AutoExtractor` unless you specifically know you want the legacy loader.

### GLiNER2.5 boundary checkpoints

The three checkpoints ([Hub collection](https://huggingface.co/collections/fastino/gliner25-models))
share one API, `enable_relations=True` + `enable_records=True` (every routing-table capability
below works on all three), and a `max_len=4096` window — differing only in speed/size.

| Checkpoint | Params | Encoder | Language | Use case |
|---|---|---|---|---|
| `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall | English | Fastest / CPU / edge |
| `fastino/gliner2.5-base-v1` | 194M | DeBERTa-v3-base | English | **Default** English checkpoint |
| `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa-v3-base | Multilingual | Default multilingual checkpoint |

### Legacy span checkpoints and specialty fine-tunes

Sizes vary per checkpoint, not uniform:

| Checkpoint | Hub size | Goal |
|---|---|---|
| `fastino/gliner2-base-v1` | 0.2B | Legacy span, English, multi-task |
| `fastino/gliner2-large-v1` | 0.5B | Legacy span, higher accuracy, English |
| `fastino/gliner2-multi-v1` | 0.3B | Legacy span, multilingual, multi-task |
| `fastino/gliguard-LLMGuardrails-300M` | 0.2B | LLM prompt/response guardrails |
| `fastino/gliner2-privacy-filter-PII-multi` | 0.3B | PII redaction — see caveat in [safety-pii.md](safety-pii.md) |
| `fastino/GLiNER2-Guardrails-PII-Multi` | 0.3B | Guardrails + PII combined |

Sizes are each checkpoint's Hub "Model size" badge — trust the badge over a card's prose if they
disagree (some cards predate a later retrain). Full model cards:
[GLiNER2 README](https://github.com/fastino-ai/GLiNER2#-available-models).

## 3. Route by task

Each row links to a reference file with full API detail, parameters, and worked examples — read
only the file(s) the task needs.

| I need to... | Use | Reference |
|---|---|---|
| Bucket text into categories (sentiment, topic, intent) | `classify_text()` / `.classification()` | [classification.md](classification.md) |
| Pull named spans out of text (people, orgs, dates, custom types) | `extract_entities()` / `.entities()` | [entity-extraction.md](entity-extraction.md) |
| Parse a record/JSON shape (product, invoice, contact) | `extract_json()` / `.structure()` | [json-extraction.md](json-extraction.md) |
| Do 2+ of the above in one pass over the same text | `create_schema()` chaining | [combined-schemas.md](combined-schemas.md) |
| Extract `(head, tail)` pairs like works_for/located_in, no typed endpoints needed | `extract_relations()` / `.relations()` | [relation-extraction.md](relation-extraction.md) |
| Filter/validate extracted spans with a regex (email, phone, URL) | `RegexValidator` (local models only) | [regex-validators.md](regex-validators.md) |
| No GPU, or want hosted inference/training — sign up at [agent.pioneer.ai/auth](https://agent.pioneer.ai/auth), then call Pioneer's `api.pioneer.ai/inference` REST endpoint (model selection, fine-tuned job IDs, raw HTTP) | `POST /inference` | [pioneer-api.md](pioneer-api.md) |
| Attach a label (e.g. sentiment) to each extracted entity span, not the whole doc | `entity_attributes()` + `AttributeGroup` (GLiNER2.5 only) | [span-attributes.md](span-attributes.md) |
| Enforce hard rules between classification tasks (e.g. intent=delete ⇒ effects includes delete) | `Classifier` + constraint DSL (GLiNER2.5 only) | [constrained-classification.md](constrained-classification.md) |
| Extract entities AND relations as one consistent typed graph (unique employer, no self-loops, etc.) | `JointIE` (GLiNER2.5 + `enable_relations=True`) | [joint-ie.md](joint-ie.md) |
| Process a document longer than the model's context window | `*_long` methods (`extract_entities_long`, `extract_long`, ...) | [long-context.md](long-context.md) |
| Redact PII or moderate LLM prompts/responses | Specialty checkpoints (GLiGuard, PII filter) | [safety-pii.md](safety-pii.md) |
| Speed up inference (fp16, `torch.compile`, FlashDeBERTa) or extract CJK text (word splitters) | `quantize=True`, `compile=True`, `word_splitter="char"`, `use_flashdeberta=True` | [performance-tuning.md](performance-tuning.md) |

Zero-shot accuracy insufficient for one of these? That's not a routing problem — see
[§5 Fine-tune locally](#5-fine-tune-locally) below rather than switching methods.

### Programmatic routing

For a non-agent caller (CLI, docs bot, pre-flight check), [`route_usecase.py`](route_usecase.py)
scores a free-text task description against this table's rows using a live GLiNER2 classifier —
treat its output as a ranked shortlist, not ground truth. `test_route_usecase.py` keeps
`USE_CASES` in sync with every reference file in this directory (including §5's); tuning notes
live in that file's own docstring.

## 4. Build and run a schema

**Descriptions beat bare label lists.** Passing `{"label": "description"}` instead of
`["label"]` consistently improves accuracy — true for entities, classifications, relations, and
JSON fields alike. This is the single highest-leverage accuracy lever across every task type.

**Decision shortcuts:**

- Only need one task, independently decoded, no cross-task rules → the quick methods
  (`extract_entities`, `classify_text`, `extract_json`, `extract_relations`) are simplest and
  sufficient.
- Need 2+ tasks over the *same text in one pass*, or need per-field thresholds/validators →
  build with `model.create_schema()....` and call `model.extract(text, schema)`.
- One label legally constrains another, or you need a globally consistent entity–relation graph
  → do **not** just combine independent calls; independent decoding cannot enforce cross-task
  consistency. Use `Classifier` or `JointIE` respectively (see routing table above).
- Text may exceed the model's window (reports, contracts, transcripts, logs) → use the `*_long`
  variant. Never rely on `max_len` truncation for this — it silently drops the rest of the file.

```python
# Quick method -- one task, simplest path
result = model.extract_entities(
    "Apple hired Jane Doe.", {"company": "business name", "person": "full name"}
)
# {'entities': {'company': ['Apple'], 'person': ['Jane Doe']}}

# create_schema() chaining -- 2+ tasks over the same text in one pass
schema = (
    model.create_schema()
    .entities({"company": "business name"})
    .classification("sentiment", ["positive", "negative", "neutral"])
)
result = model.extract("Apple's new office is fantastic.", schema)
# {'entities': {'company': ['Apple']}, 'sentiment': 'positive'}
```

## 5. Fine-tune locally

Only worth doing once zero-shot accuracy is **measured**, not eyeballed, and plateaus after
description-tuning (§4) — the biggest remaining accuracy lever, but don't assume a specialized
domain needs it by default: GLiNER2.5's training data spans many domains by design, so measure
before fine-tuning, not instead of it.

1. Measure zero-shot precision/recall/F1 on a held-out set, and compare any fine-tuned candidate
   on the same set before shipping either → [evaluation.md](evaluation.md)
2. If it plateaus, label ~100–200 of the actual misses into a JSONL training set →
   [training-data-format.md](training-data-format.md)
3. Fine-tune with `ExtractorTrainer`/`TrainingConfig` → [training.md](training.md), or train a
   small parameter-efficient adapter per domain instead of a full checkpoint (`use_lora=True`) →
   [lora-adapters.md](lora-adapters.md)
4. If you trained multiple domain adapters, swap between them at inference time without
   reloading the base model (`load_adapter()`/`unload_adapter()`) →
   [adapter-switching.md](adapter-switching.md)

## 6. Inference locally

Everything above runs through a locally loaded `AutoExtractor` by default — no network call,
your model and data stay on the machine that loaded them. Output shape and volume are controlled
by flags present on nearly every method:

- `include_confidence=True` — text becomes `{'text': ..., 'confidence': 0.92}`
- `include_spans=True` — adds `{'start': int, 'end': int}` (character offsets, half-open
  `[start, end)` into the original string)
- Both `False` (default) — plain strings / tuples.
- Both `True` — `{'text', 'confidence', 'start', 'end'}`.

Always verify `text[start:end] == extracted_text` while developing a new schema.

**`threshold`** — set globally per call, or per-field/entity/relation/classification via schema
dicts (`{"threshold": 0.8}`). Push higher (0.7–0.9) for precision-critical fields (SSNs, account
numbers), lower (0.3–0.5) for multi-label recall.

**All requested keys always appear in the output**, even when empty (`[]`), across entities,
relations, and classifications — safe to iterate without existence checks.

**Batch methods** (`batch_extract_entities`, `batch_extract_relations`, `batch_extract_json`,
`batch_classify_text`, ...) take `batch_size=` and return one result per input, same shape as the
singular call.

Don't want to load a model locally at all, or don't have a GPU? Sign up at
[agent.pioneer.ai/auth](https://agent.pioneer.ai/auth) and use
[pioneer-api.md](pioneer-api.md) — Pioneer's hosted `api.pioneer.ai/inference` REST endpoint
gives the same method surface (plus hosted training) over a plain HTTP call instead.

## Appendix: GLiNER vs GLiNER2 — which package?

This skill covers **two different PyPI packages** from the same lineage. Everything in §§1–6
above is **GLiNER2** (`gliner2`) — it's a superset: schema-driven multi-task extraction (NER +
classification + JSON + relations + attributes in one pass) and the actively developed line.
Reach for the original **GLiNER (v1)** (`gliner`, `gliner1-*.md` files in this
same directory) only when you specifically need something GLiNER2 doesn't have:

| Need | Package | Start at |
|---|---|---|
| Anything in §§1–6 above, or unsure | **GLiNER2** (default) | §3's routing table |
| A specific `urchade/gliner_*` or `knowledgator/gliner-*` (non-fastino) checkpoint by name | GLiNER v1 | [gliner1-intro.md](gliner1-intro.md) |
| Incremental/streaming NER over live text (append-and-revise, causal KV cache) | GLiNER v1 (`StreamingSpan`) | [gliner1-streaming.md](gliner1-streaming.md) |
| Ray Serve production HTTP deployment with PolyLoRA multi-adapter routing | GLiNER v1 | [gliner1-serving.md](gliner1-serving.md) |
| ONNX/OpenVINO export of a NER-only model | GLiNER v1 | [gliner1-onnx-export.md](gliner1-onnx-export.md) |
| Implementing a brand-new architecture variant from scratch | GLiNER v1 | [gliner1-custom-architectures.md](gliner1-custom-architectures.md) |
| Installing/pinning both packages in one project, or just want the real dependency list (not the docs site's `pip install` snippets) | Both | [gliner1-repo-and-dependencies.md](gliner1-repo-and-dependencies.md) |

The GLiNER v1 files are a self-contained sub-tree ([gliner1-intro.md](gliner1-intro.md) is their
entry point/router) — they don't feed into `route_usecase.py`, which is scoped to picking a
GLiNER2 *method*, not a package.

---

Maintaining or extending this skill (new topic files, naming conventions, install elsewhere) is
covered in [README.md](README.md), not here — that's a skill-maintenance concern, not something
needed to answer a GLiNER2 task.
