# GLiNER (v1): Advanced Usage

Core API surface beyond the quickstart: output shape, batching, label descriptions, thresholds,
multi-label/nested decoding, local/offline loading, and — most important — the **input-length
truncation gotcha**. Condensed from
[Advanced Usage](https://urchade.github.io/GLiNER/usage.html). See
[gliner1-performance.md](gliner1-performance.md) for dtype/quantization/compile/FlashDeBERTa,
[gliner1-configs-architectures.md](gliner1-configs-architectures.md) for picking an architecture,
and [gliner1-streaming.md](gliner1-streaming.md) for incremental text.

## Output shape

```python
entities = model.predict_entities(text, labels, threshold=0.5)
# [{'start': int, 'end': int, 'text': str, 'label': str, 'score': float}, ...]
```

`start`/`end` are character offsets into the original text, half-open `[start, end)`.

Native PyTorch models (not ONNX/OpenVINO exports) can also return contextual vectors for
downstream use, disabled by default:

```python
entities = model.predict_entities(text, labels, return_vectors=True, return_label_vectors=True)
span_vector = entities[0]["vector"]        # contextual representation of the matched span
label_vector = entities[0]["label_vector"] # representation of the matched label prompt
```

Both are detached CPU `float32` NumPy arrays (not JSON-serializable — call `.tolist()`). Only
entities that survive thresholding get vectors. `inference()` supports the same two flags for
batched input. Relation-extraction models add `head_relation_vector`/`tail_relation_vector` (triple
scorers) or `vector` (pair-projection scorers) to relation dicts under the same flags.

## Batch processing

```python
texts = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google LLC is headquartered in Mountain View.",
]
labels = ["organization", "person", "location"]

all_entities = model.inference(texts, labels, batch_size=3, threshold=0.5)
for i, entities in enumerate(all_entities):
    for entity in entities:
        print(f"  - {entity['text']} ({entity['label']}): {entity['score']:.2f}")
```

`inference()` is the batched form of `predict_entities()` — always prefer it over a Python loop for
GPU utilization and throughput.

## Label descriptions

Pass a `{label: description}` dict instead of a bare list — the dict **key** is what comes back in
`entity["label"]`, the **value** is what gets encoded as the label prompt:

```python
labels = {
    "person": "A human individual, including fictional characters",
    "organization": "A company, institution, agency, or other group of people",
}
entities = model.predict_entities(text, labels)
```

- Same dict shared across a batch: pass it directly to `inference(texts, labels)`.
- Different labels/descriptions per text: pass a **list of one dict per text**, in order — each
  dict holds the *complete* label set for its text (a list of single-key dicts means "N texts with
  one label each," not "one text with N labels").
- Descriptions must have string keys/values, unique within each dict, and aren't supported with
  precomputed prompt embeddings ([gliner1-performance.md](gliner1-performance.md)).

## ⚠️ Input limits and truncation

**GLiNER does not automatically window long documents.** This is the single most common
production correctness bug with this library.

- `config.max_len` (default `384`, checkpoint-dependent) counts **splitter tokens**, not
  characters, whitespace-words, or transformer subwords. The default `whitespace` splitter also
  splits punctuation: `"Acme, Inc."` → 4 tokens (`Acme`, `,`, `Inc`, `.`).
- If `num_tokens > config.max_len`, GLiNER (0.2.27) emits a `UserWarning`, silently **keeps only
  the first `max_len` tokens**, and returns ordinary-looking predictions with **no truncation
  flag**. Text after the cutoff is never seen by the model.
- Labels don't reduce this text budget at the `max_len` stage — but for uni-encoder architectures,
  label prompt subtokens *do* share the transformer's later subword/context sequence with the
  text, so more labels can still starve the effective remaining room even though `max_len` itself
  is unaffected. Bi-encoders keep label and text sequences separate, so this doesn't apply to them.
- Python warning de-duplication and log filtering mean you cannot rely on the warning being visible
  per-request. There is no `return_truncation_info` or `truncation="error"` mode as of 0.2.27.

**Production preflight** — check before inference, without relying on internals:

```python
def truncation_info(model, text, labels):
    prepared = model.prepare_batch(text, labels)
    num_tokens = len(prepared["tokens"][0]) if prepared["tokens"] else 0
    max_len = model.config.max_len
    return {"truncated": num_tokens > max_len, "num_tokens": num_tokens, "max_len": max_len}

info = truncation_info(model, text, labels)
if info["truncated"]:
    raise ValueError(f"GLiNER input has {info['num_tokens']} tokens; limit is {info['max_len']}")
```

**Processing long documents** — split into overlapping windows no longer than `max_len`, using the
processor's own token→character maps so window boundaries land on splitter-token edges:

```python
def iter_gliner_windows(model, text, labels, overlap):
    prepared = model.prepare_batch(text, labels)
    if not prepared["tokens"]:
        return
    tokens = prepared["tokens"][0]
    starts, ends = prepared["start_token_map"][0], prepared["end_token_map"][0]
    window_size = model.config.max_len
    step = window_size - overlap
    for first in range(0, len(tokens), step):
        last = min(first + window_size, len(tokens))
        yield starts[first], text[starts[first]:ends[last - 1]]
        if last == len(tokens):
            break
```

For span models use `overlap >= max_width - 1` so no entity is split across windows. You must merge
window outputs yourself: shift `entity["start"]`/`["end"]` by the window's `char_start`, then
group by `(start, end, label)` and keep the highest score. Relations whose endpoints never co-occur
in one window can't be recovered.

To raise the limit at load time (doesn't resize the backbone or guarantee quality beyond training
length — prefer windowing unless you know the checkpoint supports it):

```python
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", max_length=512)
```

| Architecture | Label prompt encoding | Effective-limit notes |
|---|---|---|
| UniEncoderSpan / UniEncoderToken | Shares one backbone sequence with text | `max_len` is text-only; a finite subword/backbone limit still includes the prompt |
| UniEncoder decoders | Same as above for the main encoder; decoder has separate generation limits | |
| UniEncoder relation extraction | Entity + relation prompts share the sequence | Same two-stage behavior |
| BiEncoderSpan / BiEncoderToken | Text and labels encoded **separately** | Label count doesn't consume the text sequence |
| StreamingSpan, no `session_id` | Shares one causal sequence with text | Normal stateless `max_len` truncation applies |
| Cached StreamingSpan session | Prompt + all appended text share the decoder context | `max_len` truncation is bypassed; exceeding `max_cache_length` raises `ValueError` instead of silently dropping text |

`InferencePackingConfig.max_length` ([gliner1-performance.md](gliner1-performance.md)) is a
**separate** limit measured in already-tokenized backbone IDs, not splitter tokens — it can also
silently truncate. `max_width` is not a length limit; it bounds candidate span width for span
architectures.

## Choosing an architecture (quick reference)

Full decision guide, config parameters, and training YAML:
[gliner1-configs-architectures.md](gliner1-configs-architectures.md). One line each:

```python
# UniEncoder — general purpose, < ~30 entity types
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# BiEncoder — many entity types (50-200+), pre-computable label embeddings
model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")
label_embeddings = model.encode_labels(labels, batch_size=16)      # once
entities = model.predict_with_embeds(text, label_embeddings, labels, threshold=0.5)  # reuse

# Token-level — long/multi-sentence entities
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

# Relation extraction — entities + typed relations jointly
model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")
entities, relations = model.inference(
    [text], labels=entity_labels, relations=relation_labels,
    threshold=0.5, relation_threshold=0.5,
)
head = entities[0][relations[0][0]["head"]["entity_idx"]]
```

For relation extraction, `adjacency_threshold` (default = `threshold`) governs pair-candidate
reconstruction and `relation_threshold` governs the final relation label — set
`adjacency_threshold` lower than `relation_threshold` to let more candidate pairs through for
reranking while keeping final relations precise.

## Thresholds, multi-label, flat vs nested

```python
entities_high = model.predict_entities(text, labels, threshold=0.7)  # higher precision
entities_low = model.predict_entities(text, labels, threshold=0.3)   # higher recall
```

```python
# Multi-label: allow a span to carry more than one type (works around the "one label per
# span" limitation noted in gliner1-intro.md)
entities_multi = model.predict_entities(text, labels, multi_label=True)
```

```python
# Flat NER (default): no overlapping entities, longest/highest-scoring span wins
entities_flat = model.predict_entities(text, labels, flat_ner=True)

# Nested NER: allow overlapping entities except partial overlaps
entities_nested = model.predict_entities(text, labels, flat_ner=False)
```

## Local, offline, and device loading

```python
model = GLiNER.from_pretrained("/path/to/local/model")
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", cache_dir="./model_cache")
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", map_location="cuda")  # or "cpu"
print(model.device)
```

Offline deployment — resolve dependencies once, then copy the directory:

```python
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
model.save_pretrained("gliner-offline", safe_serialization=True)
# copy gliner-offline/ (including labels_tokenizer/ or decoder_tokenizer/ if present) to the target
model = GLiNER.from_pretrained("gliner-offline", local_files_only=True)
```

`local_files_only=True` restricts to local/cached files; it does not fetch missing dependencies.
Older Hub checkpoints that only name a bare backbone (e.g. `microsoft/mdeberta-v3-base`) need the
load-then-`save_pretrained` round trip above to package a self-contained directory.

## Tips and best practices

1. Pick the architecture for your constraint (< 30 types → UniEncoder; 50-200+ → BiEncoder; long
   spans → token-level; incremental text → StreamingSpan).
2. Threshold: `0.6-0.8` high precision, `0.4-0.6` balanced, `0.2-0.4` high recall.
3. Always batch multiple documents through `inference()`, never loop `predict_entities()`.
4. BiEncoder: pre-compute label embeddings once, reuse across the whole corpus.
5. Enable FlashDeBERTa for ~3x speedup with no accuracy loss (see
   [gliner1-performance.md](gliner1-performance.md)).
6. Specific labels beat generic ones: `"tech_company"` > `"organization"` > `"entity"`.

## Troubleshooting

| Symptom | Try |
|---|---|
| Low accuracy | Lower `threshold`; use more specific labels; try a larger model (`gliner_large-v2.1`) |
| Slow inference | FlashDeBERTa; `compile_torch_model=True`; batch via `inference()`; BiEncoder + precomputed embeddings |
| Out of memory | Smaller `batch_size`; smaller model; `map_location="cpu"` |
| Entities missing near the end of a long doc | You are hitting `max_len` truncation — see "Input limits and truncation" above |
