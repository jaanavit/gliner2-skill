# Long-Context Extraction

Extract from documents longer than the model's encoded window (reports, contracts, transcripts,
logs, PDFs-as-text). Standard `extract(...)` with `max_len` **truncates**; long-context APIs
**scan** the full document with overlapping chunks and remap spans to global offsets. Mirrors
[tutorial/12-long_context.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/12-long_context.md).

Works for both span checkpoints and GLiNER2.5 boundary checkpoints. Boundary models can extract
arbitrarily long spans **inside one encoded window**, but cannot join a span whose start and end
never co-occur in the same chunk.

**Check whether you need this at all first.** GLiNER2.5 was trained on sequences up to ~4,096
words, so a single memo, email, or short report often already fits in one encoded window with
plain `extract_entities`/`extract`/etc. — no `_long` call, no chunk tuning. Reach for the `_long`
APIs below once a document (a full contract, a multi-hour transcript, a scanned report run
through OCR) exceeds that native window, not by default for anything more than a sentence.

**Fitting in the window is necessary but not sufficient for a combined/repeated-structure
schema.** A single `extract()` call combining `.entities(...)` with a multi-field
`.structure(...)` over a long document with multiple repeated top-level sections (e.g. contract
clauses) can return zero entities and merge every section into one record — no error, just
silently wrong output — even though the document fits in one window and plain entity extraction
alone over the same document works fine. If the document has multiple repeated top-level units,
pre-segment on the obvious markers (section headers, etc.) in plain code and extract per segment,
even when comfortably under `max_len`. See [combined-schemas.md](combined-schemas.md)'s best
practices for the same caveat.

## Why / how

1. Splits the document into overlapping **word** chunks (`chunk_size`, `chunk_overlap`).
2. Runs normal inference on each chunk.
3. Remaps chunk-local character spans back to the original document.
4. Merges duplicate detections from overlapping regions.

**Do not** use `extract_entities(..., max_len=512)` for long documents — it silently drops
everything past the limit.

## Setup

```python
from gliner2 import AutoExtractor, AttributeGroup

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
# Span models work identically: from gliner2 import GLiNER2; GLiNER2.from_pretrained(...)
```

## How chunking works

Chunks are counted in **whitespace word tokens**, not characters or subwords:

```
words:     0        64              384             448
chunk 1:   |------------------------|
chunk 2:            |------------------------------|
           overlap (64 words)
```

- `chunk_size=384` → each window has at most 384 words.
- `chunk_overlap=64` → next window starts 320 words later. Keep `chunk_overlap < chunk_size`.
- Offsets are **global**: `long_text[entity["start"]:entity["end"]] == entity["text"]`.
- Classification is aggregated across chunks (higher confidence wins). Span tasks merge by
  position.
- Chunking uses the model's active word splitter (default `"whitespace"`; use `"char"` for
  languages without whitespace-delimited words, e.g. Chinese or Japanese — set via
  `word_splitter=` on load or `model.set_word_splitter("char")`; see
  [performance-tuning.md](performance-tuning.md) for a worked before/after example).

## Entity extraction

```python
result = model.extract_entities_long(
    long_text, ["company", "person", "product", "location", "date"],
    chunk_size=384, chunk_overlap=64,
    include_spans=True, include_confidence=True,
)
```

Descriptions still help on noisy long text:

```python
entity_types = {
    "contract_party": "Companies or people that are parties to the agreement",
    "effective_date": "Dates when the agreement starts or becomes valid",
    "termination_clause": "Text describing when or how the agreement can end",
}
result = model.extract_entities_long(contract_text, entity_types, chunk_size=512, chunk_overlap=96, include_spans=True)
```

## Choosing chunk settings

| Goal | Starting point |
|---|---|
| General prose | `chunk_size=384`, `chunk_overlap=64` |
| Mentions often sit on boundaries | raise overlap to 96–128 |
| Faster inference | smaller `chunk_size`/`chunk_overlap` |
| Long noun phrases / wide GLiNER2.5 spans | keep the whole phrase inside one chunk |

A 200-word clause needs `chunk_size >= 200`; a clause split across chunk 1 and chunk 2 will be
missed.

## Overlap policy

All local long-document methods accept `overlap_policy` (`None` keeps the architecture default,
which is `"flat"` — resolved via weighted interval scheduling — on all three GLiNER2.5 boundary
checkpoints per their Hub model cards):

| Policy | Behavior |
|---|---|
| `allow` | Keep distinct overlapping spans |
| `nested` | Permit containment |
| `flat` / `disallow` | Remove overlaps deterministically (**default**) |
| `longest` | Drop strictly contained shorter spans |

GLiNER2 deduplicates **overlap artifacts** from adjacent chunks, but keeps distinct mentions at
different document positions even if the surface string repeats.

## Full-schema and task-specific long APIs

```python
result = model.extract_long(long_text, schema, chunk_size=384, chunk_overlap=64, include_spans=True, include_confidence=True)

classification = model.classify_text_long(long_text, {"document_type": ["report", "contract", "email"]})
relations = model.extract_relations_long(long_text, ["works_for", "located_in"], chunk_size=384, chunk_overlap=64, include_spans=True)
structured = model.extract_json_long(long_text, {"invoice": ["vendor::str", "invoice_number::str", "line_item::list"]}, include_spans=True)
```

Matching batch methods: `batch_extract_entities_long`, `batch_extract_long`,
`batch_classify_text_long`, `batch_extract_relations_long`, `batch_extract_json_long`.
`format_results=True` is required for the long batch APIs.

## Span attributes on long documents

`extract_long` runs the same schema per chunk, so `entity_attributes` (see
[span-attributes.md](span-attributes.md)) work unchanged and stay attached to globally remapped
spans.

## Classification on long documents

- **Independent heads** (`classify_text_long`, or schema classification inside `extract_long`):
  aggregates per-chunk scores (prefers higher confidence). Fine for document type/language/topic.
- **Constrained classification**: do **not** stitch independent chunk labels. Use
  `Classifier.classify_long` (see [constrained-classification.md](constrained-classification.md)),
  which aggregates logits then decodes **once** — keeps constraints global.

```python
result = clf.classify_long(long_text, schema, chunk_size=384, chunk_overlap=64, aggregate="max")
```

## Relations and Joint IE on long documents

```python
rels = model.extract_relations_long(long_text, ["works_for", "located_in", "founded"], chunk_size=384, chunk_overlap=96, include_spans=True)
```

A pair is found only if head and tail appear in the **same chunk**. For Joint IE
(see [joint-ie.md](joint-ie.md)), `JointIE.extract_long` never creates cross-chunk edges; increase
`chunk_overlap` when subject and object often sit on either side of a boundary.

## Batch long-document extraction

```python
results = model.batch_extract_entities_long(documents, ["company", "person", "product", "location", "date"],
                                             batch_size=8, chunk_size=384, chunk_overlap=64, include_spans=True)

schemas = [model.create_schema().entities(["company", "date"]), model.create_schema().entities(["person", "location"])]
results = model.batch_extract_long(documents, schemas, chunk_size=384, chunk_overlap=64, include_spans=True)  # per-doc schemas
```

## Limits

| Supported | Not supported |
|---|---|
| Arbitrary span length **inside one chunk** (GLiNER2.5) | A span whose start/end never share a chunk |
| Relations with both arguments in one chunk | Relations crossing chunk boundaries |
| Global `[start, end)` offsets into the original string | Chunk-local offsets treated as document offsets |
| Deduping overlap copies of the same mention | Assuming repeated surface forms collapse document-wide |

If a fact is systematically split across a boundary, increase `chunk_overlap` or pre-segment into
paragraphs/sentences that contain both arguments.

## Best practices

- Prefer `*_long` methods over `max_len` truncation whenever a file might exceed the window.
- Start with `chunk_size=384`, `chunk_overlap=64`; raise overlap for relations and multi-token
  names.
- Request `include_spans=True` until every offset slices back to the expected substring.
- Use label descriptions on long, repetitive documents to cut generic false positives.
- Pair `::str` structure fields with a length-bounding `RegexValidator` on long/repetitive
  documents — see [json-extraction.md](json-extraction.md)'s pitfall section: this contains
  runaway spans but won't manufacture a short answer where none exists.
- On long documents with numbered sections, a bare cross-reference (`"Section 8.4"`) can satisfy
  an entity description just by proximity. Exclude the pattern with
  `RegexValidator(r"^(Section\s+)?\d+(\.\d+)*\.?$", exclude=True)` on an `entities()` config dict
  (not just `.structure().field()`) to drop bare references while leaving real matches intact.
- For constrained labels, aggregate then decode once — never majority-vote per-chunk labels.
- For graphs, use `JointIE.extract_long` and accept that edges are intra-chunk.
- Tune `threshold`, `chunk_size`, `chunk_overlap` on a real domain sample, not just short
  sentences.
