# Combined / Multi-Task Schemas

Extract multiple task types (entities, classification, structures, relations) in one forward
pass over the same text. Mirrors
[tutorial/4-combined.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/4-combined.md).

Combining schemas: extracts multiple information types in one pass, maintains shared context
across tasks, avoids multiple model calls, and builds comprehensive extraction pipelines.

For **per-span labels** (e.g. sentiment on individual product mentions rather than the whole
document), combine entities with `entity_attributes` — see
[span-attributes.md](span-attributes.md). For **typed relation graphs** with graph-level
constraints, see [joint-ie.md](joint-ie.md) instead of plain `.relations(...)`.

## Basic combinations

```python
from gliner2 import AutoExtractor

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")

# Entities + classification
schema = (
    extractor.create_schema()
    .entities(["person", "product", "company"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("category", ["review", "news", "opinion"])
)
results = extractor.extract("Tim Cook announced Apple's new iPhone is exceeding sales expectations.", schema)
# {'entities': {...}, 'sentiment': 'positive', 'category': 'news'}

# Entities + structure
schema = (
    extractor.create_schema()
    .entities({"person": "Names of people mentioned", "date": "Dates and time references"})
    .structure("appointment")
        .field("patient", dtype="str")
        .field("doctor", dtype="str")
        .field("date")
        .field("type", dtype="str", choices=["checkup", "followup", "consultation"])
)
```

## Full multi-task example

Combine entities, classification, relations, and a structure in a single schema and a single
`extract` call:

```python
schema = (
    extractor.create_schema()
    .entities(["company", "product", "location", "person"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations(["works_for", "located_in"])
    .structure("review")
        .field("product_name", dtype="str")
        .field("rating", dtype="str")
        .field("recommendation", dtype="str")
)
results = extractor.extract(text, schema, include_confidence=True, include_spans=True)
```

`include_confidence` and `include_spans` apply uniformly across **every** task type in a combined
schema — entities, classifications, relations, and structures all pick up the same flags in one
call.

## Real-world composition patterns

**Comprehensive document analysis** (invoice): document-type + payment-status classification,
key entities (company/person/date/amount), an `invoice_header` structure, a repeated `line_item`
structure, and a `payment_info` structure — all declared on one schema and extracted in one pass.

**Customer feedback analysis**: `sentiment` + multi-label `intent` classification, entities for
`product`/`feature`/`competitor`, plus `issue` and `suggestion` structures.

**News article analysis**: `category`/`bias`/`factuality` classification, `person`/
`organization`/`location`/`event` entities, `quote` and `claim` structures.

General shape for any domain:

```python
schema = (
    extractor.create_schema()
    .classification("document_type", [...])       # 1+ classification heads
    .entities({...descriptions...})                # entity types with descriptions
    .structure("some_record")                      # 1+ repeated structures
        .field("f1", dtype="str")
        .field("f2", dtype="list", choices=[...])
    .relations([...])                               # optional independent relations
)
```

## Batch processing

`extractor.batch_extract(texts, schema)` runs a combined schema over many documents in one call:

```python
results = extractor.batch_extract(texts, schema, batch_size=8, include_spans=True)
# list of one result dict per text, same shape as a single extract() call

# Or a different schema per document (list must match len(texts)):
results = extractor.batch_extract(texts, [schema_a, schema_b, ...], batch_size=8)
```

## Best practices

- Reach for `create_schema()` chaining as soon as you need **2 or more** task types over the same
  text — it's strictly more capable than combining separate calls and only costs one forward
  pass.
- Keep entity/classification/relation names and descriptions **consistent across the whole
  schema** (e.g. don't call the same concept `company` in entities and `organization` in
  relations).
- Structures declared with `.structure(name)` automatically capture **every** instance found in
  text, just like `extract_json` — no extra config needed for repeated records.
- For documents longer than the model's window, use `extract_long(text, schema, ...)` — see
  [long-context.md](long-context.md); it accepts any schema built here unchanged.
- **Combining a heavy multi-field `.structure(...)` with `.entities(...)` over a long document
  (thousands of words, even if under `max_len`) can silently starve or corrupt one of the tasks**
  — failure mode: entities come back as empty lists (no error) while a repeated structure that
  should produce one record per section instead collapses every section's fields into a single
  merged record. Nothing errors or warns; the only symptom is output counts lower than expected.
  If a long document has many repeated top-level sections (contract clauses, chat turns, ticket
  entries), don't rely on the model to find and anchor each instance in one combined pass —
  pre-segment the document on the obvious structural markers (headers, delimiters) in plain code
  first, then run one `extract()`/`extract_json()` call per segment. Reserve one-call combined
  schemas for short-to-medium text or for documents whose repeated units aren't already
  unambiguous in plain Python.
