# Relation Extraction

Extract relationships between entities as directional `(head, tail)` tuples — **independent**
decoding, no typed endpoints or graph-level constraints. Mirrors
[tutorial/6-relation_extraction.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/6-relation_extraction.md).

When you need typed endpoints, per-head uniqueness, or a globally consistent graph, use
[joint-ie.md](joint-ie.md) (`JointIE`, GLiNER2.5 + `enable_relations=True`) instead.

## Basic usage

```python
from gliner2 import AutoExtractor

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")

text = "John works for Apple Inc. and lives in San Francisco."
results = extractor.extract_relations(text, ["works_for", "lives_in"])
# {'relation_extraction': {'works_for': [('John', 'Apple Inc.')], 'lives_in': [('John', 'San Francisco')]}}

# Equivalent via schema
schema = extractor.create_schema().relations(["works_for", "lives_in"])
results = extractor.extract(text, schema)
```

**Every requested relation type always appears in the output, even as an empty list `[]`** if
nothing was found — safe to iterate without existence checks:

```python
results = extractor.extract_relations(text, ["manages", "reports_to", "founded"])
# {'relation_extraction': {'manages': [...], 'reports_to': [...], 'founded': []}}
```

Multiple instances per type are all returned automatically — no special config needed.

## Descriptions improve accuracy

```python
schema = extractor.create_schema().relations({
    "works_for": "Employment relationship where person works at organization",
    "founded": "Founding relationship where person created organization",
    "acquired": "Acquisition relationship where company bought another company",
})
```

## Thresholds

```python
# Global
results = extractor.extract_relations(text, ["acquired", "merged_with"], threshold=0.8)

# Per-relation, via config dict
schema = extractor.create_schema().relations({
    "acquired": {"description": "Company acquisition relationship", "threshold": 0.9},
    "partnered_with": {"description": "Partnership or collaboration relationship", "threshold": 0.6},
})
```

## Confidence and spans

```python
results = extractor.extract_relations(text, ["works_for"], include_confidence=True, include_spans=True)
# {'relation_extraction': {'works_for': [{
#     'head': {'text': 'John', 'confidence': 0.95, 'start': 0, 'end': 4},
#     'tail': {'text': 'Apple Inc.', 'confidence': 0.92, 'start': 15, 'end': 25},
# }]}}
```

When both flags are `False` (default), relations are plain `(head, tail)` tuples; when either is
`True`, each relation becomes a `{'head': {...}, 'tail': {...}}` dict.

## Batch processing

```python
results = extractor.batch_extract_relations(
    texts, ["works_for", "founded", "reports_to", "lives_in"], batch_size=8,
)
# list of one result dict per text; every relation type present in every result, even if empty
```

## Combining with other tasks

```python
schema = (
    extractor.create_schema()
    .entities(["person", "organization", "location"])
    .relations(["works_for", "located_in"])
)
```

See [combined-schemas.md](combined-schemas.md) for richer multi-task compositions.

## Best practices

1. **Clear, specific relation names** (`works_for`, `reports_to`) over generic ones (`related`).
2. **Add descriptions** for ambiguous relations, especially when several relation types could
   plausibly apply to the same entity pair.
3. **Set higher thresholds** (0.8–0.9) for high-stakes relations like `acquired`; lower (0.5–0.6)
   for softer/implicit ones like `competes_with`.
4. **Relations are directional** — `works_for` is `(person, organization)`, `reports_to` is
   `(subordinate, manager)`. Verify your relation name's implied direction matches the tuple
   order you're producing/consuming.
5. **Always check for empty lists**, not absence of key, when branching on whether a relation was
   found — the key is always present.
6. Combine with entity extraction so relation arguments have known types for downstream use.
7. **Pronouns referring to a known party (`"reports to me"`) don't resolve to a name and the
   relation is silently dropped** — resolve known pronouns from structured metadata you already
   have (e.g. regex-substitute `"me"` with the sender's name from an email header) before calling
   `extract_relations`; push resolution upstream rather than expecting the model to do it.
8. **Repeated subjects and objects can produce Cartesian all-pairs relations** — e.g. two
   founders and two companies in one document may yield all four pairings. Pre-segment by
   sentence/section when relations are local; use typed [Joint IE](joint-ie.md) when endpoint
   types and graph consistency matter; and validate co-occurrence or domain constraints before
   writing relations to a knowledge base.
