# Span Attributes (GLiNER2.5 only)

Attach labels such as **sentiment** to extracted entity *spans* — not the whole document. The
model first finds entities, then scores attribute labels at those exact spans, so a review can be
mixed overall while individual products are positive or negative. Mirrors
[tutorial/13-span_attributes.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/13-span_attributes.md).

Requires a **boundary** (GLiNER2.5) checkpoint loaded via `AutoExtractor`.

## Setup

```python
from gliner2 import AutoExtractor, AttributeGroup

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
```

`entity_attributes()` must be called **after** `entities()`. Attribute group names cannot be
`text`, `confidence`, `start`, or `end` (reserved output keys).

## How it works

1. Declare content entity types (`product`, `company`, …) via `.entities(...)`.
2. Declare one or more `AttributeGroup`s via `.entity_attributes({...})`.
3. Attribute labels are internal queries — they are **not** returned as extra entity types.
4. After entity decoding, each retained span is force-scored for every applicable attribute
   group, and the chosen label(s) are attached to that entity object.

Single-label groups use softmax (one value); multi-label groups use sigmoid + `threshold`.

## Sentiment on entity mentions

```python
schema = (
    model.create_schema()
    .entities(["product"])
    .entity_attributes({
        "sentiment": AttributeGroup(
            ["positive", "negative", "neutral"],
            applies_to=["product"],
            qualify_labels=True,
        )
    })
)

result = model.extract(
    "The new iPhone camera is excellent, but the battery life is disappointing.",
    schema, include_spans=True, include_confidence=True,
)
# {'entities': {'product': [
#     {'text': 'iPhone camera', 'start': 8, 'end': 21, 'confidence': 0.91,
#      'sentiment': {'label': 'positive', 'confidence': 0.87}},
#     {'text': 'battery life', 'start': 35, 'end': 47, 'confidence': 0.88,
#      'sentiment': {'label': 'negative', 'confidence': 0.84}},
# ]}}
```

Always request `include_spans=True` while developing and assert
`text[entity["start"]:entity["end"]] == entity["text"]`.

## Restricting to specific entity types with `applies_to`

Only entity types listed in `applies_to` receive the attribute group; other entities in the same
schema are extracted without it. If `applies_to` is omitted, the group scores on **every**
declared entity type. An unknown name in `applies_to` raises `ValueError`.

## Single-label vs multi-label groups

```python
# Single-label (default) — one mutually exclusive value via softmax
AttributeGroup(["positive", "negative", "neutral"], multi_label=False)

# Multi-label — independent sigmoid decisions, several labels can fire on one span
AttributeGroup(["price", "quality", "design", "battery", "support"], multi_label=True, threshold=0.4, applies_to=["product"])
# result: "aspects": [{"label": "price", "confidence": 0.71}, {"label": "quality", "confidence": 0.82}, ...]
```

Lower `threshold` to recall more aspects; raise it to reduce noise.

## Multiple attribute groups + `qualify_labels`

A span can carry several independent groups. Labels must be unique across groups; if a label
string would collide with an entity type or another group's labels, set `qualify_labels=True`
(prefixes the model-facing label with the group name, e.g. `sentiment: positive`, while the result
still reports the short label `positive`).

```python
schema = (
    model.create_schema()
    .entities({"product": "Consumer devices or software products", "company": "Company or brand names"})
    .entity_attributes({
        "sentiment": AttributeGroup(["positive", "negative", "neutral"], applies_to=["product"], qualify_labels=True),
        "urgency": AttributeGroup(["low", "medium", "high"], applies_to=["product"], qualify_labels=True),
    })
)
```

Without `qualify_labels=True`, a colliding label raises:
`Attribute labels collide with entity labels: ...; use qualify_labels=True`.

## Combining with other schema tasks

Attributes compose with classification, relations, and structures in one `extract()` call —
document-level classification stays independent of per-span attributes:

```python
schema = (
    model.create_schema()
    .entities(["product", "company"])
    .entity_attributes({"sentiment": AttributeGroup(["positive", "negative", "neutral"], applies_to=["product"], qualify_labels=True)})
    .classification("review_type", ["unboxing", "complaint", "comparison", "praise"])
)
```

For documents longer than the model window, use `extract_long` with the same schema — see
[long-context.md](long-context.md).

## Reading the output

| `include_spans` | `include_confidence` | Entity object |
|---|---|---|
| `True` | `True` | `{text, start, end, confidence, sentiment: {label, confidence}}` |
| `True` | `False` | `{text, start, end, sentiment: ...}` |
| `False` | `True` | `{text, confidence, sentiment: ...}` |

```python
entity["sentiment"]["label"]                       # single-label value
[item["label"] for item in entity["aspects"]]      # multi-label values
```

Attributes are only attached to **retained** entities — a mention dropped by threshold or overlap
policy is not attributed.

## Use case: clinical extraction

Attach negation status and dosage form to each symptom/medication span in the same forward
pass, instead of a second classification pass over every extracted span:

```python
schema = (
    model.create_schema()
    .entities({"symptom": "Clinical symptoms or patient-reported complaints", "medication": "Drug or treatment names"})
    .entity_attributes({
        "negation": AttributeGroup(["present", "negated"], applies_to=["symptom"], qualify_labels=True),
        "dosage_form": AttributeGroup(["oral", "topical", "injectable", "unspecified"], applies_to=["medication"], qualify_labels=True),
    })
)
result = model.extract("No fever, but persistent cough. Prescribed oral Amoxicillin.", schema)
# symptom "fever" -> negation: negated; symptom "cough" -> negation: present;
# medication "Amoxicillin" -> dosage_form: oral
```

## Best practices

- Call `entities()` before `entity_attributes()`.
- **Missed entities and wrong attributes are two different failure modes — diagnose which one
  you have before tuning.** If an expected instance is missing entirely from the output, that's
  an entity-*recall* problem (the span was never found), not an attribute-scoring problem — no
  amount of `AttributeGroup`/threshold tuning fixes it. Fix recall first with the same lever
  `SKILL.md`'s cross-cutting mechanics recommend everywhere else: a more specific/example-bearing
  description on `.entities(...)` (e.g. add the missed instance's own wording as an example in
  the description). Only once every expected span is actually retained does it make sense to
  debug or tune the attribute group scoring on top of it.
- Use `applies_to` so unrelated entity types (companies, dates, locations) don't get spurious
  product-style attributes.
- Prefer `qualify_labels=True` in any multi-task schema.
- Keep single-label attribute vocabularies short and mutually exclusive.
- Verify offsets with `include_spans=True` until confirmed correct.
- Never reuse reserved names (`text`, `start`, `end`, `confidence`) as a group name.
- For whole-document sentiment, use classification instead of (or alongside) span attributes.
- **Worked examples in this file (and others) show one illustrative run, not a reproducible
  contract.** Re-running the exact sentiment example above on a live checkpoint returned only
  one of the two documented entities, at different confidence values. Exact spans/confidences
  are checkpoint-version- and run-dependent zero-shot output — verify shape (which keys appear,
  under what nesting) against these examples, but don't treat the specific entities/numbers as a
  regression target.
