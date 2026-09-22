# Training Data Format (JSONL)

JSONL format for training GLiNER2/GLiNER2.5 models: each line has `input`/`output` (or
equivalently `text`/`schema`). Covers **entities, classifications, json structures, and
relations**; the same format works for span and boundary models. Span-attribute labels are
inferred at inference from entity spans — there is no separate attribute field in JSONL yet.
Mirrors [tutorial/8-train_data.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md).

## Format

```jsonl
{"input": "text to process", "output": {"schema_definition": "with_annotations"}}
```
or equivalently `{"text": "...", "schema": {...}}` — both accepted.

| Output key | Type | Description |
|---|---|---|
| `entities` | `dict[str, list[str]]` | Entity type → mentions |
| `entity_descriptions` | `dict[str, str]` | Entity type → description |
| `classifications` | `list[dict]` | Classification tasks |
| `json_structures` | `list[dict]` | Structured extractions |
| `json_descriptions` | `dict[str, dict[str, str]]` | Parent → field → description |
| `relations` | `list[dict]` | Relation extractions |

**Every example must have at least one non-empty task.** A fully empty output fails validation.

## Entities

```jsonl
{"input": "John Smith works at OpenAI in San Francisco.", "output": {"entities": {"person": ["John Smith"], "organization": ["OpenAI"], "location": ["San Francisco"]}}}
```

With descriptions:

```jsonl
{"input": "Dr. Sarah Johnson prescribed Metformin 500mg twice daily.", "output": {"entities": {"person": ["Dr. Sarah Johnson"], "medication": ["Metformin"], "dosage": ["500mg"]}, "entity_descriptions": {"person": "Names of people mentioned", "medication": "Names of drugs"}}}
```

## Classifications

Required fields: `task`, `labels`, `true_label`. Optional: `multi_label`, `prompt`, `examples`,
`label_descriptions`.

```jsonl
{"input": "This movie is fantastic!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}]}}
```

- `true_label` accepts either a string (`"positive"`) or a list (`["positive"]`) for single-label
  — internally normalized to a list. **Multi-label must always use a list.**
- `multi_label: true` for several simultaneous labels:
  ```jsonl
  {"input": "Amazing camera but poor battery.", "output": {"classifications": [{"task": "aspects", "labels": ["camera", "battery", "screen"], "true_label": ["camera", "battery"], "multi_label": true}]}}
  ```
- `examples` (few-shot): list of `[input_text, output_label]` pairs — each exactly 2 elements.
  ```jsonl
  {"input": "This service exceeded expectations!", "output": {"classifications": [{"task": "sentiment", "labels": ["positive", "negative"], "true_label": ["positive"], "examples": [["Great product!", "positive"], ["Terrible experience.", "negative"]]}]}}
  ```
- Multiple tasks in one example: include multiple dicts in the `classifications` list.

## JSON structures

Parent name is the dict key; fields are `field_name: value` (`str`, `list[str]`, or a choice
dict).

```jsonl
{"input": "Contact John Doe at john.doe@email.com or call (555) 123-4567.", "output": {"json_structures": [{"contact": {"name": "John Doe", "email": "john.doe@email.com", "phone": "(555) 123-4567"}}]}}
```

**Multiple instances of the same parent** are separate dicts in the list (union of fields across
instances is used — unlike relations, fields may vary between instances):

```jsonl
{"input": "Hotel Paradise: 4 stars, pool, wifi, $150/night. Budget Inn: 2 stars, parking, $80/night.", "output": {"json_structures": [{"hotel": {"name": "Hotel Paradise", "stars": "4", "amenities": ["pool", "wifi"], "price": "$150/night"}}, {"hotel": {"name": "Budget Inn", "stars": "2", "amenities": ["parking"], "price": "$80/night"}}]}}
```

Choice fields (classification-style, within a structure):

```jsonl
{"input": "Book a single room for 2 nights with breakfast.", "output": {"json_structures": [{"booking": {"room_type": {"value": "single", "choices": ["single", "double", "suite"]}, "nights": "2", "meal_plan": {"value": "breakfast", "choices": ["none", "breakfast", "full-board"]}}}]}}
```

`json_descriptions`: `{"parent": {"field": "description"}}`.

## Relations

Flexible field names (not limited to `head`/`tail`) — but **the first occurrence of a relation
type fixes the field structure for every subsequent instance of that type**, enforced during
validation (`TrainingDataset.validate_relation_consistency()`).

```jsonl
{"input": "Alice works for Google. Bob works for Microsoft.", "output": {"relations": [{"works_for": {"head": "Alice", "tail": "Google"}}, {"works_for": {"head": "Bob", "tail": "Microsoft"}}]}}
```

Custom field names, still consistent per type:

```jsonl
{"input": "Alice sent $100 to Bob. Charlie sent $50 to David.", "output": {"relations": [{"transaction": {"sender": "Alice", "recipient": "Bob", "amount": "$100"}}, {"transaction": {"sender": "Charlie", "recipient": "David", "amount": "$50"}}]}}
```

Different relation types may each define their own field structure independently. This strict
per-type consistency is the key difference from `json_structures`, which allows the field set to
vary freely between instances of the same parent.

## Multi-task combined examples

Any combination of `entities` / `classifications` / `json_structures` / `relations` may appear in
one example — this is how multi-task models are trained:

```jsonl
{"input": "Breaking: Apple announces new iPhone 15. Analysts are optimistic.", "output": {"entities": {"company": ["Apple"], "product": ["iPhone 15"]}, "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"], "true_label": ["positive"]}], "json_structures": [{"news_article": {"company": "Apple", "product": "iPhone 15"}}], "relations": [{"product_of": {"head": "iPhone 15", "tail": "Apple"}}]}}
```

An example with only an empty `entities: {}` but a non-empty `classifications` list is valid
(has ≥1 task); an example with **all** tasks empty is not.

## Alternative input formats

The loader auto-detects and accepts:

1. JSONL file(s): `{"input": ..., "output": {...}}` or `{"text": ..., "schema": {...}}`
2. Python API: `InputExample` / `TrainingDataset` from `gliner2.training.data`
3. Raw dict lists in the same shape as JSONL

## Edge cases worth knowing

- Unicode/non-ASCII, quotes, newlines, special characters (`$1,299.99`, `C++`, `@`) are all
  supported directly in JSON strings.
- Numbers as entity/field values are plain strings (`"123"`, not `123`).
- Empty string values (`""`) are valid for a missing field on a structure.

## Validation

```python
from gliner2.training.data import TrainingDataset

dataset = TrainingDataset.load("train.jsonl")
dataset.validate(raise_on_error=True)   # checks format, required fields, AND that entity/relation values exist in the input text
dataset.validate_relation_consistency()  # per-relation-type field consistency
dataset.print_stats()
```

**`validate()` takes no `strict` argument — passing `strict=True` raises `TypeError`.** As of
`gliner2==2.0.0`, the signature is `validate(self, raise_on_error=True)` only. The substring/
text-exists check is not a separate opt-in mode — `validate()` always checks that entity mentions
and relation values exist in the input text (case-insensitive substring match) as part of its one
pass. `raise_on_error=False` returns a report dict (`{'valid', 'invalid', 'total',
'invalid_indices', 'errors'}`) instead of raising, which is what you want during interactive
dataset cleanup.

## Tips

1. Use diverse, realistic domain examples; balance classification classes.
2. Provide descriptions wherever available — same accuracy lever as at inference time.
3. Include multiple `json_structures` instances to teach multi-record extraction.
4. Mix task types per example so the model learns multi-task composition.
5. Validate with `dataset.validate(raise_on_error=True)` before training, not after.
