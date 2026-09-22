# Structured Data / JSON Extraction

Parse complex structured information (records) with field-level type/choice control. Mirrors
[tutorial/3-json_extraction.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/3-json_extraction.md).

On **GLiNER2.5 boundary** checkpoints, prefer record mode
(`structure(..., mode="natural", anchor=...)`) when repeated instances must keep field identity —
see the [boundary architecture guide](https://github.com/fastino-ai/GLiNER2/blob/main/docs/boundary_baseline.md).

## Quick API: `extract_json()`

For structure-only extraction, use the simple dict format directly — no schema builder needed.

```python
from gliner2 import AutoExtractor

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")

text = "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage."
results = extractor.extract_json(text, {"product": ["name::str", "price", "features"]})
# {'product': [{'name': 'MacBook Pro', 'price': ['$1999'], 'features': ['M3 chip', '16GB RAM', '512GB storage']}]}
```

## Field specification syntax

Fields use `::`-delimited specs:

```
"field_name"                                    # simple, defaults to list type
"field_name::description"                       # defaults to list type
"field_name::type"                              # "field::str" or "field::list"
"field_name::type::description"
"field_name::[choice1|choice2|choice3]::type::description"   # classification-style field
```

```python
results = extractor.extract_json(
    text,
    {"event": [
        "name::str::Event or conference name",
        "date::str::Event date",
        "location::str",
        "topics::list::Conference topics",
        "registration_fee::str",
    ]},
)
```

### Choice fields

```python
results = extractor.extract_json(
    text,
    {"reservation": [
        "restaurant::str::Restaurant name",
        "party_size::[1|2|3|4|5|6+]::str::Number of guests",
        "seating::[indoor|outdoor|bar]::str::Seating preference",
        "dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions",
    ]},
)
```

**Choice fields do not vary per-instance inside a repeated structure.** A `[choice]`/`choices=`
field nested in a structure that matches multiple times in one document (e.g. `sentiment` on a
`dish` structure matched once per dish in a review) is scored **once at the document level** and
copied into every matched instance — e.g. two dishes with opposite sentiment in the same review
can come back with byte-identical labels *and* identical confidence floats. This holds for
both the `extract_json()` `::[choice]` shorthand above and the schema-builder `.field(...,
choices=[...])` form in the next section — same underlying bug, identical output. If you need a
categorical label that actually varies per matched instance (sentiment per product, status per
line item, etc.), use `entity_attributes()`/`AttributeGroup` instead — see
[span-attributes.md](span-attributes.md); it is the only mechanism that re-scores per span rather
than per document. Plain `str`/`list` fields with no `choices` are unaffected and do vary
correctly per instance (see the `merchant`/`amount` fields in the transactions example below).

## Multiple instances

GLiNER2 automatically extracts **all** instances of a structure found in the text — no special
syntax needed, just a natural-language document with repeated occurrences:

```python
text = """
Recent transactions:
- Jan 5: Starbucks $5.50 (food)
- Jan 5: Uber $23.00 (transport)
- Jan 6: Amazon $156.99 (shopping)
"""
results = extractor.extract_json(
    text,
    {"transaction": ["date::str", "merchant::str", "amount::str", "category::[food|transport|shopping]::str"]},
)
# {'transaction': [{'date': 'Jan 5', 'merchant': 'Starbucks', ...}, {'date': 'Jan 5', 'merchant': 'Uber', ...}, ...]}
```

## Record mode — explicit instance identity (`mode="natural"`, boundary checkpoints)

The `extract_json()`/`::`-spec form above already pairs fields correctly for most documents. When
building through the schema builder on a **boundary (GLiNER2.5)** checkpoint trained with
`enable_records=True` (all three `fastino/gliner2.5-*-v1` checkpoints), you can be explicit about
which field anchors each instance — required when two instances could otherwise be ambiguous
about which values belong together:

```python
schema = (
    extractor.create_schema()
    .structure("purchase", mode="natural", anchor="buyer")
    .field("buyer", dtype="str", cardinality="required_one")
    .field("item", dtype="str", cardinality="required_one")
)
result = extractor.extract("Alice bought apples and Bob bought oranges.", schema)
# {'purchase': [{'buyer': 'Alice', 'item': 'apples'}, {'buyer': 'Bob', 'item': 'oranges'}]}
```

`anchor` names the field whose each new occurrence starts a new record instance; other fields'
values attach to the nearest anchor rather than flattening into unrelated per-field lists (the
failure mode this guards against: `{'buyer': ['Alice', 'Bob'], 'item': ['apples', 'oranges']}`
with the pairing lost). If omitted, `anchor` defaults to the first declared `.field(...)` — see
[training.md](training.md)'s note that ordinary cases need no extra config. `cardinality`
(`"required_one"` here) enforces exactly one value of that field per instance; leave it unset for
the default list-per-field behavior. Legacy span checkpoints don't have `enable_records=True` and
don't support this builder form — use the plain `extract_json()`/`::`-spec form on those instead.

**A record instance with no real match for a field can silently inherit a nearby wrong value
instead of returning null.** E.g. an item mentioned but never actually purchased/ordered can end
up with another instance's price copied in, at a plausible-looking confidence (0.6–0.8) that
survives normal thresholding — this is a pairing error, not low-confidence noise, so raising
`threshold` will not reliably catch it. Sanity-check anchor-paired fields against the source text
directly (not just checking confidence) whenever an instance might legitimately be missing a
field.

## Schema builder — only for multi-task scenarios

Reach for `create_schema().structure(...)` instead of `extract_json` when you also need entities
or classification alongside structures in the same pass:

```python
schema = (
    extractor.create_schema()
    .entities(["person", "company", "location"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
        .field("category", dtype="str", choices=["electronics", "software", "service"])
)
results = extractor.extract(text, schema)
```

Per-field threshold and description via the builder:

```python
schema = (
    extractor.create_schema()
    .structure("support_ticket")
        .field("ticket_id", dtype="str", threshold=0.9)  # high precision
        .field("customer", dtype="str", description="Customer name")
        .field("priority", dtype="str", choices=["low", "medium", "high", "urgent"])
        .field("tags", dtype="list", choices=["bug", "feature", "support", "billing"])
)
```

## Confidence and spans

Behaves like entities: string fields become `{'text', 'confidence', 'start', 'end'}` dicts, list
fields become lists of such dicts, when either flag is `True`.

```python
results = extractor.extract_json(
    text,
    {"product": ["name::str", "price", "features"]},
    include_confidence=True,
    include_spans=True,
)
# {'product': [{
#     'name': {'text': 'MacBook Pro', 'confidence': 0.95, 'start': 4, 'end': 15},
#     'price': [{'text': '$1999', 'confidence': 0.92, 'start': 22, 'end': 27}],
#     'features': [{'text': 'M3 chip', 'confidence': 0.88, 'start': 32, 'end': 39}, ...],
# }]}
```

## Batch processing

```python
results = extractor.batch_extract_json(
    texts, {"product": ["name::str", "price", "features"]}, batch_size=8,
)
# list of one result dict per text
```

For the schema-builder form, `extractor.batch_extract(texts, schema)` also works — a single
schema for every text, or a list of per-document schemas the same length as `texts`.

## Pitfall: long or repetitive documents can balloon `::str` fields

GLiNER2.5 boundary checkpoints have no maximum span width (entity-extraction.md's best practice
#5) — on a long, repetitive document a `::str` field's description can end up satisfied by a
multi-hundred-word span instead of a short value, e.g. an unbounded `resolution::str` field
capturing several full turns of dialogue instead of a short phrase. Pairing the field with a
length-bounding [`RegexValidator`](regex-validators.md) (`r"^.{1,80}$"`) contains the damage
**only when a shorter valid candidate span exists at all** — if no candidate under the length cap
exists, the validator leaves the field `null` instead of blob-filled. Treat a validator-filtered
`null` on a long/repetitive document as a real miss to fix with a tighter description or
fine-tuning — not as "solved" just because the blob is gone.

```python
short_value = RegexValidator(r"^.{1,80}$")
schema = (
    extractor.create_schema()
    .structure("complaint")
        .field("order_id", dtype="str", validators=[short_value])
        .field("resolution", dtype="str", validators=[short_value])
)
```

See [long-context.md](long-context.md) for chunking `extract_long` over the full document.

## Best practices

- `::str` for single values (IDs, names, amounts). `::list` (or omit) for multiple values
  (features, items, tags).
- Use `[opt1|opt2|opt3]` choice syntax for standardized/enum-like values instead of free text.
- Add descriptions for complex or domain-specific fields.
- **`extract_json()`**: structure-only, single task, quick parsing.
- **`create_schema().structure(...).extract()`**: multi-task (entities + structures +
  classification), complex pipelines, per-field thresholds/validators.
- For a categorical/choice label that must vary correctly per matched instance in a repeated
  structure, use `entity_attributes()` ([span-attributes.md](span-attributes.md)), not a
  `choices=` field on `structure()` — the latter is scored once per document, not once per
  instance.
