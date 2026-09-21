# Entity Extraction (NER)

Extract named entities with optional descriptions for precision. Mirrors
[tutorial/2-ner.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/2-ner.md).

## Setup

```python
from gliner2 import AutoExtractor

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
```

## Basic extraction

```python
text = "Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023."
results = extractor.extract_entities(text, ["company", "person", "product", "location", "date"])
# {'entities': {'company': ['Apple Inc.'], 'person': ['Tim Cook'], 'product': ['iPhone 15'],
#               'location': ['Cupertino', 'California'], 'date': ['September 12, 2023']}}

# Equivalent via schema builder
schema = extractor.create_schema().entities(["company", "person", "product", "location", "date"])
results = extractor.extract(text, schema)
```

## Descriptions improve accuracy

```python
schema = extractor.create_schema().entities({
    "drug": "Pharmaceutical drugs, medications, or treatment names",
    "disease": "Medical conditions, illnesses, or disorders",
    "symptom": "Clinical symptoms or patient-reported symptoms",
    "dosage": "Medication amounts like '50mg' or '2 tablets daily'",
})
```

## Single vs multiple per type

```python
# Default: extract every match
schema = extractor.create_schema().entities(["person", "organization"], dtype="list")

# Only the single best match per type
schema = extractor.create_schema().entities(["company", "ceo"], dtype="str")
```

## Confidence and spans

```python
results = extractor.extract_entities(text, ["company", "person"], include_confidence=True)
# {'entities': {'company': [{'text': 'Apple Inc.', 'confidence': 0.95}], ...}}

results = extractor.extract_entities(text, ["company", "person"], include_spans=True)
# {'entities': {'company': [{'text': 'Apple Inc.', 'start': 0, 'end': 9}], ...}}
```

| `include_confidence` | `include_spans` | Shape |
|---|---|---|
| `False` | `False` | plain string list: `['Apple Inc.', 'Tim Cook']` |
| `True` | `False` | `{'text', 'confidence'}` |
| `False` | `True` | `{'text', 'start', 'end'}` |
| `True` | `True` | `{'text', 'confidence', 'start', 'end'}` |

## Thresholds

```python
# Global
results = extractor.extract_entities(text, ["email", "phone", "address"], threshold=0.8)

# Per-entity, full config dict form
schema = extractor.create_schema().entities({
    "email": {"description": "Email addresses", "dtype": "list", "threshold": 0.9},
    "phone": {"description": "Phone numbers", "dtype": "list", "threshold": 0.7},
    "name":  {"description": "Person names", "dtype": "list", "threshold": 0.5},
})
```

## Incremental / mixed schema construction

`.entities(...)` can be called repeatedly and accepts a list, a `{type: description}` dict, or a
full `{type: {"description", "dtype", "threshold"}}` config dict — mix freely:

```python
schema = extractor.create_schema()
schema.entities(["date", "time", "currency"])
schema.entities({"technical_term": "Technical jargon or specialized terminology"})
schema.entities({"competitor": {"description": "Competing companies or products", "dtype": "list", "threshold": 0.7}})
```

## Batch processing

```python
results = extractor.batch_extract_entities(
    texts, ["company", "person", "product", "location"],
    batch_size=8, include_spans=True,
)
# list of one result dict per text
```

Also accepts a schema built with `create_schema()` via `extractor.batch_extract(texts, schema)` —
pass a single schema (applied to every text) or a list of schemas the same length as `texts` (one
schema per document).

## Domain patterns

```python
legal_schema = extractor.create_schema().entities({
    "party": "Parties involved in legal proceedings (plaintiff, defendant, etc.)",
    "law_firm": "Law firm or legal practice names",
    "court": "Court names or judicial bodies",
    "statute": "Legal statutes, laws, or regulations cited",
})

finance_schema = extractor.create_schema().entities({
    "ticker": "Stock ticker symbols (e.g., AAPL, GOOGL)",
    "financial_metric": "Financial metrics like P/E ratio, market cap",
    "currency_amount": "Monetary values with currency symbols",
})
```

## Best practices

1. **Use specific, descriptive entity type names** (`drug_name`, not `thing`).
2. **Add descriptions** for anything domain-specific or ambiguous — they materially raise
   precision/recall, especially in medical, legal, and financial text.
3. **Choose `dtype="str"`** when you only want the single best mention per type (e.g. "the CEO"),
   `dtype="list"` (default) when multiple mentions matter.
4. For documents longer than the model's context window, use `extract_entities_long` instead —
   see [long-context.md](long-context.md).
5. GLiNER2.5 boundary checkpoints have **no maximum span width** — a full postal address, a
   clause-length legal reference, or a multi-line table cell can be extracted as one entity.
   Legacy span checkpoints cap width at `max_width` (12 words by default); see
   [long-context.md](long-context.md)'s limits table for what still requires chunking either way.
6. **Watch for the defined-term literal-match trap in contracts/legal text.** Documents that
   define a term once ("'Insurer' means Continental Assurance Underwriters...") and then reuse
   the capitalized placeholder throughout will tempt the model into extracting the placeholder
   word itself ("Insurer", repeated dozens of times) instead of resolving it to the actual proper
   name it stands for — because the description you wrote ("companies that are parties to the
   agreement") matches the placeholder's surface usage just as well as the real name. Ask
   explicitly for the proper name (e.g. `"insurer_name": "The proper name of the insurance
   company issuing the policy, not the word 'Insurer' itself"`) and consider extracting each
   defined role (`insurer_name`, `insured_name`, ...) as its own entity type rather than one
   generic `party` type when the document has this pattern.
