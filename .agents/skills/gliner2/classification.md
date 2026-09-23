# Text Classification

Single- or multi-label classification with configurable confidence. Mirrors
[tutorial/1-classification.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md).

For **hard cross-task constraints** (one label legally implies/forbids another), use
[constrained-classification.md](constrained-classification.md)'s `Classifier` instead —
`classify_text()` / `.classification()` always decode each task independently.

## Setup

```python
from gliner2 import AutoExtractor

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
```

## Single-label

```python
schema = extractor.create_schema().classification("sentiment", ["positive", "negative", "neutral"])
results = extractor.extract("This product exceeded my expectations!", schema)
# {'sentiment': 'positive'}

# With confidence
results = extractor.extract(text, schema, include_confidence=True)
# {'sentiment': {'label': 'neutral', 'confidence': 0.82}}
```

## Multi-label

Use `multi_label=True` with a lower `cls_threshold` (categories are not mutually exclusive).

```python
schema = extractor.create_schema().classification(
    "topics",
    ["technology", "business", "health", "politics", "sports"],
    multi_label=True,
    cls_threshold=0.3,
)
results = extractor.extract(text, schema)
# {'topics': ['technology', 'business', 'health']}
# with include_confidence=True: [{'label': 'technology', 'confidence': 0.92}, ...]
```

## Descriptions improve accuracy

Pass `{label: description}` instead of a bare list — this is the biggest accuracy lever for
classification, especially with ambiguous or domain-specific label names.

```python
schema = extractor.create_schema().classification(
    "document_type",
    {
        "invoice": "A bill for goods or services with payment details",
        "receipt": "Proof of payment for a completed transaction",
        "contract": "Legal agreement between parties with terms and conditions",
        "proposal": "Document outlining suggested plans or services with pricing",
    },
)
```

## Quick API (no schema builder)

```python
# Single task
results = extractor.classify_text(text, {"sentiment": ["positive", "negative", "neutral"]})

# Multiple tasks in one call
results = extractor.classify_text(
    text,
    {
        "sentiment": ["positive", "negative", "neutral"],
        "urgency": ["high", "medium", "low"],
        "category": {"labels": ["tech", "finance", "politics", "sports"], "multi_label": False},
    },
)
# {'sentiment': 'negative', 'urgency': 'high', 'category': 'tech'}

# Multi-label with config dict
results = extractor.classify_text(
    text,
    {"product_aspects": {"labels": ["camera", "battery", "display"], "multi_label": True, "cls_threshold": 0.4}},
)
```

## Multiple classification tasks in one schema

Chain `.classification(...)` calls to run several independent tasks over the same text in one
pass — mix single- and multi-label freely:

```python
schema = (
    extractor.create_schema()
    .classification("primary_topic", ["tech", "business", "health", "sports", "politics"])
    .classification("urgency", ["immediate", "soon", "later", "not_urgent"])
    .classification("emotions", ["happy", "sad", "angry", "surprised"], multi_label=True, cls_threshold=0.4)
    .classification(
        "content_flags",
        ["inappropriate", "spam", "promotional", "personal_info"],
        multi_label=True,
        cls_threshold=0.3,
    )
)
results = extractor.extract(text, schema)
# {'primary_topic': 'business', 'urgency': 'immediate', 'emotions': ['happy', 'fearful'], 'content_flags': ['promotional']}
```

## Advanced configuration

```python
# Per-task threshold
schema = (
    extractor.create_schema()
    .classification("priority", ["urgent", "high", "normal", "low"], cls_threshold=0.8)
    .classification("department", ["sales", "support", "billing", "other"], cls_threshold=0.5)
)

# Force activation function: "sigmoid" | "softmax" | "auto" (default)
schema = extractor.create_schema().classification("category", ["A", "B", "C", "D"], class_act="softmax")
```

## Batch processing

```python
results = extractor.batch_classify_text(
    texts, {"sentiment": ["positive", "negative", "neutral"]}, batch_size=8,
)
# list of one result dict per text
```

For a schema built with `create_schema()`, use `extractor.batch_extract(texts, schema)` instead —
one schema for every text, or a list of per-document schemas the same length as `texts`.

## Best practices

1. **Use descriptions** whenever labels are ambiguous or domain-specific — see above.
2. **Threshold ranges**: 0.5–0.7 for single-label, 0.3–0.5 for multi-label.
3. **Multi-label vs single-label**: use `multi_label=True` only when categories genuinely
   coexist (e.g. product features); use single-label for mutually-exclusive scales (e.g. size).
4. **Test with real domain examples** before shipping thresholds.

## Common use cases

Sentiment analysis, intent classification (chatbot/support routing), document classification
(email filtering), content moderation, topic tagging.
