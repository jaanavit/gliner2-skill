# GLiNER (v1): High-Level Pipelines, spaCy, and Recipes

GLiNER-Multitask checkpoints (e.g. `knowledgator/gliner-multitask-large-v0.5`) extend NER to
classification, QA, relation extraction, open IE, and summarization through thin wrapper classes in
`gliner.multitask`. Condensed from
[Advanced Usage](https://urchade.github.io/GLiNER/usage.html#high-level-pipelines-pipelines). If
your task is genuinely multi-task (multiple structured outputs from one pass), prefer GLiNER2's
`create_schema()` chaining instead — see [combined-schemas.md](combined-schemas.md); these
pipelines are useful mainly when you're already committed to a GLiNER v1 multitask checkpoint.

## Classification

**Prefer GLiNER2's `classify_text()` for new work** — see [classification.md](classification.md).
It's a single call with no wrapper class, supports multi-label + descriptions natively, and this
`GLiNERClassifier` wrapper exists only to retrofit classification onto a v1 NER checkpoint.

```python
from gliner import GLiNER
from gliner.multitask import GLiNERClassifier

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
classifier = GLiNERClassifier(model=model)

labels = ['science', 'technology', 'business', 'sports']
classifier("SpaceX successfully launched a new rocket into orbit.", classes=labels, multi_label=False)
# [[{'label': 'technology', 'score': 0.84}]]
classifier(text, classes=labels, multi_label=True)
# [[{'label': 'technology', 'score': 0.84}, {'label': 'science', 'score': 0.72}]]

classifier.evaluate('dair-ai/emotion')  # {'micro': 0.4465, 'macro': 0.4243, 'weighted': 0.4884}
```

## Question-Answering

```python
from gliner.multitask import GLiNERQuestionAnswerer, GLiNERSquadEvaluator

answerer = GLiNERQuestionAnswerer(model=model)
answerer(text, questions="Who founded SpaceX?")
# [[{'answer': 'Elon Musk', 'score': 0.998}]]
answerer(text, questions=["Who founded SpaceX?", "When was SpaceX founded?"])  # batched questions

GLiNERSquadEvaluator(model_id="knowledgator/gliner-multitask-large-v0.5").evaluate(threshold=0.25)
# {'exact': 29.41, 'f1': 29.80, 'total': 11873, ...}
```

## Relation Extraction

**Prefer GLiNER2 for new work**: [relation-extraction.md](relation-extraction.md)'s
`extract_relations()` for independent tuples, or [joint-ie.md](joint-ie.md)'s `JointIE` for typed,
graph-consistent extraction — both are native to the model, not a wrapper bolted onto NER output
the way `GLiNERRelationExtractor` is here.

```python
from gliner.multitask import GLiNERRelationExtractor

relation_extractor = GLiNERRelationExtractor(model=model)
predictions = relation_extractor(
    text, entities=['person', 'company', 'year', 'goal'],
    relations=['founded', 'founded_in', 'goal'], threshold=0.5,
)
for pred in predictions[0]:
    print(f"{pred['source']} --[{pred['relation']}]--> {pred['target']}  ({pred['score']:.3f})")
```

## Open Information Extraction

Prompt-driven extraction — describe what to pull out in natural language instead of a fixed label:

```python
from gliner.multitask import GLiNEROpenExtractor

extractor = GLiNEROpenExtractor(model=model, prompt="Extract all companies related to space technologies")
extractor(text, labels=['company'], threshold=0.5)
# [[{'text': 'SpaceX', 'score': 0.962}, {'text': 'Tesla', 'score': 0.936}, ...]]
```

Other prompt examples: `"Extract product descriptions and features"`,
`"Extract technical specifications and requirements"`,
`"Extract all contact information including emails and phone numbers"`.

## Summarization

```python
from gliner.multitask import GLiNERSummarizer

summarizer = GLiNERSummarizer(model=model)
summary = summarizer(text, threshold=0.1)   # list of extracted key sentences
# Higher threshold (0.5) -> shorter/more selective; lower (0.05) -> longer/more comprehensive
```

## Advanced relation extraction with UTCA

Same preference as above: [joint-ie.md](joint-ie.md)'s `JointIE` gets you typed endpoints,
per-head uniqueness, and graph constraints natively on GLiNER2.5 — reach for UTCA only if you're
already committed to a v1 checkpoint and need `pairs_filter`/`distance_threshold`-style filtering
that GLiNER2 doesn't expose the same way.

For distance-filtered, multi-relation-schema extraction beyond `GLiNERRelationExtractor`:

```bash
pip install utca -U
```

```python
from utca.core import RenameAttribute
from utca.implementation.predictors import GLiNERPredictor, GLiNERPredictorConfig
from utca.implementation.tasks import (
    GLiNER, GLiNERPreprocessor, GLiNERRelationExtraction, GLiNERRelationExtractionPreprocessor,
)

predictor = GLiNERPredictor(GLiNERPredictorConfig(
    model_name="knowledgator/gliner-multitask-large-v0.5", device="cuda:0",
))

pipe = (
    GLiNER(predictor=predictor, preprocess=GLiNERPreprocessor(threshold=0.7))
    | RenameAttribute("output", "entities")
    | GLiNERRelationExtraction(
        predictor=predictor,
        preprocess=GLiNERPreprocessor(threshold=0.5) | GLiNERRelationExtractionPreprocessor(),
    )
)

result = pipe.run({
    "text": text,
    "labels": ["organization", "person", "position", "date"],
    "relations": [
        {"relation": "founder", "pairs_filter": [("organization", "person")], "distance_threshold": 100},
        {"relation": "inception_date", "pairs_filter": [("organization", "date")]},
        {"relation": "held_position", "pairs_filter": [("person", "position")]},
    ],
})
for relation in result["output"]:
    print(f"{relation['source']['span']} --[{relation['relation']}]--> {relation['target']['span']} ({relation['score']:.3f})")
```

`pairs_filter` restricts which entity-type pairs are considered per relation; `distance_threshold`
caps head-tail character distance — both narrow the search space and reduce false positives versus
`GLiNERRelationExtractor`'s unfiltered pairing.

## spaCy integration

```bash
pip install gliner-spacy
```

```python
import spacy
from gliner_spacy.pipeline import GlinerSpacy

nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config={
    "gliner_model": "urchade/gliner_mediumv2.1",
    "chunk_size": 250,
    "labels": ["person", "organization", "email"],
    "style": "ent",
    "threshold": 0.3,
    "map_location": "cpu",   # v0.0.7+
})
doc = nlp("This is a text about Bill Gates and Microsoft.")
for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score)   # ent._.score in v0.0.7+
```

## Recipes (condensed from "Practical Examples")

**PII redaction** (`urchade/gliner_multi_pii-v1`, 40+ types, 100+ languages):

```python
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
entities = model.predict_entities(text, ["person", "date of birth", "social security number", "email", ...], threshold=0.5)
redacted = text
for e in sorted(entities, key=lambda e: e["start"], reverse=True):
    redacted = redacted[:e["start"]] + f"[{e['label'].upper()}]" + redacted[e["end"]:]
```

**Knowledge graph construction** — a relex checkpoint jointly extracts entities and relations in
one pass; see [gliner1-usage.md](gliner1-usage.md)'s architecture quick reference. **For new
work, prefer [joint-ie.md](joint-ie.md)'s `JointIE`** — it enforces the same
uniqueness/acyclic/no-self-loop constraints a real graph needs, and is what this skill's own
["Use case: knowledge graph construction"](joint-ie.md#use-case-knowledge-graph-construction)
section recommends.

**Large-scale entity extraction** — BiEncoder + `encode_labels()` once, `batch_predict_with_embeds()`
across millions of documents for max throughput.

**Domain-specific NER** — fine-tune a general checkpoint on 50-200 labeled domain examples (see
[gliner1-training.md](gliner1-training.md)); even a small in-domain set meaningfully improves
recall on specialized entity types (medications, legal clauses, financial instruments).

**Multi-lingual extraction** — `urchade/gliner_multi-v2.1` handles 100+ languages with the same
`predict_entities()` call, no per-language setup.

**Search/retrieval augmentation** — run `predict_entities()` on a user query, turn the results into
`{label: text}` filters to enrich a RAG/vector-DB query rather than passing raw query text.
