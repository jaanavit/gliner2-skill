# Evaluation

How to measure whether a GLiNER2 schema or model is actually good on your data — a held-out
validation/test split, task-appropriate metrics, and comparing zero-shot vs. fine-tuned before you ship
either one. GLiNER2's own tutorials don't cover this; there is no upstream tutorial to mirror
here, so treat this file as this skill's own addition rather than a condensed source doc.

## This is not `training.md`'s `eval_strategy`

`training.md`'s `eval_strategy`/`eval_steps`/`compute_metrics` evaluate **loss** during
training, to pick the best checkpoint and drive early stopping. That answers "did training
converge," not "is this good enough to ship." This file covers the latter: task-level metrics
computed over a labeled held-out set, for a zero-shot model or a fine-tuned one, before a
deploy decision.

## Build validation and held-out test sets

Use the same JSONL shape as [training-data-format.md](training-data-format.md). Tune descriptions,
thresholds, and model choices on validation data; reserve the test set for the final comparison:

```python
from gliner2.training.data import TrainingDataset

dataset = TrainingDataset.load("labeled.jsonl")
train, val, test = dataset.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True, seed=42)
val.save("validation.jsonl")  # tune descriptions, thresholds, and model choices here
test.save("test.jsonl")       # frozen -- use only for the final comparison
```

Include the failure cases that motivated fine-tuning in the first place. A test set that's
mostly easy examples can't distinguish a zero-shot model from a fine-tuned one — both will
score >95% and the comparison tells you nothing.

## Train-or-not decision

1. Freeze the test set.
2. Tune label/field descriptions and thresholds on validation data.
3. Run the base checkpoint on the frozen test set and compare the result with the product's
   precision/recall/F1 target.
4. If it meets the target, keep base inference. Do not fine-tune merely because training is
   available.
5. If it remains below target, label roughly 100–200 representative misses, train LoRA first
   when data or compute is limited, and compare it with both the base model and any full
   fine-tune on the identical frozen set.

The sample count is a starting point, not a guarantee. Difficult or imbalanced domains may need
more examples; learning curves are better evidence than a fixed number.

## Entity extraction metrics

Exact text match by default — safer starting point than fuzzy overlap, which hides real misses.
**Micro-average** (accumulate raw tp/fp/fn across the whole set) rather than macro-average
(average each example's F1): macro overweights tiny examples where one wrong span swings F1
from 0 to 1.

```python
def accumulate_entity_counts(
    predicted: dict[str, list[str]], gold: dict[str, list[str]], counts: dict[str, int]
) -> None:
    for label in set(predicted) | set(gold):
        pred_set = {s.strip().lower() for s in predicted.get(label, [])}
        gold_set = {s.strip().lower() for s in gold.get(label, [])}
        counts["tp"] += len(pred_set & gold_set)
        counts["fp"] += len(pred_set - gold_set)
        counts["fn"] += len(gold_set - pred_set)


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
```

Run it over the test set:

```python
from gliner2 import AutoExtractor
from gliner2.training.data import TrainingDataset

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
test_data = TrainingDataset.load("test.jsonl")

counts = {"tp": 0, "fp": 0, "fn": 0}
for example in test_data.examples:
    predicted = extractor.extract_entities(example.text, list(example.entities))
    accumulate_entity_counts(predicted["entities"], example.entities, counts)

print(precision_recall_f1(counts))
```

`.strip().lower()` normalizes whitespace/case before comparing — drop `.lower()` if case is
semantically meaningful for the task (e.g. distinguishing a proper noun from a common word).
Span-level (start/end offset) comparison is stricter than text-level; pick one and don't mix
the two within a report.

## Classification metrics (single- and multi-label)

Per-label tp/fp/fn, same accumulate-then-divide shape, plus a macro-F1 across labels (mean of
each label's F1) since multi-label classes are rarely balanced:

```python
def accumulate_classification_counts(
    predicted_labels: set[str], gold_labels: set[str], all_labels: list[str], per_label: dict[str, dict[str, int]]
) -> None:
    for label in all_labels:
        c = per_label.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        if label in predicted_labels and label in gold_labels:
            c["tp"] += 1
        elif label in predicted_labels:
            c["fp"] += 1
        elif label in gold_labels:
            c["fn"] += 1


def macro_f1(per_label: dict[str, dict[str, int]]) -> float:
    scores = [precision_recall_f1(c)["f1"] for c in per_label.values()]
    return sum(scores) / len(scores) if scores else 0.0
```

For single-label tasks, `predicted_labels`/`gold_labels` are one-element sets and this reduces
to ordinary accuracy once summed.

## Relation extraction metrics

Exact-match on the full `(head, relation_type, tail)` triple — a relation is only correct if
all three components match, not just the type:

```python
def relation_set(relations: list[dict]) -> set[tuple[str, str, str]]:
    return {(r["head"].strip().lower(), r["type"], r["tail"].strip().lower()) for r in relations}

predicted_triples = relation_set(predicted_relations)
gold_triples = relation_set(gold_relations)
tp = len(predicted_triples & gold_triples)
fp = len(predicted_triples - gold_triples)
fn = len(gold_triples - predicted_triples)
```

## Structured and combined-schema metrics

For structured extraction, compare each requested field independently and report exact-match
precision/recall/F1 per field. Normalize only what the product contract permits (for example,
case or whitespace); do not use fuzzy matching by default because it can hide incorrect values.
For repeated records, match records by their declared anchor field before scoring the remaining
fields.

For a combined schema, report each task separately: entity metrics, classification metrics,
relation triples, and structured fields. Do not collapse them into one blended score. The same
rule applies to `JointIE`: report entity quality and relation-triple quality independently even
though they were decoded together.

## Comparing zero-shot vs. fine-tuned

Same test set, same metric function, for both models — never compare a zero-shot run against
the fine-tuned model's own training-time validation numbers, and never let either model see
the test set before this comparison:

```python
zero_shot = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
fine_tuned = AutoExtractor.from_pretrained("./my_model/best")

for name, model in [("zero-shot", zero_shot), ("fine-tuned", fine_tuned)]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for example in test_data.examples:
        predicted = model.extract_entities(example.text, list(example.entities))
        accumulate_entity_counts(predicted["entities"], example.entities, counts)
    print(name, precision_recall_f1(counts))
```

## Threshold sweep

Pick the operating point empirically on the validation set, not by guessing — lower `threshold`
trades precision for recall. After selecting it, evaluate that fixed threshold once on the
frozen test set:

```python
validation_data = TrainingDataset.load("validation.jsonl")

for t in [0.3, 0.5, 0.7, 0.9]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for example in validation_data.examples:
        predicted = extractor.extract_entities(example.text, list(example.entities), threshold=t)
        accumulate_entity_counts(predicted["entities"], example.entities, counts)
    print(t, precision_recall_f1(counts))

selected_threshold = 0.7  # chosen from validation results
counts = {"tp": 0, "fp": 0, "fn": 0}
for example in test_data.examples:
    predicted = extractor.extract_entities(
        example.text, list(example.entities), threshold=selected_threshold
    )
    accumulate_entity_counts(predicted["entities"], example.entities, counts)
print("final test", precision_recall_f1(counts))
```

## Common pitfalls

- **A too-easy test set.** If zero-shot already scores >95%, the set can't show a fine-tune's
  benefit — pull in the actual production misses that motivated fine-tuning.
- **Touching the test set mid-iteration.** Tune on validation data and keep the test set frozen.
  If you must change it, re-run every model you've compared so far, not just the
  one you're currently working on — this is the same "re-run the full battery" discipline
  `SKILL.md`'s router-tuning caveat describes.
- **Mixing micro and macro without saying so.** Report which one a number is; they diverge a lot
  on imbalanced label sets.
- **Reporting one blended number across task types.** A schema with both NER and classification
  can be strong at one and weak at the other — report per-task-type, not one aggregate score.

## Best practices

1. Freeze the test set before tuning anything. Tune schemas, thresholds, and checkpoints on
   validation data; the test set is the one artifact you don't touch mid-iteration.
2. Report micro-averaged F1 as the headline number, but keep the per-label/per-type breakdown —
   an aggregate can hide one badly-underperforming label.
3. Log metrics per schema/checkpoint version (a CSV or JSON line per run is enough) so a
   regression from a later change is visible, not just improvements.
4. Compare zero-shot and fine-tuned on the identical test set and code path — same extractor
   call, same metric function, only the checkpoint differs.
