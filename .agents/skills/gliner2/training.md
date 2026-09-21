# Model Training

Fine-tune or train GLiNER2 models with `ExtractorTrainer` (`GLiNER2Trainer` is a backward-
compatible alias). Works with both span and boundary base checkpoints loaded via
`AutoExtractor.from_pretrained(...)`. Mirrors
[tutorial/9-training.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md).
Data format reference: [training-data-format.md](training-data-format.md).

```bash
pip install gliner2[train]
```

## Quick start

```python
from gliner2 import AutoExtractor
from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

examples = [
    InputExample(text="John works at Google in California.", entities={"person": ["John"], "company": ["Google"], "location": ["California"]}),
    InputExample(text="Apple released iPhone 15.", entities={"company": ["Apple"], "product": ["iPhone 15"]}),
]

model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")   # or fastino/gliner2.5-base-v1 for boundary
config = TrainingConfig(output_dir="./output", num_epochs=10, batch_size=8, encoder_lr=1e-5, task_lr=5e-4)

trainer = ExtractorTrainer(model, config)
trainer.train(train_data=examples)          # or train_data="train.jsonl"
```

## End-to-end pipeline

```python
from gliner2.training.data import InputExample, TrainingDataset

train_examples = [
    InputExample(
        text="Tim Cook is the CEO of Apple Inc., based in Cupertino, California.",
        entities={"person": ["Tim Cook"], "company": ["Apple Inc."], "location": ["Cupertino", "California"]},
        entity_descriptions={"person": "Full name of a person", "company": "Business organization name"},
    ),
    # ...more examples
]

dataset = TrainingDataset(train_examples)
dataset.validate(strict=True, raise_on_error=True)
dataset.print_stats()

train_data, val_data, _ = dataset.split(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, shuffle=True, seed=42)
train_data.save("train.jsonl")
val_data.save("val.jsonl")

model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./ner_model", num_epochs=15, batch_size=16, encoder_lr=1e-5, task_lr=5e-4,
    warmup_ratio=0.1, scheduler_type="cosine", fp16=True,
    eval_strategy="epoch", save_best=True, early_stopping=True, early_stopping_patience=3,
)
trainer = ExtractorTrainer(model, config)
results = trainer.train(train_data=train_data, eval_data=val_data)

best_model = AutoExtractor.from_pretrained("./ner_model/best")
```

## Multi-task training (entities + classification + relations)

```python
from gliner2.training.data import InputExample, Classification, Relation

examples = [
    InputExample(
        text="John Smith works at Google in California. The company is thriving.",
        entities={"person": ["John Smith"], "company": ["Google"], "location": ["California"]},
        classifications=[Classification(task="sentiment", labels=["positive", "negative", "neutral"], true_label="positive")],
        relations=[Relation("works_at", head="John Smith", tail="Google"), Relation("located_in", head="Google", tail="California")],
    ),
]
```

## Structured data with `Structure` / `ChoiceField`

```python
from gliner2.training.data import InputExample, Structure, ChoiceField

example = InputExample(
    text="Order #12345 for laptop shipped on 2024-01-15.",
    structures=[Structure("order", order_id="12345", product="laptop", date="2024-01-15",
                           status=ChoiceField(value="shipped", choices=["pending", "processing", "shipped", "delivered"]))],
)
```

When saved as JSONL and trained on a boundary model, structures automatically use `natural`
record formation with the **first declared field as anchor** — no extra `record_metadata` needed
for ordinary cases.

## Data input formats accepted by `trainer.train(train_data=...)`

```python
trainer.train(train_data="train.jsonl")                     # single JSONL file
trainer.train(train_data=["train1.jsonl", "train2.jsonl"])   # multiple files
trainer.train(train_data=[InputExample(...), ...])           # list of InputExample
trainer.train(train_data=TrainingDataset.load("train.jsonl")) # TrainingDataset object
trainer.train(train_data=[{"input": "...", "output": {...}}]) # raw dict list
```

## Data management

```python
dataset = TrainingDataset.load("full_data.jsonl")
train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True, seed=42)
entity_only = dataset.filter(lambda ex: len(ex.entities) > 0)
small_sample = dataset.sample(n=100, seed=42)

combined = TrainingDataset()
combined.add_many(TrainingDataset.load("d1.jsonl").examples)
combined.add_many(TrainingDataset.load("d2.jsonl").examples)
```

## Checkpointing & evaluation

This section is loss-based evaluation for picking a checkpoint during training. For measuring
whether the result is actually good on your data — precision/recall/F1, comparing zero-shot vs.
fine-tuned — see [evaluation.md](evaluation.md) instead.

`eval_strategy` controls **both** when to evaluate and when to save checkpoints:

| `eval_strategy` | Behavior |
|---|---|
| `"steps"` (default) | Evaluate + save every `eval_steps` |
| `"epoch"` | Evaluate + save at end of each epoch |
| `"no"` | No evaluation/checkpointing except the final one |

`save_best=True` additionally always keeps the best checkpoint by `metric_for_best`
(`"eval_loss"` default, `greater_is_better=False`).

## Configuration presets

```python
# Fast prototyping
TrainingConfig(output_dir="./quick_test", num_epochs=3, batch_size=16, max_train_samples=100, eval_strategy="no")

# Production
TrainingConfig(output_dir="./production_model", num_epochs=20, batch_size=32, gradient_accumulation_steps=2,
                encoder_lr=5e-6, task_lr=1e-4, scheduler_type="cosine", fp16=True,
                eval_strategy="steps", eval_steps=500, save_total_limit=5, save_best=True,
                early_stopping=True, early_stopping_patience=5, report_to_wandb=True, wandb_project="gliner2-production")

# Memory-optimized
TrainingConfig(output_dir="./large_model", batch_size=8, gradient_accumulation_steps=8,
                gradient_checkpointing=True, fp16=True, encoder_lr=1e-6, task_lr=5e-5, max_grad_norm=0.5, num_workers=2)
```

Key knobs: `num_epochs`, `max_steps`, `batch_size`/`eval_batch_size`,
`gradient_accumulation_steps`, `encoder_lr`/`task_lr`, `weight_decay`, `max_grad_norm`,
`scheduler_type` (`linear`|`cosine`|`cosine_restarts`|`constant`), `warmup_ratio`/`warmup_steps`,
`fp16`/`bf16`, `gradient_checkpointing`, `early_stopping[_patience/_threshold]`,
`report_to_wandb`/`wandb_*`, `seed`, `deterministic`, `validate_data`/`strict_validation`. LoRA
knobs live here too — see [lora-adapters.md](lora-adapters.md).

## Loading checkpoints / continuing training

```python
trainer = ExtractorTrainer(model, config)
trainer.load_checkpoint("./output/checkpoint-1000")   # weights only — no optimizer/scheduler state
trainer.train(train_data=examples)                     # training always starts fresh from these weights
```

## Advanced

- **Custom metrics**: pass `compute_metrics=fn` to `ExtractorTrainer(...)`.
- **Distributed training**: launch with `torchrun`; `TrainingConfig(local_rank=int(os.environ.get("LOCAL_RANK", -1)))`.
- **W&B**: `report_to_wandb=True`, `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags`, `wandb_notes`.
- **Data augmentation**: build extra `InputExample`s programmatically (e.g. shuffled entity dict
  order) and merge into a `TrainingDataset` before training.

## Troubleshooting

| Symptom | Fixes |
|---|---|
| OOM | lower `batch_size`; raise `gradient_accumulation_steps`; `gradient_checkpointing=True`; `fp16=True`; switch to LoRA (`use_lora=True`); lower `num_workers` |
| Slow training | raise `batch_size`/`num_workers`; `fp16=True`; reduce eval frequency (`eval_steps` larger); use LoRA |
| Validation errors | `dataset.validate(raise_on_error=False)` then inspect `report['errors']`; common cause: entity mention text not found verbatim in the input |
| Not learning | try `encoder_lr` in `5e-6`–`5e-5`, `task_lr` in `1e-4`–`1e-3`; more epochs; `warmup_ratio=0.1`; lower `weight_decay`; try `scheduler_type="cosine"`; re-check data quality |

## Best practices

1. Always `dataset.validate()` + `print_stats()` before training.
2. Prototype on `max_train_samples=100` first.
3. Use `early_stopping=True` for long runs.
4. Checkpoint on `eval_strategy="steps"` with `save_best=True` for anything non-trivial.
5. Add entity/field descriptions in training data — same accuracy lever as inference time.
6. Split data properly (`0.8/0.1/0.1` is a reasonable default) and keep a real test set.
7. Typical LRs: full fine-tune encoder `1e-6`–`5e-5` (commonly `1e-5`), task heads `1e-4`–`1e-3`
   (commonly `5e-4`); LoRA task LR `1e-4`–`1e-3` (commonly `5e-4`).
8. Consider LoRA (see [lora-adapters.md](lora-adapters.md)) whenever memory or multi-domain
   deployment is a concern.
