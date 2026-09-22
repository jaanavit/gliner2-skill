# LoRA Adapters

Parameter-efficient fine-tuning: train a small adapter (~2–10 MB) per domain instead of a full
model (~100–500 MB), while keeping the base model frozen. Mirrors
[tutorial/10-lora_adapters.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/10-lora_adapters.md)
and the LoRA section of
[tutorial/9-training.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md).
For loading/routing between already-trained adapters at inference time, see
[adapter-switching.md](adapter-switching.md).

## Why LoRA

```
Full fine-tuning:  legal 450MB + medical 450MB + financial 450MB = 1.35 GB
LoRA adapters:     base 450MB + legal 5MB + medical 5MB + financial 5MB = 465 MB (~65% less)
```

Also: ~2–3x faster training (only ~0.1–5% of parameters are trainable) and lower GPU memory.
Loading a saved checkpoint back depends on `save_adapter_only` — see below.

## Basic LoRA training

```python
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")

config = TrainingConfig(
    output_dir="./output_lora", num_epochs=10, batch_size=16,
    use_lora=True,
    lora_r=16,                 # rank: higher = more params, better approximation (typical 4/8/16/32/64)
    lora_alpha=32.0,           # scaling factor, conventionally 2*r
    lora_dropout=0.1,          # regularization
    lora_target_modules=["encoder"],
    save_adapter_only=True,    # save only the adapter (~2-10MB), not the full model
    task_lr=5e-4,              # used for LoRA + task heads; encoder_lr is ignored when LoRA is enabled
    fp16=True, eval_strategy="epoch", save_best=True,
)
trainer = ExtractorTrainer(model, config)
trainer.train(train_data="train.jsonl", eval_data="val.jsonl")

# save_adapter_only=True above -- the checkpoint is adapter-only, not merged.
# Load the base model first, then attach the adapter:
best_model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")
best_model.load_adapter("./output_lora/best")
```

## Loading a saved checkpoint

Which call is correct depends on how the checkpoint was saved:

- **Full/merged checkpoint** (`save_adapter_only=False`): the directory already has base weights
  and adapter deltas merged. `AutoExtractor.from_pretrained(path)` loads it directly.
- **Adapter-only checkpoint** (`save_adapter_only=True`, the recommended default below): the
  directory has only the small adapter weights, not the base model. Load the base model first,
  then attach the adapter: `model = AutoExtractor.from_pretrained(base_model_name)` followed by
  `model.load_adapter(path)`.

Calling `AutoExtractor.from_pretrained()` directly on an adapter-only checkpoint's path is the
wrong call for that save mode — match the loading call to how `save_adapter_only` was actually
set during training, not the other way around.

## LoRA parameters

| Param | Meaning | Typical |
|---|---|---|
| `lora_r` | Rank of the low-rank decomposition; higher = more capacity/params | 4 (small data) – 64 (large data); start at 8–16 |
| `lora_alpha` | Scaling factor; effective scaling is `alpha/r` | `2 * lora_r` |
| `lora_dropout` | Dropout on the LoRA path | 0.0–0.1 |
| `lora_target_modules` | Which module groups receive LoRA | see below |
| `save_adapter_only` | `True` saves ~2–10MB adapter; `False` saves full merged model | `True` |
| `task_lr` | LR for LoRA + task heads when `use_lora=True` (`encoder_lr` unused) | `1e-4`–`1e-3`, commonly `5e-4` |

### Target module groups (span/legacy checkpoints)

| Group | Scope |
|---|---|
| `"encoder"` | All encoder layers (query, key, value, dense) |
| `"encoder.query"` / `"encoder.key"` / `"encoder.value"` | Only that attention projection |
| `"encoder.dense"` | Only FFN layers |
| `"span_rep"` | All linear layers in span representation |
| `"classifier"` | All linear layers in the classifier head |
| `"count_embed"` / `"count_pred"` | Count embedding / prediction layers |

Default target set (max adaptation): `["encoder", "span_rep", "classifier", "count_embed", "count_pred"]`.

### GLiNER2.5 boundary checkpoints — high-level aliases

```python
model.apply_lora(targets=["encoder", "all_task_heads"])
# or selectively: ["extractive_head", "classification_head", "record_head", "relation_head"]
```

See the
[boundary architecture guide](https://github.com/fastino-ai/GLiNER2/blob/main/docs/boundary_baseline.md)
for the full alias list.

## Sizing recipes

```python
# Small datasets (<1K examples)
TrainingConfig(use_lora=True, lora_r=4, lora_alpha=8.0, num_epochs=10)

# Medium datasets (1K-10K examples)
TrainingConfig(use_lora=True, lora_r=8, lora_alpha=16.0, num_epochs=5)

# Large datasets (>10K examples)
TrainingConfig(use_lora=True, lora_r=16, lora_alpha=32.0, num_epochs=3)

# Memory-constrained
TrainingConfig(use_lora=True, lora_r=8, batch_size=32, lora_target_modules=["encoder.query", "encoder.key", "encoder.value"])

# Higher performance (adds task heads, higher rank)
TrainingConfig(use_lora=True, lora_r=32, lora_alpha=64, lora_dropout=0.05,
               lora_target_modules=["encoder", "span_rep", "classifier"], task_lr=1e-3, num_epochs=15)
```

## LoRA vs. full fine-tuning

| Aspect | LoRA | Full fine-tuning |
|---|---|---|
| Trainable params | ~0.1–1% of model | 100% |
| Memory | Low | High |
| Speed | Fast | Slower |
| Checkpoint size | Small (adapter-only) or large (if merged) | Large |
| Performance | Good, often comparable | Best |
| Best for | Limited data, multiple domains/tasks | Large single-domain datasets |

## Domain-adapter pattern (train several adapters from one base)

```python
def train_domain_adapter(base_model_name, examples, domain_name, output_dir="./adapters"):
    config = TrainingConfig(
        output_dir=f"{output_dir}/{domain_name}_adapter", num_epochs=10, batch_size=8,
        gradient_accumulation_steps=2, task_lr=5e-4,
        use_lora=True, lora_r=8, lora_alpha=16.0, lora_target_modules=["encoder"], save_adapter_only=True,
        eval_strategy="no", fp16=True,
    )
    model = AutoExtractor.from_pretrained(base_model_name)
    trainer = ExtractorTrainer(model, config)
    trainer.train(train_data=examples)
    return f"{output_dir}/{domain_name}_adapter/final"

legal_path = train_domain_adapter("fastino/gliner2-base-v1", legal_examples, "legal")
medical_path = train_domain_adapter("fastino/gliner2-base-v1", medical_examples, "medical")
```

Then load/swap with `model.load_adapter(path)` — see [adapter-switching.md](adapter-switching.md).

## Best practices

1. Start with defaults (`r=16`, `alpha=32`, `dropout=0.1`); raise `r` to 32/64 only if
   performance is insufficient.
2. Target attention layers first (`["encoder.query", "encoder.key", "encoder.value"]`); add
   `"encoder.dense"` or task heads if needed.
3. Use `task_lr` in `5e-4`–`1e-3` for LoRA.
4. `save_adapter_only=True` almost always, for the storage savings — but that means the
   checkpoint is adapter-only, not pre-merged: load it with `model.load_adapter(path)` on an
   already-loaded base model (see "Loading a saved checkpoint" above), not
   `AutoExtractor.from_pretrained(path)` directly.
5. Version and record adapter metadata (rank, alpha, training date, sample count, eval F1) per
   adapter directory so you can compare revisions later.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM with LoRA | lower `batch_size`; raise `gradient_accumulation_steps`; smaller `lora_r`; `fp16=True`; target only attention layers |
| LoRA underperforms full fine-tune | raise `lora_r`; add task heads to `lora_target_modules`; raise `task_lr`; train longer |
