# GLiNER (v1): Training & Fine-Tuning

Fine-tuning workflow, dataset formats, and a runnable training script. Condensed from
[Training](https://urchade.github.io/GLiNER/training.html). For the GLiNER2 equivalent, see
[training.md](training.md) and [training-data-format.md](training-data-format.md); for per-
architecture config knobs, see
[gliner1-configs-architectures.md](gliner1-configs-architectures.md).

```bash
pip install gliner[training]
```

## Minimal example

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

train_data = [
    {"tokenized_text": ["Apple", "Inc.", "is", "headquartered", "in", "Cupertino"],
     "ner": [[0, 1, "organization"], [5, 5, "location"]]},
    {"tokenized_text": ["Steve", "Jobs", "founded", "Apple"],
     "ner": [[0, 1, "person"], [3, 3, "organization"]]},
]

trainer = model.train_model(
    train_dataset=train_data,
    eval_dataset=train_data,          # use a separate eval set in practice
    output_dir="./my_model",
    max_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
)
trainer.save_model()
```

## Dataset format

**Basic** (`"ner"` entries are inclusive **token** indices, not character offsets):

```python
{
    "tokenized_text": List[str],                     # pre-tokenized words
    "ner": List[List[Union[int, str]]],               # [[start_idx, end_idx, label], ...]
}
```

```python
{"tokenized_text": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
 "ner": [[0, 1, "person"], [5, 5, "location"]]}   # "Barack Obama" spans tokens 0-1
```

`"ner": []` is valid (a negative example with no entities).

**Explicit labels and hard negatives** — pass `ner_labels` (types relevant to this example) and
`ner_negatives` (types to sample as negatives) for finer control over label sampling:

```python
{
    "tokenized_text": ["Google", "CEO", "Sundar", "Pichai", "announced", "Pixel"],
    "ner": [[0, 0, "organization"], [1, 1, "position"], [2, 3, "person"], [5, 5, "product"]],
    "ner_labels": ["organization", "person", "position", "product"],
    "ner_negatives": ["company", "individual", "job_title", "brand"],  # similar types as hard negatives
}
```

Benefits: explicit control over which types are considered, hard negatives for confusable types,
curriculum learning (easy → hard negatives), and domain focus.

**Relation extraction** (`UniEncoderSpanRelex`/`UniEncoderTokenRelex`) — add `"relations"` with
indices into `"ner"`, plus optional `rel_labels`/`rel_negatives`:

```python
{
    "tokenized_text": ["John", "Smith", "works", "at", "Microsoft", "in", "Seattle"],
    "ner": [[0, 1, "person"], [4, 4, "organization"], [6, 6, "location"]],
    "relations": [[0, 1, "works_at"], [1, 2, "located_in"]],   # [head_ner_idx, tail_ner_idx, type]
    "rel_labels": ["works_at", "located_in", "founded_by"],
    "rel_negatives": ["competitor_of", "subsidiary_of"],
}
```

**Decoder-based models** (`UniEncoderSpanDecoder`/`UniEncoderTokenDecoder`) — same shape as basic;
`ner_labels` becomes the set of labels the decoder learns to generate.

## Training via config file

```yaml
model:
  model_name: "microsoft/deberta-v3-base"
  span_mode: "markerV0"
  max_width: 12
  hidden_size: 768
  dropout: 0.4
  max_len: 384
  max_types: 25

training:
  prev_path: null              # set to a checkpoint path to fine-tune instead of train from scratch
  num_steps: 10000
  train_batch_size: 8
  eval_every: 1000
  warmup_ratio: 0.1
  scheduler_type: "cosine"
  lr_encoder: 1e-5
  lr_others: 5e-5
  weight_decay_encoder: 0.01
  weight_decay_other: 0.01
  max_grad_norm: 1.0
  loss_alpha: -1                # focal loss alpha; >=0 activates it
  loss_gamma: 0
  loss_reduction: "sum"
  negatives: 1.0
  masking: "none"               # "none" | "global" | "label" | "span"
  save_total_limit: 3
  freeze_components: null       # e.g. ["text_encoder"]

data:
  root_dir: "models"
  train_data: "data/train.json"
  val_data_dir: "data/val.json"
```

A full `train.py` entry point that loads this config, builds/loads the model
(`GLiNER.from_config` or `GLiNER.from_pretrained(prev_path)`), and calls `model.train_model(...)`
with the mapped arguments is in the
[Training doc's Training Script section](https://urchade.github.io/GLiNER/training.html#training-script) —
copy it directly rather than hand-rolling the argument plumbing. Run with:

```bash
python train.py --config config.yaml
```

## Best practices

1. **Start from a pretrained checkpoint** — fine-tuning beats training from scratch almost always.
2. Use a **separate validation set**, not the training set, to catch overfitting.
3. Experiment with `negatives`/`masking` sampling strategies for your label distribution.
4. Use **focal loss** (`focal_loss_alpha`/`focal_loss_gamma` > 0) for imbalanced label sets.
5. **Freeze the encoder** (`freeze_components=["text_encoder"]`) for quick adaptation on small
   datasets.
6. Include **hard negatives** — similar-but-wrong types (e.g. "person" as a negative when
   "organization" is positive) sharpen the decision boundary.
7. Keep `save_total_limit > 1` to retain multiple checkpoints.
8. Track loss/metrics with TensorBoard or W&B.
9. Pilot on a data subset before a full run.
10. Double-check indices are token-level (not character-level) and within bounds — this is the
    most common silent data-format bug.

## Architecture-specific training details

See [gliner1-configs-architectures.md](gliner1-configs-architectures.md) for the full per-
architecture config parameter tables (BiEncoder's `labels_encoder`, Decoder's `labels_decoder`/
`decoder_mode`, Relex's `relations_layer`/`triples_layer`) and complete training YAML examples per
architecture. StreamingSpan trains on the ordinary span dataset format with no pre-chunking — see
[gliner1-streaming.md](gliner1-streaming.md#configuration-and-training).
