# GLiNER (v1): Architectures & Config Reference

Which architecture to pick, why it exists, and every config parameter needed to instantiate or
train it. Condensed from [Architectures](https://urchade.github.io/GLiNER/architectures.html) and
[Components & Configs](https://urchade.github.io/GLiNER/configs.html). GLiNER auto-selects the
implementation from the checkpoint's config — you generally only choose an architecture when
training from scratch or picking a checkpoint; for inference-only use, see
[gliner1-usage.md](gliner1-usage.md)'s "Choosing an architecture (quick reference)" instead.

## Architecture comparison

| Architecture | Encoding | Prediction level | Best for |
|---|---|---|---|
| `UniEncoderSpan` | One encoder for text+labels | Span-level | General NER, < ~30 entity types (original GLiNER) |
| `UniEncoderToken` | One encoder for text+labels | Token-level (BIO) | Long entities, summarization, multi-task |
| `BiEncoderSpan` | Separate text/label encoders | Span-level | 50-200+ entity types, production (cacheable label embeddings) |
| `BiEncoderToken` | Separate text/label encoders | Token-level | Long entities + many types |
| `UniEncoderSpanDecoder` | One encoder + generative decoder | Span-level + generation | Open-domain NER, on-the-fly label discovery |
| `UniEncoderTokenDecoder` | One encoder + generative decoder | Token-level + generation | Long entities + open vocabulary |
| `UniEncoderSpanRelex` | One encoder + relation layers | Span-level + relations | Knowledge graph construction |
| `UniEncoderTokenRelex` | One encoder + relation layers | Token-level + relations | Long entities with relations |
| `StreamingSpan` | Causal decoder, reusable KV cache | Span-level | Live transcripts, chat, log streams — see [gliner1-streaming.md](gliner1-streaming.md) |

### Decision flowchart

```
Incremental / live input?              → StreamingSpan
Complete input, need relations?
  Yes, long entities expected?  Yes    → UniEncoderTokenRelex
                                 No     → UniEncoderSpanRelex
  No, need open-vocabulary labels?
  Yes, long entities expected?  Yes    → UniEncoderTokenDecoder
                                 No     → UniEncoderSpanDecoder
  No, many entity types (>30)?
  Yes, long entities expected?  Yes    → BiEncoderToken
                                 No     → BiEncoderSpan
  No, long entities expected?   Yes    → UniEncoderToken
                                 No     → UniEncoderSpan
```

### Why these exist (one line each)

- **UniEncoderSpan** (vanilla GLiNER): concatenates entity-type prompt and text, one `[ENT]`
  marker per label, spans built from start/end token pairs up to `max_width=12`, scored by
  sigmoid(dot(label, span)). Trained with BCE. Decoding: flat (best non-overlapping spans) or
  nested (overlap allowed, no partial overlaps).
- **UniEncoderToken** (GLiNER Multi-task): token-level BIO scoring instead of span enumeration —
  removes the `max_width` ceiling, so it's the pick for multi-sentence entities. Adds a BiLSTM over
  encoder outputs for stability. Same family also gained QA/relation/summarization/classification
  pipelines (see [gliner1-pipelines.md](gliner1-pipelines.md)) and a self-training recipe
  (generate weak labels with the pretrained model, fine-tune on them with label smoothing).
- **BiEncoderSpan / BiEncoderToken**: decouples label encoding (a sentence-transformer, e.g.
  `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) from text encoding (DeBERTa)
  so label embeddings can be pre-computed once and reused. Fixes UniEncoder's ~30-type ceiling and
  the wasted compute of encoding types absent from a given document. Trained in two stages: 1M-
  sample joint pretraining, then 35k-sample high-quality fine-tuning. Focal loss recommended
  (mitigates the class imbalance that large batches worsen for span/label pair matching). Does
  **not** support token-embedding resizing (label-encoder vocabulary is fixed).
- **UniEncoderSpanDecoder / UniEncoderTokenDecoder**: adds a GPT-2-class generative decoder that
  proposes label text for detected spans instead of just classifying against a fixed set —
  `decoder_mode="prompt"` replaces the predefined types entirely, `"span"` adds generated labels
  alongside them. Two joint losses (`span_loss_coef`, `decoder_loss_coef`).
- **UniEncoderSpanRelex / UniEncoderTokenRelex**: adds a `[REL]` token family and a relation layer
  that builds an entity-pair adjacency matrix, then classifies relation type per surviving pair.
  Three joint losses: span, adjacency, relation.
- **StreamingSpan**: swaps the bidirectional encoder for a causal decoder so text can be appended
  incrementally without re-encoding history. See [gliner1-streaming.md](gliner1-streaming.md) for
  the full session/cache model.

## Base config parameters (`BaseGLiNERConfig`, shared by every architecture)

| Param | Default | Notes |
|---|---|---|
| `model_name` | `"microsoft/deberta-v3-small"` | Backbone encoder (Hub id or local path) |
| `max_width` | `12` | Widest candidate span, in splitter tokens (span architectures only) |
| `hidden_size` | `512` | Internal hidden dim |
| `dropout` | `0.4` | |
| `fine_tune` | `True` | Fine-tune the encoder during training |
| `subtoken_pooling` | `"first"` | `"first"` \| `"last"` \| `"mean"` \| `"max"` — keep consistent between train and inference |
| `span_mode` | `"markerV0"` | See span representation table below |
| `post_fusion_schema` | `""` | Hyphen-joined `l2l`/`t2t`/`l2t`/`t2l` attention steps fusing span+label embeddings; `""` disables fusion |
| `num_post_fusion_layers` | `1` | How many times the fusion schema repeats |
| `max_types` | `25` | Max entity types per batch |
| `max_len` | `384` | Text-only splitter-token limit — see [gliner1-usage.md](gliner1-usage.md)'s truncation section |
| `words_splitter_type` | `"whitespace"` | `whitespace`\|`universal`\|`spacy`\|`mecab`\|`jieba`\|`hanlp`\|`janome`\|`camel`\|`hindi`\|`stanza` |
| `num_rnn_layers` | `1` | LSTM layers atop encoder output; `0` disables |
| `fuse_layers` | `False` | Combine multi-encoder representations |
| `embed_ent_token` | `True` | Pool `< >` per label if `True`, else the label's first token |
| `encoder_config` | — | Nested backbone config dict |
| `span_encoder_config` | `None` | Optional `deberta-v2`/`modernbert`/`rnn` context encoder before span construction |
| `ent_token` / `sep_token` | `"< >"` | Boundary markers |
| `_attn_implementation` | — | e.g. `"eager"` to disable Flash Attention |

`span_mode` values: `markerV0` (default, lightweight MLP projection), `marker` (deeper 2-layer,
better for complex tasks), `query` (learned per-width query vectors), `mlp` (fast,
position-agnostic), `cat` (concat + width embedding), `conv_conv`/`conv_max`/`conv_mean`/`conv_sum`/
`conv_share` (convolutional span pooling variants).

## Per-architecture config

**UniEncoderSpan** — no extra params beyond base; leave `labels_encoder`/`labels_decoder`/
`relations_layer` unset.

**UniEncoderToken** — `span_mode="token-level"` (fixed); `num_rnn_layers=1` recommended.

**BiEncoderSpan / BiEncoderToken**

| Param | Notes |
|---|---|
| `labels_encoder` | Required — sentence-transformer id, e.g. `sentence-transformers/all-MiniLM-L6-v2` |
| `labels_encoder_config` | Optional nested config for it |

```python
from gliner import GLiNERConfig, GLiNER
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-base",
    labels_encoder="sentence-transformers/all-MiniLM-L6-v2",
    max_width=12, hidden_size=768, span_mode="markerV0",
)
model = GLiNER.from_config(config)
label_embeddings = model.encode_labels(labels)
entities = model.batch_predict_with_embeds(texts=[text], labels_embeddings=label_embeddings, labels=labels)
```

**UniEncoderSpanDecoder** / **UniEncoderTokenDecoder**

| Param | Default | Notes |
|---|---|---|
| `labels_decoder` | required | e.g. `"gpt2"`, `"distilgpt2"`, `"EleutherAI/gpt-neo-125M"` |
| `decoder_mode` | — | `"prompt"` (decoder output replaces fixed types) or `"span"` (adds to them) |
| `full_decoder_context` | `True` | Full span tokens vs. just boundary markers |
| `blank_entity_prob` | `0.1` | Probability of a generic "entity" label during training |
| `decoder_loss_coef` / `span_loss_coef` | `0.5` / `0.5` | Joint loss weights |
| `token_loss_coef` | — | Token variant only — third joint-loss term for the BIO token classifier |
| `represent_spans` | — | Token variant only — use a dedicated `SpanRepLayer` for richer span representations before decoding |

| Aspect | UniEncoderSpanDecoder | UniEncoderTokenDecoder |
|---|---|---|
| Entity detection | Span enumeration (max width 12) | Token-level BIO tagging |
| Long entities | Limited by max span width | No length limitation |
| Computation | O(n × max_width) spans | O(n) tokens |
| Best for | Standard NER entities + open labels | Long-form extraction + open labels |

Inference-time generation controls (both decoder variants) — pass a generic prompt label and let
the decoder propose the actual type text, optionally constrained to a candidate set:

```python
entities = model.inference(
    [text],
    labels=["entity"],                                   # generic prompt
    gen_constraints=["company", "person", "location"],   # optional: constrain generation to these
    num_gen_sequences=1,
)
for entity in entities[0]:
    print(entity["text"], "=>", entity["label"])
    if "generated_labels" in entity:
        print("  Generated:", entity["generated_labels"])  # present when decoding produced text
```

Without `gen_constraints`, the decoder generates free-text labels — useful for open-vocabulary
discovery, but unconstrained generation can produce labels outside any fixed taxonomy; validate
downstream if the caller needs a closed set.

**UniEncoderSpanRelex** / **UniEncoderTokenRelex**

| Param | Default | Notes |
|---|---|---|
| `relations_layer` | required | `"dot"` \| `"gcn"` \| `"gat"` |
| `triples_layer` | — | `"distmult"` \| `"complex"` \| `"transe"` |
| `embed_rel_token` | `True` | |
| `rel_token` | `"< >"` | |
| `span_loss_coef` / `adjacency_loss_coef` / `relation_loss_coef` | `1.0` each | Joint loss weights |

Special tokens: `[ENT]` marks entity types (as in vanilla GLiNER), `[REL]` marks relation types,
`[SEP]` separates entity types / relation types / input text. Forward pass: encode text with both
prompts → extract entity spans → build candidate entity pairs from the predicted adjacency matrix
→ classify relation type per surviving pair. `relations_layer` controls how a pair's joint
representation is built (`"dot"` = simple dot product; `"gcn"`/`"gat"` = graph-network message
passing over all pairs, capturing multi-hop interactions a pairwise dot product misses);
`triples_layer` then scores the resulting `(head, relation, tail)` triple.

Training data adds a `"relations"` key: `[[head_entity_idx, tail_entity_idx, relation_type], ...]`
(indices into that example's `"ner"` list).

## Training config (YAML) shape

All architectures share the same top-level shape; only the model block changes:

```yaml
model_name: microsoft/deberta-v3-base
labels_encoder: null          # set for BiEncoder*
labels_decoder: null          # set for *Decoder
relations_layer: null         # set for *Relex
name: "my gliner model"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
span_mode: markerV0           # or token-level, per architecture
post_fusion_schema: ""

num_steps: 30000
train_batch_size: 8           # smaller (4-6) for decoder/relex — heavier per-step compute
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: cosine

loss_alpha: -1                # >=0 activates focal loss; recommended (e.g. 0.25) for BiEncoder
loss_gamma: 0                 # focal loss gamma, e.g. 2.0 with loss_alpha
label_smoothing: 0
loss_reduction: sum

lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

train_data: "data.json"
prev_path: null                # set to a checkpoint path to fine-tune instead of train from scratch
save_total_limit: 3
max_types: 25
max_len: 384
```

## `TrainingArguments` (GLiNER extensions to `transformers.TrainingArguments`)

| Param | Default | Notes |
|---|---|---|
| `others_lr` | `learning_rate` | LR for non-encoder params (span layers, label encoder) |
| `others_weight_decay` | `0.0` | |
| `focal_loss_alpha` | `-1` | `>= 0` activates focal loss: `FL(p_t) = -α(1-p_t)^γ log(p_t)` |
| `focal_loss_gamma` | `0` | Higher = more focus on hard examples |
| `focal_loss_prob_margin` | `0.0` | |
| `label_smoothing` | `0.0` | |
| `loss_reduction` | `"sum"` | `"sum"` \| `"mean"` |
| `negatives` | `1.0` | Negative:positive span sampling ratio |
| `masking` | — | `"none"` \| `"global"` \| `"label"` \| `"span"` |

See [gliner1-training.md](gliner1-training.md) for the full training workflow and dataset format.
