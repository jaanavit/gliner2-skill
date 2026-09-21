# GLiNER (v1): StreamingSpan (Incremental Text)

For text that arrives incrementally — live transcripts, chat, log streams, progressively-uploaded
documents. Condensed from
[Advanced Usage](https://urchade.github.io/GLiNER/usage.html#streamingspan-models) and
[Architectures](https://urchade.github.io/GLiNER/architectures.html#gliner-streamingspan). GLiNER2
has no equivalent — this is a GLiNER (v1)-only architecture; if you don't need incremental
append-and-revise semantics, use `*_long` methods in [long-context.md](long-context.md) (GLiNER2)
or windowing in [gliner1-usage.md](gliner1-usage.md) (GLiNER v1) instead.

## What it is

StreamingSpan replaces the bidirectional text encoder with a **causal decoder** backbone. Later
calls append new text to the decoder's KV cache instead of re-encoding the whole document, and the
model revises recent span predictions as new right-context arrives. `GLiNER.from_pretrained` reads
`model_type="gliner_streaming_span"` and dispatches to `StreamingSpanGLiNER` automatically.

Don't confuse this with `UniEncoderSpanDecoder` — that uses an *auxiliary* decoder to generate
entity **labels**; StreamingSpan uses a decoder as its **text backbone** and still scores spans
against caller-provided entity types.

Starter checkpoint: `knowledgator/gliner-stream-pii-v1.0`.

## Quick start

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-stream-pii-v1.0")
model.eval()

labels = ["person", "email address", "phone number"]
session_id = "support-call-42"
chunks = [
    "Customer Alice Johnson ",
    "can be reached at alice@example.com ",
    "or +1 202-555-0147.",
]

try:
    for chunk in chunks:
        snapshot = model.inference([chunk], labels, session_id=[session_id], threshold=0.5)[0]
        print(snapshot)
finally:
    model.clear_session(session_id)
```

Each `snapshot` is the **complete current entity set for the accumulated session text**, not a
delta of just the new chunk. Diff consecutive snapshots by `(start, end, label)` if you need an
event stream. Offsets are document-relative. Keep each session's labels/descriptions/order
consistent across calls — changing them requires `recompute=True` or clearing the session. A blank
chunk returns `[]` and doesn't advance the session.

## How it works

1. **Cold pass**: the label prompt (`person< >email address< >< >`) plus the first chunk pass
   through the causal decoder. A compact label context encoder (DeBERTa-v2/ModernBERT/RNN)
   processes the prompt slice up to `< >`, producing one cached representation per entity type.
   Text subtokens are pooled into cached word states.
2. **Warm pass**: only the new chunk goes through the decoder, attending to cached history via
   continuing position IDs. Label representations are reused; new word states are pooled and
   appended.
3. **Rolling revision**: candidate spans use the `markerV2` representation (first word + last word
   + latest visible word, each projected). Every append scores spans ending in the new chunk *and*
   re-scores spans whose end is within `right_context_width` words of the newest word (default =
   `max_width`; `0` disables revision — append-only). New scores replace stored scores for the same
   `(start, end)`. `recompute=True` rebuilds everything from the full accumulated text — a
   correctness check that forfeits incremental savings for that call.

## Choosing an inference surface

| Surface | Cache ownership | Scheduling | Best fit |
|---|---|---|---|
| `predict_entities`/`inference` without `session_id` | None | Ordinary batching | Complete, independent texts |
| `inference(..., session_id=[...])` | One cache per ID on the model | Compatible sessions batched per call | Flexible synchronous session sets |
| `create_streaming_batch(...)` | One persistent batched cache | Fixed rows advance together | Stable groups with aligned arrival cadence |
| `create_async_streaming_engine(...)` | One cache per ID | Dynamic microbatching | Concurrent, independently-arriving streams |

```python
# Flexible synchronous sessions — one stable, unique session_id per text
session_ids = ["call-a", "call-b"]
first = model.inference(["Alice Johnson ", "Bob Smith "], labels, session_id=session_ids)
second = model.inference(["shared her email.", "shared his number."], labels, session_id=session_ids)
model.clear_session(session_ids)
```

```python
# Persistent fixed-order batch — avoids repeated stack/split of historical KV caches
with model.create_streaming_batch(session_ids=["call-a", "call-b"], labels=labels) as stream:
    first = stream.append(["Alice Johnson ", "Bob Smith "])
    third = stream.append(["", " It is +1 202-555-0147."])  # "" keeps call-a's row unchanged
# stream.reset() discards the cache but keeps the handle; close() (or exiting `with`) ends it
```

```python
# Async dynamic microbatching — independently-arriving concurrent streams
import asyncio

async def consume(engine, session_id, chunks):
    async for latest in engine.stream(session_id, chunks, labels, threshold=0.5):
        print(session_id, latest)

async def main():
    async with model.create_async_streaming_engine(max_batch_size=32, batch_wait_timeout_ms=2) as engine:
        await asyncio.gather(
            consume(engine, "call-a", ["Alice ", "shared her email."]),
            consume(engine, "call-b", ["Bob ", "shared his number."]),
        )
        await engine.clear_session("call-a")
        await engine.clear_session("call-b")
```

`max_batch_size` caps a microbatch, `batch_wait_timeout_ms` trades latency for batching
opportunity, `queue_capacity` applies backpressure. Calls for one session stay FIFO; different
sessions run concurrently.

## Session lifecycle and limits

```python
model.clear_session("call-a")
model.clear_session(["call-b", "call-c"])
model.clear_sessions()          # every model-owned session
print(model.session_count)      # persistent StreamingBatch handles are NOT counted here —
                                 # use the handle's reset()/close() instead
```

No automatic TTL or LRU eviction — always clear abandoned sessions.

**Context limit** = min(`max_cache_length`, the decoder's native positional limit) minus the
serialized label prompt. Stateless `config.max_len` truncation ([gliner1-usage.md](gliner1-usage.md))
does **not** apply to cached sessions — instead, an append that would exceed the limit **raises
`ValueError`** (no silent truncation). Finish/clear the session or start a new one.

Cached state, for reference: decoder KV + attention state, pooled word states, label
representations, text/token/character-offset maps, and a span-score history (kept on CPU so it
doesn't grow accelerator memory).

## Prediction controls

Standard span-decoding options apply: `threshold` (default `0.5`), `flat_ner`, `multi_label`,
`return_class_probs`, `recompute`. **Not supported** with `session_id`: `packing_config`,
`input_spans`, external model-input tensors (these remain available on the stateless path).

## Chunking guidance

- Prefer chunks ending at word/punctuation/sentence boundaries.
- Preserve boundary whitespace — `"Alice "` + `"joined"` → `"Alice joined"`; `"Alice"` + `"joined"`
  → `"Alicejoined"`.
- Smaller chunks = earlier updates, more overhead. Larger chunks = better throughput, later first
  prediction.
- Use stable, tenant-safe session IDs; always clear abandoned sessions.
- Treat every response as complete replaceable state, never append snapshots as if they were
  deltas.

## Configuration and training

Checkpoints use `model_type: gliner_streaming_span`. Key fields: `model_name`/`decoder_config`
(causal backbone), `label_token`/`sep_token` (default `< >`), `labels_encoder_config`,
`span_mode` (defaults `markerV2`), `span_encoder_config`, `subtoken_pooling`, `max_width`,
`right_context_width`, `max_cache_length`, `max_len` (stateless-only). Train/fine-tune with the
normal GLiNER data format (no pre-chunking needed) via `configs/config_streaming_span.yaml` and
`python train.py --config configs/config_streaming_span.yaml` — see
[gliner1-training.md](gliner1-training.md).
