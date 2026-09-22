# Pioneer-Hosted Training & Inference API (`api.pioneer.ai`)

Fastino's hosted platform: upload a dataset, train a model, and run inference against it, all
through `https://api.pioneer.ai`. Covers model selection, training, dataset upload, the
combined-schema inference request shape, batching, retries, and the OpenAI-compatible surface.

## Setup

No Pioneer account yet? Sign up or log in at
[agent.pioneer.ai/auth](https://agent.pioneer.ai/auth) (Google, GitHub, magic link, or email),
then get a key from Settings → API Keys.

```bash
export PIONEER_API_KEY="pio_sk_..."
```

See **Request shape** below for a full worked call.

## Picking a base model

```bash
curl "https://api.pioneer.ai/base-models?supports_inference=true&task_type=encoder" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

| `model_id` | Notes |
|---|---|
| `fastino/gliner2-base-v1` | Legacy span, English |
| `fastino/gliner2-large-v1` | Legacy span, higher accuracy |
| `fastino/gliner2-multi-v1` | Legacy span, multilingual |
| `fastino/gliner2-multi-large-v1` | Legacy span, multilingual, larger |
| `fastino/gliguard-LLMGuardrails-300M` | LLM safety / guardrails moderation |
| `fastino/gliner2-privacy-filter-PII-multi` | PII detection |
| `fastino/gliguard-PII-multi` | Combined guardrails + PII checkpoint |

GLiNER2.5 (boundary) isn't in this catalog yet — only legacy span checkpoints are. Always confirm
the `model_id` you need against `GET /base-models` rather than assuming a name works unchanged.

## Uploading a dataset

Three calls: request a presigned upload URL, `PUT` the file to it, then tell Pioneer to process it.

```bash
curl -X POST https://api.pioneer.ai/felix/datasets/upload/url \
  -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{"dataset_name": "sentiment-ds", "dataset_type": "classification", "format": "jsonl", "filename": "sentiment.jsonl"}'
# {"presigned_url": "https://...", "dataset_id": "...", "dataset_name": "sentiment-ds", "version_number": "1", "expires_in": 3600}

curl -X PUT "$PRESIGNED_URL" -H "Content-Type: application/octet-stream" --data-binary @sentiment.jsonl

curl -X POST https://api.pioneer.ai/felix/datasets/upload/process \
  -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{"dataset_id": "..."}'
```

Poll `GET /felix/datasets/{name}/{version}` until `"status": "ready"` (it starts at
`"initialized"`). A successful response includes `sample_size` and an auto-detected `labels` list.

**Row format is per `dataset_type` and different from local `training-data-format.md`:**

```json
// dataset_type: "classification" -- one label per row
{"text": "I love this product!", "label": "positive"}
```

```json
// dataset_type: "ner" -- flat [span, label] pairs, not the local {"entities": {label: [spans]}} shape
{"text": "John Smith works at OpenAI in San Francisco.", "entities": [["John Smith", "person"], ["OpenAI", "organization"], ["San Francisco", "location"]]}
```

A row that doesn't match the expected shape for `dataset_type` fails dataset processing with a
`processing_error` naming the exact columns expected — check that field if `status` comes back
`"failed"` instead of `"ready"`.

## Training a model

```bash
curl -X POST https://api.pioneer.ai/felix/training-jobs \
  -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-classifier-v1",
    "base_model": "fastino/gliner2-base-v1",
    "datasets": [{"name": "sentiment-ds", "version": "1"}],
    "nr_epochs": 10,
    "validation_data_percentage": 0.2
  }'
```

Required: `model_name`, `base_model`, `datasets` (a list of `{name, version}` references to an
already-`ready` dataset — not raw text/files). Commonly-used optional fields: `training_type`
(`"lora"` default or `"full"`), `nr_epochs`, `learning_rate`, `batch_size`,
`validation_data_percentage`, `early_stopping_patience`, `warmup_ratio`, `seed`, `project_id`.
`task_type` and `labels` are inferred from the dataset, not passed in the request.

The response includes an `id` (job UUID) immediately — training runs asynchronously from there.

## Monitoring a job

```bash
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}" -H "X-API-Key: $PIONEER_API_KEY"
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}/logs" -H "X-API-Key: $PIONEER_API_KEY"
```

`status`/`normalized_status`/`is_terminal_status` track progress; `job_reference` shows the
underlying provider job once dispatched (e.g. `modal:fc-...`). On success, `trained_model_path`
is populated and `is_deployable` flips `true`. **The job's `id` is the UUID to pass as `model_id`
at `/inference` below** — no separate deploy step needed to call it.

As of this writing, training jobs against `fastino/gliner2-base-v1` reliably reach a terminal
`"errored"` state within seconds regardless of `task_type`/`training_type`, with
`error_message: "Modal training failed"` and no further detail in `/logs`. Dataset upload and job
creation both work mechanically (confirmed end-to-end); the actual training run failing looks
like a current provider-side issue rather than a request-format problem — verify job status
before assuming a training run will complete.

## Request shape

```bash
curl -X POST https://api.pioneer.ai/inference \
  -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model_id": "fastino/gliner2-base-v1",
    "text": "Apple announced the MacBook Pro at WWDC in Cupertino.",
    "schema": {"entities": ["organization", "product", "event", "location"]},
    "threshold": 0.5
  }'
# {"type": "encoder", "inference_id": "...", "result": {"entities": {
#   "organization": [{"text": "Apple", "confidence": 1.0, "start": 0, "end": 5}],
#   "product": [{"text": "MacBook Pro", ...}], "event": [{"text": "WWDC", ...}],
#   "location": [{"text": "Cupertino", ...}]}},
#  "model_id": "fastino/gliner2-base-v1", "latency_ms": 222.5, "token_usage": 136,
#  "model_used": "fastino/gliner2-base-v1"}
```

`model_id` is either a catalog base model or a training-job UUID from above. `schema` is always
a dict (never a flat list of label strings), and takes any combination of these keys in one
request:

```json
{
  "entities": ["organization", "product", "location"],
  "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"]}],
  "relations": ["works_for", "lives_in"],
  "structures": {"product": ["name::str", "price::str", "features::list"]}
}
```

- `relations` is a flat list of relation names, not a list of `{relation, head, tail}` objects.
  The response returns one `{relation_type: {text, confidence}, ...}` object per detected
  relation instance — a different shape from the local package's per-type list in
  [relation-extraction.md](relation-extraction.md), so read the response directly rather than
  assuming it matches.
- `structures` uses the same `field::type::description` spec syntax as
  [json-extraction.md](json-extraction.md)'s `extract_json()`. Verify list-valued fields
  (`::list`) against your own data — don't assume every match in the text is collected without
  checking a sample response.
- Do not add a `"task"` field to the request. Omit it entirely; the `schema` dict above is the
  complete, self-describing request.

## Batch and output shape

```bash
curl -X POST https://api.pioneer.ai/inference -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model_id": "fastino/gliner2-base-v1", "text": ["Google hired Jane Doe.", "Tesla launched the Model 3."], "schema": {"entities": ["company", "person", "product"]}}'
# result: [{...}, {...}] -- one object per input text, same shape as a single call
```

`include_confidence` and `include_spans` both default to `true` and collapse the output when set
`false`:

```bash
-d '{"model_id": "fastino/gliner2-base-v1", "text": "...", "schema": {"entities": [...]}, "include_confidence": false, "include_spans": false}'
# entities become plain strings: {"organization": ["Apple"], "product": ["MacBook Pro"]}
```

Other useful request fields: `format_results` (default `true`; `false` returns raw tuples),
`is_warmup` (default `false`; `true` skips logging to inference history — use for health checks),
`store` (default `true`; `false` opts a real request out of history logging), `project_id`
(associate the call with a Pioneer project).

## Retry on cold start

A request against an idle deployment can return:

```json
{"detail": "Inference request timed out waiting for provider capacity. The underlying deployment may be cold-starting or temporarily over-subscribed. Retry after a few seconds (see the Retry-After response header)."}
```

with a `Retry-After` response header. Always retry (respecting `Retry-After`) before surfacing an
error to a caller — this is normal for any model, not just specialty checkpoints, and the model
answers in well under 200ms once warm.

```python
import time
import requests

def infer_with_retry(payload, api_key, max_attempts=5):
    for attempt in range(max_attempts):
        resp = requests.post(
            "https://api.pioneer.ai/inference",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code == 200:
            return resp.json()
        retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
        if attempt == max_attempts - 1:
            resp.raise_for_status()
        time.sleep(retry_after)
```

## OpenAI-compatible surface

For chat-shaped access to the same Pioneer models (mainly relevant for decoder/LLM job IDs, not
GLiNER2 encoder schemas), Pioneer also exposes `base_url="https://api.pioneer.ai/v1"` as a
drop-in OpenAI SDK target:

```python
from openai import OpenAI

client = OpenAI(api_key="pio_sk_...", base_url="https://api.pioneer.ai/v1")
response = client.chat.completions.create(
    model="YOUR_TRAINING_JOB_ID",
    messages=[{"role": "user", "content": "Extract entities from: Apple launched the iPhone."}],
    extra_body={"schema": {"entities": ["organization", "product"]}},
)
```

`schema` is a Pioneer extension passed via `extra_body` (Python SDK) or at the request's top
level (raw HTTP) — not part of the OpenAI spec. `GET /v1/models` lists what you can call this
way. For GLiNER2 encoder schemas, prefer the native `/inference` endpoint above.

## Error handling

| Cause | Response |
|---|---|
| Unknown `model_id` | `400`-class `{"detail": "Model '...' is not a recognised model id. ... To call a fine-tuned model, pass its training-job UUID."}` |
| Missing `X-API-Key` header | `401`, empty body |
| Malformed key (wrong prefix) | `{"detail": "Invalid API key format. API keys must start with 'pio_sk_'. ..."}` |
| Cold start / capacity | See above — retry, don't fail immediately |
| `422` validation error | `HTTPValidationError` shape (FastAPI default) — field-level messages |

## Best practices

- Always send `schema` as a dict and omit `task` entirely.
- Always implement retry-with-backoff on `Retry-After` for `/inference` calls.
- Check `GET /base-models` for the current `model_id` you need rather than assuming a Hugging
  Face repo name maps unchanged.
- Poll `GET /felix/datasets/{name}/{version}` to `"ready"` before referencing a dataset in a
  training job — `upload/process` returns immediately but processing is async.
- Poll `GET /felix/training-jobs/{job_id}` to a terminal status before using its `id` as an
  inference `model_id`.
