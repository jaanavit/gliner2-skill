# Pioneer-Hosted Inference API (`api.pioneer.ai`)

**Pioneer-specific** — this file covers Fastino's own hosted inference platform
(`https://api.pioneer.ai`), not the generic public `gliner2` package. If you're building against
the open-source library with no Pioneer account, use [api-access.md](api-access.md)'s
`GLiNER2API`/`GLiNER2.from_api()` instead — a Python SDK that mirrors the local method interface
but only calls the default model. Use **this** file when you need model selection (a specific
base model, or your own fine-tuned training-job ID), the full combined-schema request shape, or
the OpenAI-compatible surface.

## Setup

```bash
export PIONEER_API_KEY="pio_sk_..."
```

```bash
curl -X POST https://api.pioneer.ai/inference \
  -H "X-API-Key: $PIONEER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "fastino/gliner2-base-v1", "text": "...", "schema": {"entities": [...]}}'
```

## Picking a model

```bash
curl "https://api.pioneer.ai/base-models?supports_inference=true&task_type=encoder" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

`model_id` is either a base model from that catalog or a training-job UUID from
`POST /felix/training-jobs` to call your own fine-tune. Current encoder catalog:

**`/felix/training-jobs` is real but unsupported by this skill/SDK — no client code wraps it,
and part of its input is undocumented.** There's no `gliner2` SDK method for it (confirmed —
nothing named `felix`/`training_job` exists anywhere in the installed package); you must call it
with raw HTTP. Reverse-engineered from a `POST` with an empty body and from a `GET` job listing,
using a real key against `https://api.pioneer.ai`:

```bash
curl -X POST https://api.pioneer.ai/felix/training-jobs \
  -H "X-API-Key: $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-classifier-v1",
    "base_model": "fastino/gliner2-base-v1",
    "datasets": [{"name": "my-dataset", "version": "1"}],
    "nr_epochs": 10,
    "learning_rate": 0.0001,
    "batch_size": 8,
    "validation_data_percentage": 0.2
  }'
```

- **Required** (an empty-body `POST` returns `422` naming these explicitly): `model_name`,
  `base_model`, `datasets`.
- **Observed optional fields** (from an existing job's shape, not confirmed exhaustive):
  `validation_data_percentage`, `nr_epochs`, `learning_rate`, `batch_size`, `seed`,
  `instance_type`, `task_type`, `training_type`, `project_id`, and (for classification jobs)
  `labels`.
- **`datasets` takes `{name, version}` references, not raw text/files** — your data must
  already exist as a registered dataset in Pioneer first. **How a dataset gets created/uploaded
  is not documented anywhere in this skill or the SDK** — this is the actual blocker to using
  this endpoint, not the training-job call itself.
- `GET /felix/training-jobs` lists your jobs with `status`/`normalized_status`/
  `is_terminal_status`, and `trained_model_path` + `job_reference` once complete.
- Once a job completes, its `id` is exactly the UUID to pass as `model_id` at `/inference` above.

Given the missing dataset-registration step and zero SDK support, prefer the fully-documented,
SDK-native local path in [training.md](training.md) for fine-tuning; treat this endpoint as a
known gap, not a supported workflow, until the dataset-upload step is found and documented.

| `model_id` | Notes |
|---|---|
| `fastino/gliner2-base-v1` | Legacy span, English |
| `fastino/gliner2-large-v1` | Legacy span, higher accuracy |
| `fastino/gliner2-multi-v1` | Legacy span, multilingual |
| `fastino/gliner2-multi-large-v1` | Legacy span, multilingual, larger |
| `fastino/gliguard-LLMGuardrails-300M` | LLM safety / guardrails moderation |
| `fastino/gliner2-privacy-filter-PII-multi` | PII detection |
| `fastino/gliguard-PII-multi` | Combined guardrails + PII checkpoint |

GLiNER2.5 boundary-only features (span attributes, constrained classification, joint IE, record
mode) are not exposed through a base `model_id` here — those need the local `gliner2[local]` +
`AutoExtractor` path documented in the rest of this skill. Always confirm the `model_id` you need
against `GET /base-models` rather than assuming a Hugging Face repo name works unchanged.

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

`schema` is always a dict (never a flat list of label strings), and takes any combination of
these keys in one request:

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

## Error handling

| Cause | Response |
|---|---|
| Unknown `model_id` | `400`-class `{"detail": "Model '...' is not a recognised model id. ... To call a fine-tuned model, pass its training-job UUID."}` |
| Missing `X-API-Key` header | `401`, empty body |
| Malformed key (wrong prefix) | `{"detail": "Invalid API key format. API keys must start with 'pio_sk_'. ..."}` |
| Cold start / capacity | See above — retry, don't fail immediately |
| `422` validation error | `HTTPValidationError` shape (FastAPI default) — field-level messages |

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
way. For GLiNER2 encoder schemas, prefer the native `/inference` endpoint above — it's the more
direct path and this skill's examples are built around it.

## Best practices

- Always send `schema` as a dict and omit `task` entirely.
- Always implement retry-with-backoff on `Retry-After` for `/inference` calls.
- Check `GET /base-models` for the current `model_id` you need rather than assuming a Hugging
  Face repo name maps unchanged.
- Don't assume GLiNER2.5-only capabilities (attributes, constrained classification, joint IE,
  records) work against a base `model_id` here — the local `gliner2[local]` path is the way to
  get those.
- For local development or when you don't need model selection, `api-access.md`'s
  `GLiNER2.from_api()` is a lighter-weight Python-native alternative to raw HTTP calls here.
