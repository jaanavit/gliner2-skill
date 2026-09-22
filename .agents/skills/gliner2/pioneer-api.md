# Pioneer-Hosted Training & Inference API (`api.pioneer.ai`)

Fastino's hosted platform at `https://api.pioneer.ai`. This file covers the readiness gate,
model selection, inference, training, evaluation, and provenance.

## Fastino readiness gate

Run this gate only after the user chooses Fastino. If the account is already authenticated,
funded, and has a working key, skip directly to model discovery.

1. Sign up or log in at
   [agent.pioneer.ai/auth](https://agent.pioneer.ai/auth) (Google, GitHub, magic link, or email).
2. Create a key under Settings → API Keys and expose it to the current shell:

```bash
export PIONEER_API_KEY="pio_sk_..."
```

3. Verify authentication and inspect the available balance with an authenticated endpoint:

```bash
curl "https://api.pioneer.ai/billing/ledger/balance" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

   A `401` means authenticate or replace the key. Do not use `GET /base-models` to verify a key:
   the catalog is public and invalid credentials are treated as anonymous access.
4. If funding is insufficient, guide the user to
   [agent.pioneer.ai/billing](https://agent.pioneer.ai/billing) to add credits or adjust their
   spend limit. Payment details remain user-entered; the agent should guide and then verify,
   never handle card data.
5. Discover an inference-capable encoder through `GET /base-models`.

## Picking a base model

Query the capability required by the selected path:

```bash
# Hosted inference
curl "https://api.pioneer.ai/base-models?supports_inference=true&task_type=encoder" \
  -H "X-API-Key: $PIONEER_API_KEY"

# Hosted training
curl "https://api.pioneer.ai/base-models?supports_training=true&task_type=encoder" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

| `model_id` | Notes |
|---|---|
| `fastino/gliner2.5-multi-v1` | GLiNER2.5 boundary, multilingual, 4,096-token context; hosted inference only |
| `fastino/gliner2-base-v1` | Legacy span, English |
| `fastino/gliner2-large-v1` | Legacy span, higher accuracy |
| `fastino/gliner2-multi-v1` | Legacy span, multilingual |
| `fastino/gliner2-multi-large-v1` | Legacy span, multilingual, larger |
| `fastino/gliguard-LLMGuardrails-300M` | LLM safety / guardrails moderation |
| `fastino/gliner2-privacy-filter-PII-multi` | PII detection |
| `fastino/gliguard-PII-multi` | Combined guardrails + PII checkpoint |

`fastino/gliner2.5-multi-v1` is the hosted GLiNER2.5 option. It supports base-model inference
but not Fastino training; the GLiNER2.5 base and small checkpoints remain local-only unless they
appear in the live catalog later. Always confirm both `supports_inference` and
`supports_training` through `GET /base-models` rather than assuming every Hugging Face
checkpoint supports both.

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

Multi-label classification uses `labels`:

```json
{"text": "Fast delivery and great quality.", "labels": ["shipping", "quality"]}
```

```json
// dataset_type: "ner" -- flat [span, label] pairs, not the local {"entities": {label: [spans]}} shape
{"text": "John Smith works at OpenAI in San Francisco.", "entities": [["John Smith", "person"], ["OpenAI", "organization"], ["San Francisco", "location"]]}
```

A row that doesn't match the expected shape for `dataset_type` fails dataset processing with a
`processing_error` naming the exact columns expected — check that field if `status` comes back
`"failed"` instead of `"ready"`. Use one consistent row shape per dataset. A hosted
classification dataset represents one classification target; use separate datasets and
training jobs for independent targets such as queue, priority, and ticket type.

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
`validation_data_percentage`, `early_stopping_patience`, `warmup_ratio`, and `project_id`.
`task_type` and `labels` are inferred from the dataset, not passed in the request.

Do not send `seed` on an automatically routed encoder job. The API can reject it with `422`
unless the request explicitly selects a compatible provider route.

The response includes an `id` (job UUID) immediately — training runs asynchronously from there.

## Monitoring a job

```bash
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}" -H "X-API-Key: $PIONEER_API_KEY"
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}/logs" -H "X-API-Key: $PIONEER_API_KEY"
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}/checkpoints" -H "X-API-Key: $PIONEER_API_KEY"
curl "https://api.pioneer.ai/felix/training-jobs/{job_id}/billing" -H "X-API-Key: $PIONEER_API_KEY"
```

`status`/`normalized_status`/`is_terminal_status` track progress. On success,
`is_deployable` flips `true`. **The job's `id` is the UUID to pass as `model` to
`/v1/chat/completions` below.** A locally trained checkpoint cannot be uploaded for
Fastino-hosted inference; hosted fine-tuned inference requires a job trained through Fastino.

Use `/logs` for stdout/stderr diagnostics, `/checkpoints` for checkpoint/metric progress, and
`/billing` for cost. Training-job controls include:

- `POST /felix/training-jobs/{job_id}/stop` — stop while preserving checkpoints.
- `POST /felix/training-jobs/{job_id}/terminate` — irreversibly stop and delete artifacts.
- `DELETE /felix/training-jobs/{job_id}` — delete the job record.

Ask for confirmation before stop, delete, or terminate; call out that terminate destroys
artifacts. Multiple jobs may run in parallel and `project_id` associates jobs with separate
projects, subject to the service's active-job and billing limits.

## Evaluate a Fastino training job

Use project-scoped Evaluation Suites, not the retired `/felix/evaluations` routes:

1. `POST /projects/{project_id}/evaluation-suites` — create a reusable held-out suite.
2. `POST /projects/{project_id}/evaluation-suites/{suite_id}/runs` — run the base model and
   completed training-job UUIDs on the same cases.
3. `GET /projects/{project_id}/evaluation-runs/{run_id}` — poll the run.
4. `GET /projects/{project_id}/evaluation-runs/{run_id}/results` — inspect aggregate and
   per-case results.

Keep the cases, schema, and thresholds fixed between base and fine-tuned runs. Do not attempt
hosted inference until training is terminal-successful and `is_deployable` is `true`.

## Request shape

```bash
curl -X POST https://api.pioneer.ai/v1/chat/completions \
  -H "Authorization: Bearer $PIONEER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "fastino/gliner2.5-multi-v1",
    "messages": [{"role": "user", "content": "Apple announced the MacBook Pro at WWDC in Cupertino."}],
    "schema": {"entities": [
      {"name": "organization", "description": "business or institution name"},
      {"name": "product", "description": "named commercial product"},
      {"name": "event", "description": "named conference or event"},
      {"name": "location", "description": "city, region, or place"}
    ]},
    "threshold": 0.5
  }'
# choices[0].message.content is a JSON string:
# {"entities":{"organization":[...],"product":[...],"event":[...],"location":[...]}}
# x_pioneer.inference_id identifies the persisted inference when store=true.
```

`model` is either an inference-capable catalog model returned by `GET /base-models` or a
Fastino training-job UUID from above. `schema` is always a dict (never the deprecated flat list
of entity labels), and takes any combination of these keys in one request:

```json
{
  "entities": [
    {"name": "organization", "description": "business or institution name"},
    {"name": "product", "description": "named commercial product"}
  ],
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

## Output shape and repeated inputs

The endpoint returns an OpenAI chat-completion envelope. Parse
`choices[0].message.content` as JSON to get the GLiNER2 result; request metadata such as the
persisted inference ID is under `x_pioneer`. Unlike the removed native endpoint,
`/v1/chat/completions` accepts one conversation per request — send several requests concurrently
when processing a batch rather than putting a list of texts in one message.

`include_confidence` and `include_spans` both default to `true` and collapse the output when set
`false`:

```bash
-d '{"model": "fastino/gliner2-base-v1", "messages": [{"role": "user", "content": "..."}], "schema": {"entities": [{"name": "organization"}]}, "include_confidence": false, "include_spans": false}'
# entities become plain strings: {"organization": ["Apple"], "product": ["MacBook Pro"]}
```

`store` defaults to `true`; set it to `false` to opt the request out of inference-history
storage. Do not send fields from the removed native contract such as `model_id`, `text`,
`format_results`, or `is_warmup`.

## Retry on cold start

A request against an idle deployment can return:

```json
{"detail": "Inference request timed out waiting for provider capacity. The underlying deployment may be cold-starting or temporarily over-subscribed. Retry after a few seconds (see the Retry-After response header)."}
```

with a `Retry-After` response header. Always retry (respecting `Retry-After`) before surfacing an
error to a caller. Fine-tuned deployments may return `425` while the model is warming; treat
that as retryable alongside `429` and `503`. Do not retry authentication, validation, billing,
or unknown-model errors.

```python
import time

import requests


def infer_with_retry(payload, api_key, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                "https://api.pioneer.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
        except requests.ReadTimeout:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2**attempt)
            continue

        if response.ok:
            return response.json()

        if response.status_code not in {425, 429, 503} or attempt == max_attempts - 1:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        time.sleep(int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt)
```

## OpenAI SDK

The OpenAI-compatible route is the primary hosted inference surface for GLiNER2 base models and
Fastino-trained job IDs:

```python
import json

from openai import OpenAI

client = OpenAI(api_key="pio_sk_...", base_url="https://api.pioneer.ai/v1")
response = client.chat.completions.create(
    model="fastino/gliner2.5-multi-v1",  # or a completed Fastino training-job UUID
    messages=[{"role": "user", "content": "Extract entities from: Apple launched the iPhone."}],
    extra_body={
        "schema": {
            "entities": [
                {"name": "organization", "description": "business or institution name"},
                {"name": "product", "description": "named commercial product"},
            ]
        }
    },
)

result = json.loads(response.choices[0].message.content)
```

`schema` is a Pioneer extension passed via `extra_body` (Python SDK) or at the request's top
level (raw HTTP) — not part of the OpenAI spec. `GET /v1/models` lists models visible through
the compatibility surface; `GET /base-models?supports_inference=true&task_type=encoder` is the
authoritative filtered catalog for hosted GLiNER2 base inference.

## Error handling

| Cause | Response |
|---|---|
| Unknown `model` | OpenAI-compatible `400` error; use a catalog ID or completed Fastino training-job UUID |
| Missing API key (`Authorization: Bearer` or `X-API-Key`) | `401` OpenAI-compatible error |
| Malformed key (wrong prefix) | `{"detail": "Invalid API key format. API keys must start with 'pio_sk_'. ..."}` |
| Insufficient credits / spend limit | Billing `402`; guide the user to billing and verify funding before retrying |
| Failed, cancelled, incomplete, or non-deployable job ID | Do not infer; inspect job status, `error_message`, and `deployability_reason` |
| Cold start / capacity | See above — retry, don't fail immediately |
| `422` validation error | `HTTPValidationError` shape (FastAPI default) — field-level messages |

## Best practices

- Always send `schema` as a dict and omit `task` entirely.
- Always implement retry-with-backoff on `Retry-After` for `/v1/chat/completions` calls.
- Check `GET /base-models` for the current `model` you need rather than assuming a Hugging
  Face repo name maps unchanged.
- Poll `GET /felix/datasets/{name}/{version}` to `"ready"` before referencing a dataset in a
  training job — `upload/process` returns immediately but processing is async.
- Poll `GET /felix/training-jobs/{job_id}` to a terminal status before using its `id` as an
  inference `model`.
- Do not offer Fastino-hosted inference for a checkpoint trained locally; keep it local or train
  through Fastino to obtain a hosted job ID.
- Use project Evaluation Suites for hosted comparisons; `/felix/evaluations` is retired.
