# Pioneer-Hosted Training & Inference API (`api.pioneer.ai`)

Fastino's hosted platform can upload a dataset, train a GLiNER model, and serve it through the
OpenAI-compatible chat-completions API. This reference follows the current documentation index at
[`https://docs.fastino.ai/llms.txt`](https://docs.fastino.ai/llms.txt) and the classification,
NER, and extraction guides linked from it.

## Setup

Sign up or log in at [pioneer.ai](https://pioneer.ai), create an API key under
Settings → API Keys, and export it without committing it:

```bash
export PIONEER_API_KEY="pio_sk_..."
```

## Pick a base model

Always query the live catalog before choosing a model:

```bash
curl "https://api.pioneer.ai/base-models?task_type=encoder&supports_training=true" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

Common training targets:

| Model | Use |
|---|---|
| `fastino/gliner2-base-v1` | English, general purpose |
| `fastino/gliner2-large-v1` | English, higher accuracy |
| `fastino/gliner2-multi-v1` | Multilingual |
| `fastino/gliner2-multi-large-v1` | Multilingual, higher accuracy |

GLiNER2.5 boundary checkpoints are not currently hosted training targets. Do not assume a local
Hugging Face checkpoint name is available on Pioneer; trust the live catalog.

## Upload a dataset

Request a presigned URL, upload the file, then start processing:

```bash
curl -X POST https://api.pioneer.ai/felix/datasets/upload/url \
  -H "X-API-Key: $PIONEER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "sentiment-ds",
    "dataset_type": "classification",
    "format": "jsonl",
    "filename": "sentiment.jsonl"
  }'

curl -X PUT "$PRESIGNED_URL" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @sentiment.jsonl

curl -X POST https://api.pioneer.ai/felix/datasets/upload/process \
  -H "X-API-Key: $PIONEER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "DATASET_ID"}'
```

Poll `GET /felix/datasets/{name}/{version}` until `status` is `ready`. Stop and inspect
`processing_error` if it becomes `failed`.

Pioneer's hosted row format differs from the local `training-data-format.md` format:

```json
{"text": "I love this product!", "label": "positive"}
```

Single-label classification uses `label`; multi-label classification uses `labels`:

```json
{"text": "Fast delivery and great quality.", "labels": ["shipping", "quality"]}
```

NER uses flat `[span, label]` pairs:

```json
{"text": "John works at OpenAI.", "entities": [["John", "person"], ["OpenAI", "organization"]]}
```

Use one consistent row shape per dataset. A hosted classification dataset represents one
classification target; separate independent targets such as queue, priority, and ticket type
into separate datasets and training jobs.

## Start training

```bash
curl -X POST https://api.pioneer.ai/felix/training-jobs \
  -H "X-API-Key: $PIONEER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-classifier-v1",
    "base_model": "fastino/gliner2-base-v1",
    "datasets": [{"name": "sentiment-ds", "version": "1"}],
    "training_type": "lora",
    "nr_epochs": 5,
    "learning_rate": 0.00005,
    "validation_data_percentage": 0.2
  }'
```

Required fields are `model_name`, `base_model`, and `datasets`. `training_type` accepts
`"lora"` or `"full"`. Common optional fields include `nr_epochs`, `learning_rate`, `batch_size`,
`validation_data_percentage`, `early_stopping_patience`, `warmup_ratio`, and `project_id`.
Task type and labels are inferred from the ready dataset.

Do not send `seed` on an automatically routed encoder job. The API can reject it with `422`
unless the request explicitly selects a compatible provider route.

The response returns the training-job UUID in `id`.

## Monitor training

```bash
curl "https://api.pioneer.ai/felix/training-jobs/JOB_ID" \
  -H "X-API-Key: $PIONEER_API_KEY"

curl "https://api.pioneer.ai/felix/training-jobs/JOB_ID/logs" \
  -H "X-API-Key: $PIONEER_API_KEY"
```

Poll until `is_terminal_status` is true. A successful encoder job commonly reports:

- `normalized_status: "complete"`
- `artifact_ready: true`
- `provider_ready: true`
- `is_deployable: true`
- `provider_deployments.modal.status: "active"`

Use the job UUID as `model` in the chat-completions request below. No separate deployment call
is required for the job-scoped endpoint.

Both LoRA and full encoder fine-tuning have been verified end to end with this flow.

## Run inference

Use the current OpenAI-compatible endpoint:

```bash
curl -X POST https://api.pioneer.ai/v1/chat/completions \
  -H "X-API-Key: $PIONEER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "JOB_ID_OR_BASE_MODEL",
    "messages": [
      {
        "role": "user",
        "content": "The service has been unavailable since this morning."
      }
    ],
    "schema": {
      "classifications": [
        {
          "task": "ticket_type",
          "labels": ["Incident", "Request", "Problem", "Change"],
          "multi_label": false,
          "top_k": 1
        }
      ]
    }
  }'
```

For a base model, set `model` to a live catalog ID. For a trained model, set it to the completed
training job's UUID.

The assistant message's `content` is a JSON string:

```json
{
  "ticket_type": {
    "label": "Incident",
    "confidence": 0.5853
  }
}
```

Parse `choices[0].message.content` as JSON before reading task results.

The same endpoint accepts the other encoder schemas:

```json
{
  "entities": ["organization", "product", "location"],
  "classifications": [
    {
      "task": "sentiment",
      "labels": ["positive", "negative", "neutral"],
      "multi_label": false,
      "top_k": 1
    }
  ],
  "relations": ["works_for", "lives_in"],
  "structures": {
    "product": ["name::str", "price::str", "features::list"]
  }
}
```

`schema` is a top-level Pioneer extension. With the OpenAI SDK, pass it through `extra_body`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="pio_sk_...",
    base_url="https://api.pioneer.ai/v1",
)

response = client.chat.completions.create(
    model="JOB_ID_OR_BASE_MODEL",
    messages=[{"role": "user", "content": "Apple launched the iPhone."}],
    extra_body={"schema": {"entities": ["organization", "product"]}},
)
```

## Handle cold starts

A newly deployed or idle encoder can take longer than two minutes to answer its first request.
Use a read timeout of at least 300 seconds and retry a timeout or `503` response. A request that
times out can still warm the deployment; the next request may succeed quickly.

```python
import time
import requests


def infer_with_retry(payload, api_key, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                "https://api.pioneer.ai/v1/chat/completions",
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
        except requests.ReadTimeout:
            if attempt == max_attempts - 1:
                raise
            continue

        if response.ok:
            return response.json()
        if response.status_code != 503 or attempt == max_attempts - 1:
            response.raise_for_status()
        time.sleep(int(response.headers.get("Retry-After", 2 ** attempt)))
```

## Best practices

- Check the live base-model catalog before creating a job.
- Keep API keys in environment variables or a secret manager.
- Poll dataset processing to `ready` before training.
- Poll the job to terminal status and confirm `is_deployable`, `artifact_ready`, and
  `provider_ready` before inference.
- Use `/v1/chat/completions`, put the model identifier in `model`, and put input text inside
  `messages`.
- Retry long cold starts with a 300-second read timeout.
- Evaluate on a held-out dataset; training and validation loss alone do not establish model
  quality.
