# Cloud API Access (`GLiNER2API`)

Use GLiNER2 through a cloud API without loading models locally — no GPU/CPU model download, near-
zero memory footprint. Mirrors
[tutorial/7-api.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/7-api.md).

## Setup

```bash
pip install gliner2
export PIONEER_API_KEY="your-api-key-here"   # get one at https://gliner.pioneer.ai
```

**The SDK's hardcoded default host is dead — always pass `api_base_url` explicitly.**
`GLiNER2API.DEFAULT_BASE_URL` (used whenever `from_api()` is called with no `api_base_url`) is
`https://api.fastino.ai`, not `https://api.pioneer.ai`. That default host is decommissioned
(confirmed: expired TLS cert, and `/felix/training-jobs` there 404s with Vercel's
`DEPLOYMENT_NOT_FOUND`) — `from_api()` with no override will fail outright. The real, live
endpoint is `https://api.pioneer.ai` (verified working, exact match to this file's documented
`/inference` response shape). Always pass it explicitly, or set `GLINER2_API_BASE_URL`:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_api(api_base_url="https://api.pioneer.ai")   # reads PIONEER_API_KEY for the key
# or: extractor = GLiNER2.from_api(api_key="...", api_base_url="https://api.pioneer.ai")

# torch-free alternative for local batch partitioning / long-doc scanning without loading a model
from gliner2 import API, InputExample, Schema, TrainingDataset
client = API()
results = client.batch_extract_entities(documents, ["company", "person"], batch_size=8)
long_result = client.extract_entities_long(annual_report, ["company", "person"], chunk_size=384, chunk_overlap=64, include_spans=True)
```

## Same interface as local models

Every method below mirrors its local-model counterpart exactly — `extract_entities`,
`classify_text`, `extract_json`, `extract_relations`, `create_schema().extract(...)`,
`batch_extract_*`, `include_confidence`, `include_spans`, `threshold` all work identically. See
[entity-extraction.md](entity-extraction.md), [classification.md](classification.md),
[json-extraction.md](json-extraction.md), [relation-extraction.md](relation-extraction.md), and
[combined-schemas.md](combined-schemas.md) for the shared API surface — this file only covers
what differs for the API path.

```python
results = extractor.extract_entities(
    "Elon Musk founded SpaceX in 2002 and Tesla in 2003.",
    ["person", "company", "date"],
)
# {'entities': {'person': ['Elon Musk'], 'company': ['SpaceX', 'Tesla'], 'date': ['2002', '2003']}}
```

## Error handling

```python
from gliner2 import GLiNER2, GLiNER2APIError, AuthenticationError, ValidationError

try:
    extractor = GLiNER2.from_api()
    results = extractor.extract_entities(text, entity_types)
except AuthenticationError:
    print("Invalid API key. Check PIONEER_API_KEY.")
except ValidationError as e:
    print(f"Invalid request: {e}")
except GLiNER2APIError as e:
    print(f"API error: {e}")
```

## Connection settings

```python
extractor = GLiNER2.from_api(api_key="your-key", timeout=60.0, max_retries=5)
```

## Raw results (advanced)

```python
results = extractor.extract_entities(
    text, entity_types,
    format_results=False,       # get raw tuples instead of the formatted dict
    include_confidence=True,
    include_spans=True,
)
# Returns tuples: (text, confidence, start_char, end_char)
```

## API vs local — when to use which

| Feature | API (`from_api()`) | Local (`from_pretrained()`) |
|---|---|---|
| Setup | Just an API key | GPU/CPU + model download |
| Memory | ~0 MB | 2–8 GB+ |
| Latency | Network dependent | Faster for single texts |
| Batch | Optimized | Optimized |
| Cost | Per request | Free after setup |
| Offline | ❌ | ✅ |
| `RegexValidator` | ❌ | ✅ |

**Use the API** for production without a GPU, serverless functions (AWS Lambda, etc.), quick
prototyping, low-memory environments, mobile/edge.

**Use local** for high-volume processing, offline requirements, sensitive data that must not
leave the network, `RegexValidator` support, or cost optimization at scale.

## Limitations (API-only)

1. **No `RegexValidator`** — use a local model for regex-based span filtering.
2. **Multi-schema batch** (different schema per text) works but is slower than a shared schema.
3. **No custom models** — the API serves the default GLiNER2 model only.

## Seamless local ↔ API switching

The API mirrors the local interface exactly, so swapping is a one-line change:

```python
extractor = GLiNER2.from_api()  # development

# production, if you need it:
# from gliner2 import AutoExtractor
# extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")

results = extractor.extract_entities(text, entity_types)  # same code either way
```

## Best practices

1. Store the API key in an environment variable, never hardcoded.
2. Handle `GLiNER2APIError`/`AuthenticationError`/`ValidationError` explicitly — network calls can
   fail.
3. Prefer batch methods over per-text loops.
4. Increase `timeout` for large texts or long-document calls.
5. Cache results client-side to avoid redundant calls on identical content.
