# GLiNER (v1): Production Serving (Ray Serve)

Production HTTP deployment via Ray Serve — dynamic batching, memory-aware batch sizing,
precompiled power-of-two batch sizes, multi-replica scaling. Source: `gliner/serve/`. Condensed
from [Serving](https://urchade.github.io/GLiNER/serving.html).

```bash
uv pip install gliner[serve]        # or: pip install gliner ray[serve]
```

## Start a server and predict

```bash
python -m gliner.serve --model urchade/gliner_small-v2.1
```

```python
from gliner.serve import GLiNERClient

client = GLiNERClient()   # defaults to http://localhost:8000/gliner
result = client.predict("John works at Google in Mountain View", labels=["person", "organization", "location"])
# {'entities': [{'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.95}, ...]}
```

`GLiNERClient` is a pure-stdlib HTTP client — it does not import `ray` and can run from any Python
process, including ones without `ray` installed:

```python
client = GLiNERClient(
    base_url="http://gliner.internal:8000", route_prefix="/gliner", timeout=30.0, max_concurrency=32,
)
```

Passing a **list** of texts dispatches each as its own concurrent HTTP request so the server's
`@serve.batch` can coalesce them into one forward pass:

```python
outputs = client.predict(["John works at Google", "Paris is in France"], labels=["person", "organization", "location"])
# list[dict], one per input text
```

Per-text label descriptions/sets follow the same shape as local inference (see
[gliner1-usage.md](gliner1-usage.md)'s "Label descriptions"):

```python
outputs = client.predict(
    ["John works at Google", "Paris is in France"],
    labels=[
        {"person": "A human individual", "organization": "A company or institution"},
        {"location": "A geographical place"},
    ],
)
```

Errors surface as `gliner.serve.client.GLiNERClientError`.

> **Truncation warning**: the response has no truncation flag. `--max-model-len` overrides the
> loaded checkpoint's `config.max_len`; over-limit requests can still return HTTP success after the
> server keeps only the prefix (see [gliner1-usage.md](gliner1-usage.md)'s truncation section for
> the underlying behavior and a public preflight check).

Raw HTTP: `POST /gliner` with JSON body `{"text": "...", "labels": [...]}`.

## CLI options (selected)

```bash
python -m gliner.serve --model urchade/gliner_small-v2.1 --device cuda --dtype bfloat16
python -m gliner.serve --model urchade/gliner_small-v2.1 --enable-flashdeberta --enable-sequence-packing --max-batch-size 64
python -m gliner.serve --model urchade/gliner_small-v2.1 --num-replicas 4 --num-gpus-per-replica 1
python -m gliner.serve --model urchade/gliner_small-v2.1 --enable-polylora --polylora-max-gpu-adapters 8 \
    --polylora-disk-cache-dir /models/polylora-cache
```

| Group | Key flags |
|---|---|
| Model | `--model` (required), `--device` (cuda\|cpu), `--dtype` (float32\|float16\|bfloat16), `--quantization int8` |
| Limits | `--max-model-len` (2048), `--max-span-width` (12), `--max-labels` (-1 = unlimited) |
| Thresholds | `--default-threshold` (0.5), `--default-relation-threshold` (0.5) |
| Replicas | `--num-replicas` (1), `--num-gpus-per-replica` (1.0), `--num-cpus-per-replica` (1.0) |
| Batching | `--max-batch-size` (32), `--batch-wait-timeout-ms` (10.0), `--request-timeout-s` (30.0), `--max-ongoing-requests` (256), `--queue-capacity` (4096), `--precompiled-batch-sizes` (1,2,4,8,16,32) |
| Server | `--route-prefix` (/gliner), `--port` (8000), `--ray-address` |
| Perf | `--tokenizer-threads`/`--decoding-threads` (4), `--no-compile`, `--enable-sequence-packing`, `--enable-flashdeberta`, `--warmup-iterations` (3) |
| Memory | `--target-memory-fraction` (0.9), `--memory-overhead-factor` (1.3) |
| PolyLoRA | `--enable-polylora`, `--polylora-max-rank` (16), `--polylora-max-gpu-adapters` (8), `--polylora-max-cpu-adapters` (128), `--polylora-disk-cache-dir`, `--polylora-adapter-id-pattern` |

## Programmatic usage

```python
from gliner.serve import GLiNERFactory, GLiNERServeConfig

config = GLiNERServeConfig(model="urchade/gliner_small-v2.1", device="cuda", dtype="bfloat16", max_batch_size=32)
llm = GLiNERFactory(config=config)
try:
    result = llm.predict("John works at Google", ["person", "organization"])
finally:
    llm.shutdown()
```

Low-level handle (returns Ray `ObjectRef`s):

```python
from gliner.serve import GLiNERServeConfig, serve
handle = serve(GLiNERServeConfig(model="urchade/gliner_small-v2.1"))
result = handle.predict.remote("John works at Google", ["person", "organization"]).result()
```

## PolyLoRA (multi-adapter serving)

```bash
pip install polylora
```

One deployment routes requests through different LoRA adapters without a replica per adapter.
Requires GLiNER **text encoder** models — wraps
`model.model.token_rep_layer.bert_layer.model`; other architectures raise `NotImplementedError`.

```python
from gliner.serve import GLiNERClient
client = GLiNERClient()
result = client.predict("John works at Google", labels=["person", "organization"], adapter_id="customer-a")
```

Omitting `adapter_id` uses `polylora_base_adapter_id` (default `"__base__"` = base-model
inference). Unknown adapter ids → HTTP 404. Cache introspection:
`client.adapter_cache_status()` / `client.is_adapter_cached("customer-a")`, or
`GET /gliner/adapter-cache?adapter_id=...`.

## Relation extraction serving

Auto-detected from `model.config.model_type` containing `"relex"` — no extra flag needed:

```bash
python -m gliner.serve --model knowledgator/gliner-relex-large-v1.0 --dtype bfloat16 --max-batch-size 16
```

```python
result = client.predict(
    "Bill Gates founded Microsoft in 1975. The company is headquartered in Redmond.",
    labels=["person", "organization", "date", "location"],
    relations=["founded", "founded_in", "headquartered_in"],
    threshold=0.5, relation_threshold=0.5,
)
# {"entities": [...], "relations": [{"relation", "score", "head": {"entity_idx"}, "tail": {"entity_idx"}}, ...]}
```

NER-only models omit `"relations"` from the response; passing `relations=` to one is a no-op.
`GLiNERFactory`/in-process usage and HTTP curl follow the same shape as ordinary NER serving above.

## Docker

```bash
docker build -t gliner-serve -f gliner/serve/Containerfile .
docker run --gpus all -p 8000:8000 \
    -e GLINER_MODEL=urchade/gliner_medium-v2.1 -e GLINER_ENABLE_FLASHDEBERTA=true gliner-serve
```

| Env var | Default | |
|---|---|---|
| `GLINER_MODEL` | `urchade/gliner_small-v2.1` | |
| `GLINER_DEVICE` | `cuda` | |
| `GLINER_DTYPE` | `bfloat16` | |
| `GLINER_MAX_BATCH_SIZE` | `32` | |
| `GLINER_NUM_REPLICAS` | `1` | |
| `GLINER_MEMORY_FRACTION` | `0.8` | |
| `GLINER_QUANTIZATION` | — | `int8` only |
| `GLINER_ENABLE_FLASHDEBERTA` | `false` | |
| `GLINER_ENABLE_PACKING` | `false` | |
| `GLINER_DISABLE_COMPILE` | `false` | |
| `GLINER_ROUTE_PREFIX` | `/gliner` | |

## Shutdown

```python
from gliner.serve import shutdown
shutdown()
```
