# LoRA Adapter Switching / Routing

Swap between already-trained domain adapters at inference time without reloading the base model.
Mirrors
[tutorial/11-adapter_switching.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/11-adapter_switching.md).
For training adapters in the first place, see [lora-adapters.md](lora-adapters.md).

## Quick reference

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")   # load base once

model.load_adapter("./legal_adapter")     # load an adapter (auto-unloads any previous one)
model.extract_entities("Apple sued Google", ["company"])

model.load_adapter("./medical_adapter")   # swap domains in <1s
model.extract_entities("Patient has diabetes", ["disease"])

model.unload_adapter()                    # back to base model, no adapter
model.extract_entities("Some text", ["entity"])

model.has_adapter        # bool: is an adapter currently loaded
model.adapter_config     # loaded adapter's LoRAAdapterConfig (e.g. .lora_r)
```

Adapter directories must contain `adapter_config.json` and `adapter_weights.safetensors`.
Loading a new adapter automatically unloads the previous one — no manual unload step needed
between swaps.

## Routing by document type

```python
def extract_with_routing(model, text, doc_type, adapters):
    adapter_path = adapters.get(doc_type)
    model.load_adapter(adapter_path) if adapter_path else model.unload_adapter()

    entity_types = {
        "legal": ["company", "person", "law"],
        "medical": ["disease", "drug", "symptom"],
        "support": ["order_id", "customer", "issue"],
    }
    return model.extract_entities(text, entity_types.get(doc_type, ["entity"]))

adapters = {"legal": "./legal_adapter", "medical": "./medical_adapter", "support": "./support_adapter"}
result = extract_with_routing(model, "Apple sued Google", "legal", adapters)
```

## Batch processing grouped by domain

Load each adapter once and process all documents for that domain before switching — avoids
redundant swaps:

```python
def process_by_domain(model, documents_by_domain, adapters):
    results = {}
    for domain, docs in documents_by_domain.items():
        model.load_adapter(adapters[domain])
        results[domain] = [model.extract_entities(doc, entity_types_for(domain)) for doc in docs]
    return results
```

## Reusable router class

```python
class AdapterRouter:
    def __init__(self, base_model_name, adapters):
        self.model = AutoExtractor.from_pretrained(base_model_name)
        self.adapters = adapters
        self.current_domain = None

    def extract(self, text, domain, entity_types):
        if self.current_domain != domain:
            adapter_path = self.adapters.get(domain)
            self.model.load_adapter(adapter_path) if adapter_path else self.model.unload_adapter()
            self.current_domain = domain
        return self.model.extract_entities(text, entity_types)

router = AdapterRouter("fastino/gliner2-base-v1", {"legal": "./legal_adapter", "medical": "./medical_adapter"})
router.extract("Apple sued Google", "legal", ["company"])
```

This is the pattern to reach for in a **multi-tenant service**: key `adapters` by tenant ID and
only reload when `current_tenant` changes, so a hot loop of same-tenant requests pays no swap
cost.

## Best practices

- Keep a `current_domain`/`current_tenant` guard so you don't reload the same adapter twice in a
  row (swap cost is small but not free).
- Group batch work by domain rather than interleaving to minimize the number of swaps.
- Call `model.unload_adapter()` explicitly when you need base-model behavior — it is not implicit
  just because no adapter matched.
- Check `model.has_adapter` in tests/debugging if predictions look unexpectedly unspecialized.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Predictions identical with/without adapter | Confirm `model.has_adapter` is `True` after `load_adapter`; verify the adapter path actually contains `adapter_config.json` + weights |
| `load_adapter` path not found | Check the directory exists and matches `LoRAAdapterConfig.is_adapter_path(path)`; list the parent dir for available checkpoints |
| Slow switching in a hot loop | Route/group requests by domain so you're not reloading per-request; see the router pattern above |
