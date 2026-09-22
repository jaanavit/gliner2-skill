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

model.unload_adapter()                    # intended: back to base model, no adapter
model.extract_entities("Some text", ["entity"])

model.has_adapter        # bool: is an adapter currently loaded
model.adapter_config     # loaded adapter's LoRAAdapterConfig (e.g. .lora_r)
```

Adapter directories must contain `adapter_config.json` and `adapter_weights.safetensors`.
Loading a new adapter automatically unloads the previous one — no manual unload step needed
between swaps.

**`unload_adapter()` does not reliably restore base-model behavior — do not depend on it for a
"back to base" state.** [Root cause and fix upstream](https://github.com/fastino-ai/GLiNER2/pull/160):
`load_adapter()` on an `adapter_config.json`+`adapter_weights.safetensors` checkpoint injects LoRA
layers directly into the model object without ever turning that object into a `PeftModel`
instance; `unload_adapter()`'s removal path only fires for an actual `PeftModel`, so on an
unpatched install it silently no-ops — `has_adapter` flips to `False` and looks unloaded, but the
model keeps producing adapter-influenced output indefinitely. **Direct adapter-to-adapter
swapping is not affected** — `load_adapter("./a")` then `load_adapter("./b")` with no
`unload_adapter()` call in between correctly replaces the active adapter; only the explicit
"go back to no adapter" call is broken. If you need genuine base-model behavior after any
adapter has been loaded, reload a fresh `AutoExtractor.from_pretrained(base_model_name)` instead
of calling `unload_adapter()`, until the upstream fix is released.

## Routing by document type

Given the `unload_adapter()` caveat above, route the no-match case to a **separate, never-
adapter-loaded** base model instance instead of calling `unload_adapter()` on the shared one:

```python
def extract_with_routing(model, base_model, text, doc_type, adapters):
    adapter_path = adapters.get(doc_type)
    active = model if adapter_path else base_model
    if adapter_path:
        model.load_adapter(adapter_path)

    entity_types = {
        "legal": ["company", "person", "law"],
        "medical": ["disease", "drug", "symptom"],
        "support": ["order_id", "customer", "issue"],
    }
    return active.extract_entities(text, entity_types.get(doc_type, ["entity"]))

adapters = {"legal": "./legal_adapter", "medical": "./medical_adapter", "support": "./support_adapter"}
base_model = AutoExtractor.from_pretrained("fastino/gliner2-base-v1")  # kept adapter-free
result = extract_with_routing(model, base_model, "Apple sued Google", "legal", adapters)
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
        # Separate, permanently adapter-free instance for the no-match case --
        # see the unload_adapter() caveat above for why this isn't self.model.unload_adapter().
        self.base_model = AutoExtractor.from_pretrained(base_model_name)
        self.adapters = adapters
        self.current_domain = None

    def extract(self, text, domain, entity_types):
        adapter_path = self.adapters.get(domain)
        if not adapter_path:
            return self.base_model.extract_entities(text, entity_types)
        if self.current_domain != domain:
            self.model.load_adapter(adapter_path)
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
- **Don't call `model.unload_adapter()` to reach base-model behavior** — keep a second,
  permanently adapter-free `AutoExtractor` instance for the no-match case instead (see above).
- Check `model.has_adapter` in tests/debugging if predictions look unexpectedly unspecialized —
  but note it reports `False` after `unload_adapter()` even when the adapter is still active on
  an unpatched install (see caveat above), so don't treat it as sufficient proof of a clean state.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Predictions identical with/without adapter | Confirm `model.has_adapter` is `True` after `load_adapter`; verify the adapter path actually contains `adapter_config.json` + weights |
| `load_adapter` path not found | Check the directory exists and matches `LoRAAdapterConfig.is_adapter_path(path)`; list the parent dir for available checkpoints |
| Slow switching in a hot loop | Route/group requests by domain so you're not reloading per-request; see the router pattern above |
| Predictions still adapter-influenced after `unload_adapter()` | Known issue — see the caveat above; use a separate adapter-free model instance instead of `unload_adapter()` |
