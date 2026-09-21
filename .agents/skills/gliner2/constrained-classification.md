# Constrained Classification (GLiNER2.5 only)

Classify with **hard cross-task constraints** — when one label legally implies, forbids, or caps
another (intents vs. effects, topic vs. audience, severity vs. action, policy/compliance
taxonomies). `classify_text()` scores each task independently; `Classifier` scores the same
encoder then decodes a **globally consistent** assignment. Mirrors
[tutorial/14-constrained_classification.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/14-constrained_classification.md).

## Setup

```python
from gliner2.classification import Classifier, ClassificationSchema, ClassificationConfig
from gliner2.classification import constraints as C
from gliner2.classification.errors import InfeasibleError, SchemaError

clf = Classifier.from_pretrained("fastino/gliner2.5-multi-v1")
```

`Classifier.from_pretrained` loads through `AutoExtractor`. Prediction knobs go in
`ClassificationConfig` per call, not in `from_pretrained`.

## When to use which

| Use `classify_text` | Use `Classifier` |
|---|---|
| One task, or several independent tasks | Labels on task A legally constrain task B |
| Only need argmax / threshold | Need a feasible joint assignment |
| Quick prototypes | Policies, tool routing, compliance taxonomies |

## Declaring tasks

`ClassificationSchema` is a mutable builder; declare every task **before** any constraint that
references it.

```python
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])                              # exactly one label
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)  # zero or more
    .ordinal("severity", ["low", "medium", "high", "critical"])                  # ordered exclusive scale
)
```

| Builder | Meaning | Defaults |
|---|---|---|
| `.single(name, labels)` | Exactly one label | `min_labels=1`, `max_labels=1` |
| `.multi(name, labels, min_labels=0)` | Zero or more labels | Unbounded max unless `max_labels` set |
| `.ordinal(name, labels)` | Ordered exclusive scale | Same cardinality as single; `ordered=True` |

Labels can be a list or a `{label: description}` mapping. Optional task kwargs: `threshold`,
`instruction`, `examples`, `activation` (`"auto"|"sigmoid"|"softmax"`), `temperature`, `default`.

## Constraint DSL

```python
schema.constrain(
    C.implies(("intent", "delete"), ("effects", "delete")),
    C.excludes(("intent", "read"), ("effects", "delete")),
    C.at_most("effects", 2),
)
```

| Boolean | Meaning |
|---|---|
| `C.implies(A, B)` | If A then B |
| `C.iff(A, B)` | A iff B |
| `C.excludes(A, B)` | A and B cannot both hold |
| `C.not_(A)` | A must not hold |
| `C.all_of(...)` / `C.any_of(...)` / `C.exactly_one_of(...)` | Conjunction / disjunction / exactly one |

| Cardinality (any task) | Meaning |
|---|---|
| `C.at_least(task, k)` / `C.at_most(task, k)` / `C.exactly(task, k)` | Label-count bounds |

| Ordinal (requires `.ordinal`) | Meaning |
|---|---|
| `C.at_level` / `C.min_level` / `C.max_level` / `C.between_level(task, lo, hi)` | Level constraints |

| Selection | Meaning |
|---|---|
| `C.any_selected(task)` / `C.any_other_selected(task)` | At least one label on / some other label on |

Constraints validate against tasks declared so far — referencing an undeclared task raises
`SchemaError`.

## End-to-end example

```python
schema = (
    ClassificationSchema()
    .single("intent", ["read", "write", "delete"])
    .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1, max_labels=2)
    .constrain(
        C.implies(("intent", "delete"), ("effects", "delete")),
        C.implies(("intent", "read"), ("effects", "read_only")),
        C.excludes(("intent", "read"), ("effects", "delete")),
        C.excludes(("intent", "read"), ("effects", "modify")),
    )
)

result = clf.classify("Delete the temporary file from /tmp", schema)
result.value("intent")       # "delete"
result.selected("effects")   # ("delete",)
result.feasible              # True
```

Without the constraint decoder, independent argmax could have produced the illegal combination
`intent=delete, effects=["read_only"]`.

## Use cases

- **Agent guardrails.** [`safety-pii.md`](safety-pii.md)'s GLiGuard checkpoint scores
  `prompt_safety` (safe/unsafe) and harm tasks (`prompt_toxicity`, `jailbreak_detection`)
  independently — so a prompt can come back `prompt_safety="safe"` while a toxicity label still
  fires, a contradiction that file's "Interpreting outputs" section currently resolves with a
  manual OR-rule (`unsafe if prompt_safety=="unsafe" OR any harm task is non-benign`). Declaring
  that same rule as `C.implies(("prompt_toxicity", label), ("prompt_safety", "unsafe"))` in a
  `Classifier` schema would make the contradiction structurally impossible instead of caught
  after the fact — **note the released `gliguard-LLMGuardrails-300M` checkpoint does not do this
  internally**, it's a span-architecture model called through plain `classify_text`; you'd need
  to wrap its label set in your own `Classifier` schema on a GLiNER2.5 boundary checkpoint to get
  the guarantee.
- **Model/agent routing.** This skill's own [`route_usecase.py`](route_usecase.py) is the
  unconstrained version of this idea — one `classify_text` call scoring every reference file
  independently. A router with hard compatibility rules between tasks (e.g. "if routed to a
  training-capable tool, the task type must be one that tool supports") is exactly what
  `Classifier` + `.constrain(...)` is for instead of `classify_text`.

## Reading results

```python
result.value("intent")          # exclusive task -> str
result.value("effects")         # multi task -> tuple/list of labels
result.confidence("intent")     # scalar
result.probabilities("intent")  # label -> probability
result.feasible                 # False if constraints could not be fully met
result.violations               # unresolved constraint objects, if any
result.to_dict(include_confidence=False)
result["intent"]                # per-task TaskResult (unknown name raises SchemaError)
```

## Decoding controls

```python
config = ClassificationConfig(
    decoder="auto",           # "auto" | "independent" | "exact" | "beam"
    beam_size=16,
    exact_node_budget=200_000,
    candidate_threshold=0.5,
    max_candidates_per_task=64,
    include_confidence=True,
    on_infeasible="relax",    # "relax" | "min_violations" | "raise"
    batch_size=8,
)
result = clf.classify(text, schema, config=config)
```

`decoder="independent"` ignores cross-task constraints (same spirit as `classify_text`);
`"exact"` searches for a feasible assignment within budget; `"beam"` does beam search; `"auto"`
picks exact when small enough, else beam. Split scoring/decoding with `clf.score(...)` /
`clf.decode(...)` when needed. Schemas compile and cache by fingerprint (LRU, cap 128).

## Infeasible assignments

```python
config = ClassificationConfig(on_infeasible="raise")
try:
    clf.classify(text, schema, config=config)
except InfeasibleError:
    ...
```

| `on_infeasible` | Behavior |
|---|---|
| `"relax"` (default) | Best-effort assignment; check `result.feasible` |
| `"min_violations"` | Minimize broken constraints |
| `"raise"` | Raise `InfeasibleError` |

Always inspect `result.feasible` in production if keeping the default `"relax"`.

## Descriptions, instructions, few-shot examples

```python
schema = (
    ClassificationSchema()
    .single("ticket", {"billing": "Invoices, refunds, payment failures", "technical": "Bugs, outages, login failures"},
            instruction="Classify the support ticket. Prefer technical when a bug is described.",
            examples=(("I was double charged last month", "billing"), ("The app crashes on launch", "technical")),
            threshold=0.4)
)
```

## Batch and long documents

```python
results = clf.batch_classify(texts, schema, config=ClassificationConfig(batch_size=8))

# Long docs: aggregate logits across overlapping chunks, then decode ONCE (keeps constraints global)
result = clf.classify_long(open("runbook.txt").read(), schema, chunk_size=384, chunk_overlap=64, aggregate="max")
```

Do not stitch independent per-chunk labels for constrained tasks — chunk-wise `classify_text_long`
can disagree across pages and cannot enforce `implies`.

## Best practices

- Declare every task before `.constrain(...)`.
- Use `.single`/`.ordinal` for exclusive decisions; `.multi` only when several labels can coexist.
- Encode policy as constraints instead of post-filtering independent argmax.
- Start with `decoder="auto"`, `on_infeasible="relax"`, then tighten as needed.
- Use `{label: description}` for domain jargon.
- Rebuild the schema object only when tasks/constraints change (compile cache is fingerprinted).
- A single unconstrained head is cheaper reasoning-wise with plain `model.classify_text`.
