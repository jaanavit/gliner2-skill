# Joint Information Extraction (GLiNER2.5 only)

Extract **entities and relations together** under typed endpoints and graph constraints.
Independent `extract_entities()` + `extract_relations()` can produce an org that never actually
appears as a person's employer; `JointIE` scores candidates then searches a globally consistent
graph. Mirrors
[tutorial/15-joint_ie.md](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/15-joint_ie.md).

Requires a GLiNER2.5 boundary checkpoint trained with `enable_relations=True` (e.g.
`fastino/gliner2.5-multi-v1`) for relations; entity-only Joint IE works without the relation head.
For unconstrained relation tuples with no typed endpoints, use
[relation-extraction.md](relation-extraction.md) instead.

## Setup

```python
from gliner2.joint_ie import JointIE, JointIEConfig

joint = JointIE.from_pretrained("fastino/gliner2.5-multi-v1")
```

`JointIE.from_pretrained` loads through `AutoExtractor`. Prediction knobs belong in
`JointIEConfig` per `extract` call, not on `from_pretrained`.

## Independent extraction vs. Joint IE

Independent decoding (`extract_entities` + `extract_relations`) cannot guarantee: relation
head/tail types match the declared schema, a person has at most one employer, there are no
self-loops, or inverse pairs stay consistent. Joint IE enforces all of that while choosing the
graph.

## Declaring a schema

Entities first; relations may only reference known entity types.

```python
schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization")
    .relation("located_in", "organization", "location")
)

# With descriptions, thresholds, candidate caps
schema = (
    joint.create_schema()
    .entities({
        "person": "Named people",
        "organization": {"description": "Companies, agencies, or teams", "threshold": 0.3, "max_candidates": 16},
        "location": "Cities, countries, or addresses",
    })
    .relation("works_for", head="person", tail="organization", description="Employment or affiliation", threshold=0.25)
)
# head/tail can each be a list of types: .relation("affiliated_with", head="person", tail=["organization", "location"])
```

## Relation options

```python
.relation(
    "works_for", "person", "organization",
    directed=True, allow_self=False,
    unique_head=True, unique_tail=False,       # e.g. one employer per person
    max_per_head=1, max_per_tail=None,
    symmetric=False, inverse=None, acyclic=False,
    threshold=None, candidate_threshold=None,
)
```

| Option | Effect |
|---|---|
| `unique_head=True` | Each head has at most one edge of this type |
| `unique_tail=True` | Each tail has at most one incoming edge of this type |
| `symmetric=True` | Undirected; head/tail type sets must match (cannot combine with `inverse`) |
| `inverse="employed"` | Decoder may realize the named inverse relation |
| `acyclic=True` | Forbids cycles on this relation |
| `allow_self=True` | Permits a span linked to itself |

## Graph-level constraints

```python
schema = (
    joint.create_schema()
    .entities(["person", "organization", "location"])
    .relation("works_for", "person", "organization", unique_head=True)
    .relation("located_in", "organization", "location")
    .relation("knows", "person", "person", symmetric=True, allow_self=False)
    .no_self_loops()
    .at_most("located_in", per_head=1)
    .acyclic("works_for")
)
```

| Method | Meaning |
|---|---|
| `.no_self_loops()` / `.no_self_loops("knows")` | Forbid head == tail (all or one type) |
| `.at_most("works_for", per_head=1)` | Cap edges per head/tail |
| `.acyclic("reports_to")` | No directed cycles |
| `.constraint(obj)` | Attach a constraint instance directly |

Built-ins: `NoSelfLoops`, `MaxRelationsPerHead`, `MaxRelationsPerTail`, `AcyclicRelation`,
`TypedEndpoints`, `UniqueRelationPair`, `SymmetricRelation`, `InverseRelation`. Typed endpoints
from `.relation(head, tail)` are already compiled in.

## Running extraction

```python
result = joint.extract(text, schema, config=JointIEConfig(optimizer="beam", beam_size=32))
result.feasible
result.to_dict()
# {'entities': [{'id': 'e1', 'type': 'person', 'text': 'Alice', 'start': 0, 'end': 5, 'confidence': 0.94}, ...],
#  'relations': [{'type': 'works_for', 'head': 'e1', 'tail': 'e2', 'confidence': 0.88}, ...]}
```

Offsets are half-open `[start, end)` into `result.text`; verify with
`text[entity.start:entity.end] == entity.text`.

## Reading the graph

```python
result.entities                    # list[JointEntity]
result.relations                   # list[JointRelation]
result.entity("e2")                # lookup by id
result.entities_by_type("person")
result.relations_by_type("works_for")
result.outgoing("e1") / result.incoming("e2") / result.neighbors("e2")
result.relations_of("e1")
graph = result.to_networkx()       # requires networkx
```

Relation `head`/`tail` are entity IDs, not surface strings — look them up via `result.entity(id)`.

## Decoding config

```python
config = JointIEConfig(
    optimizer="beam",              # "beam" | "greedy" | "auto"
    beam_size=32,
    candidate_threshold=0.05,
    relation_role_threshold=0.05,
    top_k_entities=32, top_k_roles=12,
    relation_pair_cap=128, max_edges_per_type=256,
    include_confidence=True, include_spans=True,
    entity_threshold=None, batch_size=8,
)
```

Lower thresholds and larger `top_k_*` improve recall at higher cost — tune on a small labeled set.
`joint.score(...)` returns raw lattices without decoding. Schemas are compiled and cached; reuse
the same `JointSchema` object.

## Feasibility

`result.feasible == False` means the decoder could not satisfy the declared hard constraints and
fell back to an empty assignment — distinct from "no entities in this text":

```python
if not result.feasible:
    ...  # constraints too tight, or candidates missing required arguments
elif not result.entities:
    ...  # genuine empty extraction
```

If many documents come back infeasible: relax uniqueness constraints, raise `top_k_entities`, or
lower `candidate_threshold`.

## Long documents and batching

```python
result = joint.extract_long(open("biography.txt").read(), schema, chunk_size=384, chunk_overlap=64,
                             config=JointIEConfig(optimizer="beam"))
```

A mention is kept only if start and end fall in the **same chunk**; a relation is kept only if
**both endpoints** were extracted in the same chunk — cross-window edges are never synthesized.
Raise `chunk_overlap` if subject/object often straddle a boundary. See
[long-context.md](long-context.md).

```python
results = joint.batch_extract(texts, schema, config=JointIEConfig(batch_size=8))
results = joint.batch_extract(texts, [schema_a, schema_b])   # per-document schemas
```

## Use case: knowledge graph construction

A knowledge graph is only as reliable as its least consistent edge — every ingestion pipeline
built on independently-thresholded triples needs its own logic to reject dangling references,
enforce cardinality, and break cycles before the graph is trustworthy. Joint IE moves that work
into the extraction itself, so what comes back is already well-formed: e.g. building an agent's
memory graph of people/projects/commitments from documents, email, and chat history, where
`unique_head`/`acyclic`/`no_self_loops` keep every edge typed and the structure consistent as the
graph grows across many extraction calls, not just within one document.

## Best practices

- Declare only the closed set of entity types you actually want — Joint IE won't invent extras.
- Put typing in `.relation(head, tail)` rather than filtering edges post hoc.
- Use `unique_head` / `at_most(..., per_head=1)` for 1-to-N facts (employer, capital, HQ).
- Keep `no_self_loops()` on unless the domain truly has reflexive edges.
- Start with `optimizer="beam"` and modest `top_k_entities`; scale up if gold edges are missing.
- Always check `result.feasible` before treating an empty graph as "no facts".
- Verify `text[e.start:e.end] == e.text` while bringing up a new schema.
- Prefer Joint IE when graph consistency matters; prefer `extract_relations` when unlabeled
  head/tail strings are enough.

## Known limitations

- **Repeated mentions of the same real-world entity become separate graph nodes, so their edges
  duplicate.** If "Ledgerly" is named three times in a document, each mention is its own
  `JointEntity` with its own id, and a relation attached to "Ledgerly" can appear once per
  mention (e.g. the same `invests_in` edge showing up 3 times with slightly different
  confidences). This is expected — Joint IE does no cross-mention coreference — but it means any
  "knowledge graph" consumer (see the use case above) needs its own canonicalization/dedup pass
  (e.g. group entities by normalized text before counting edges) if it wants one node per
  real-world entity rather than one node per mention.
- **No constraint exists for mutual exclusion *between* different relation types.** The
  constraint DSL (`unique_head`, `acyclic`, `no_self_loops`, etc.) only constrains a single
  relation type against itself. Two semantically exclusive relation types (e.g. `acquires` vs.
  `invests_in` — a company is either bought outright or given capital, not both) can both fire on
  the same head/tail pair at high confidence, even with tuned descriptions and thresholds — e.g.
  a sentence like "invested $25M in the Series B round" can fire both `acquires` (0.98) and
  `invests_in` (0.99) on the same pair. There's no built-in fix yet; workaround is a post-hoc
  filter that keeps only the highest-confidence relation type per `(head, tail)` pair when two or
  more of your declared types are meant to be mutually exclusive.
