# GLiNER2 Agent Skill

A drop-in [Agent Skill](https://docs.cursor.com) for coding agents (Cursor, Claude Code, Codex,
and anything else that reads `.agents/skills/` or `.claude/skills/`) that gives your agent full,
tested working knowledge of **GLiNER2** (`gliner2` PyPI package) and the original **GLiNER**
(`gliner` PyPI package) — schema selection, entity/JSON/relation extraction, span attributes,
Joint IE, long-context handling, training, and a running list of real zero-shot failure modes
and fixes found by actually executing the documented APIs, not just reading their docs.

Once installed, just tell your agent to use it — e.g. *"Using the GLiNER2 skill, extract
entities and sentiment from this dataset."*

## Install it in your own project

```bash
git clone https://github.com/jaanavit/gliner2-skill.git /tmp/gliner2-skill
mkdir -p .agents/skills
cp -R /tmp/gliner2-skill/.agents/skills/gliner2 .agents/skills/gliner2
rm -rf /tmp/gliner2-skill
```

- **Cursor (2.4+) / Codex** read `.agents/skills/` natively — the command above is all you need.
- **Claude Code** reads `.claude/skills/` instead — use that path in the `mkdir`/`cp` above, or
  symlink one to the other: `ln -s .agents/skills/gliner2 .claude/skills/gliner2`.

That's it — no build step. The skill is just markdown + two small Python files
(`route_usecase.py`, `test_route_usecase.py`) used for optional non-agent routing checks; it adds
no dependency to your project by itself.

## Get your agent running it

The skill covers its own setup as **Step 0** in
[`SKILL.md`](.agents/skills/gliner2/SKILL.md) — tell your agent to follow it, or run it yourself:

```bash
python3 --version                                  # must be 3.10+
python3 -m venv .venv && source .venv/bin/activate
pip install "gliner2[local]" protobuf sentencepiece
python -c "from gliner2 import AutoExtractor; print(AutoExtractor)"   # should print with no error
```

Then just ask your agent to use the skill on your data — it will pick the right extraction
method (entities, classification, structured/JSON, relations, span attributes, Joint IE, ...)
from `SKILL.md`'s routing table.

## What's in here

- [`SKILL.md`](.agents/skills/gliner2/SKILL.md) — the entry point: environment checklist,
  package/model selection, and a routing table to every method.
- One reference file per capability (`entity-extraction.md`, `json-extraction.md`,
  `relation-extraction.md`, `span-attributes.md`, `joint-ie.md`, `long-context.md`,
  `training.md`, ...) — the agent reads only the ones a task needs.
- `gliner1-*.md` — a self-contained sub-tree for the original GLiNER (v1) package, for cases
  GLiNER2 doesn't cover (streaming NER, Ray Serve deployment, ONNX export, custom architectures).
- Several caveats throughout, each backed by a reproduced failure on real data rather than
  guesswork — e.g. a choice/classification field nested in a repeated structure gets scored once
  per document and copied into every instance, a defined-term literal-match trap in legal text,
  and combined schemas silently starving under long documents. See the "Known limitations"
  sections in [`joint-ie.md`](.agents/skills/gliner2/joint-ie.md) and
  [`json-extraction.md`](.agents/skills/gliner2/json-extraction.md) in particular.

## Updating

There's no versioning yet — re-clone and re-copy over your existing
`.agents/skills/gliner2/` to pick up changes.

## License

This skill documents the third-party `gliner2`/`gliner` PyPI packages
([fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2), Apache 2.0) but is not affiliated
with or published by Fastino. No license file is included yet — ask before redistributing
beyond direct use.
