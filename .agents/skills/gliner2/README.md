# GLiNER2 agent skill

Drop-in skill for Claude Code, Cursor, Codex, and other agent environments that read
`.agents/skills/` (or `.claude/skills/` on the same path via symlink). Gives your coding agent
full context on GLiNER2 (`gliner2` PyPI package) and the original GLiNER (`gliner` PyPI package)
— model/architecture selection, every extraction method (NER, classification, structured/JSON,
relations, multi-task schemas), training/fine-tuning, and Pioneer's hosted inference API.

This page is for installing the skill somewhere that doesn't have it yet. The skill itself, read
by the agent once installed, is [`SKILL.md`](SKILL.md).

## Installation

<details>
<summary>Manual copy (works today, any agent)</summary>

This directory is self-contained — no dependency on the rest of the repo it currently lives in.
Copy the whole `gliner2/` folder (`SKILL.md` + every `*.md` reference file + `route_usecase.py` +
`test_route_usecase.py`) into your agent's skills directory:

```bash
cp -R /path/to/this/gliner2 /path/to/your-project/.agents/skills/gliner2
```

- Claude Code / Cursor: `.claude/skills/gliner2/` or `.agents/skills/gliner2/`, per your repo's
  convention (Claude Code reads `.claude/skills/` only; Cursor 2.4+ and Codex both read
  `.agents/skills/` natively).
- Codex: `.agents/skills/gliner2/` (scanned from cwd up to the repo root).

</details>

<details>
<summary>Copy to your agent</summary>

Paste this prompt into your coding agent:

```text
Install the GLiNER2 skill: copy the directory at <path-or-URL-to-gliner2-skill> into this
project's agent skills directory (.agents/skills/gliner2/, or .claude/skills/gliner2/ if this
agent only reads that path), preserving every file, including route_usecase.py and
test_route_usecase.py. Then use the GLiNER2 skill whenever working with GLiNER2 or GLiNER in
this project.
```

Replace `<path-or-URL...>` with wherever this directory lives for you — a shared drive, another
repo's path, or a public GitHub URL once one exists (see below). Use one installation method to
avoid duplicate copies.

</details>

<details>
<summary>Publishing this publicly (not done yet)</summary>

To get the same one-command install TypeSafe's skill has
(`claude plugin marketplace add ...` / `npx skills add ...`), this directory would need to move
into its own public repo with a `skills/<name>/SKILL.md` layout (the convention `skills.sh` and
the Claude Code plugin marketplace both expect), get a marketplace manifest, and pick a license.
That's a separate decision — repo name/ownership/visibility — not something to do silently as
part of writing the skill content itself.

</details>

<details>
<summary>Updates</summary>

For a manual copy, just re-copy the directory over the old one. There's no versioning yet — if
you're editing your copy locally, diff before overwriting.

</details>

## Example prompts

- **Baseline first, before fine-tuning anything:**
  ```text
  Using the GLiNER2 skill, run fastino/gliner2.5-base-v1 zero-shot over a sample of my dataset
  at <path> for <task>, and show me where it gets it wrong.
  ```
- **Escalate to fine-tuning once zero-shot plateaus** — this is the biggest accuracy lever once
  label-description tuning is exhausted, bigger than further schema tweaking:
  ```text
  Using the GLiNER2 skill, take the zero-shot misses above, help me label ~100-200 of them into
  the training-data-format.md JSONL shape, fine-tune fastino/gliner2.5-base-v1 on it per
  training.md, and compare accuracy against the zero-shot baseline on a held-out split
  (evaluation.md) with precision/recall/F1, not a spot check.
  ```
- **Point at a specific method:**
  ```text
  Using the GLiNER2 skill, set up structured JSON extraction for <schema> over my dataset at
  <path>.
  ```

## Best practices for using this skill

- Let `SKILL.md`'s routing table pick the reference file — don't guess or read everything.
- Default to GLiNER2.5 / `AutoExtractor`; only follow a `gliner1-*.md` link for the specific
  cases `SKILL.md`'s disambiguation table calls out.
- Name the skill explicitly ("use the GLiNER2 skill") if your agent doesn't auto-invoke it.

## Good practice

1. Start zero-shot. Fine-tuning needs labeled data and training compute; zero-shot with good
   label descriptions (see `SKILL.md`'s "descriptions beat bare label lists") is often enough.
2. When zero-shot plateaus, the fix is almost always fine-tuning on your own labeled misses
   (`training.md`) or a LoRA adapter (`lora-adapters.md`) — not more wordsmithing on label
   descriptions.
3. Validate training data (`dataset.validate(strict=True, raise_on_error=True)`) before spending
   compute on a training run — most "fine-tune isn't working" reports are a data problem.
4. Keep a held-out test set and compare fine-tuned vs. zero-shot on it before trusting the
   fine-tune — don't eyeball a handful of examples.

## Extending this skill

Each GLiNER2 file (no prefix) mirrors one [GLiNER2 tutorial](https://github.com/fastino-ai/GLiNER2/tree/main/tutorial),
named by topic rather than tutorial number so it survives upstream renumbering. New
capability → new `<topic>.md` (setup → API → params → examples → best practices) + one row in
`SKILL.md`'s routing table.

Each GLiNER v1 file uses a `gliner1-` prefix and mirrors one [GLiNER docs](https://urchade.github.io/GLiNER/)
User Guide page (not the auto-generated API reference — same tutorials-not-docstrings scope). New
docs-site page → matching `gliner1-<topic>.md` + a row in `SKILL.md`'s "GLiNER vs GLiNER2" table.
`gliner1-intro.md` is that sub-tree's own router — extend it there, not `SKILL.md`'s GLiNER2
table.

Keep `SKILL.md` itself under a few hundred lines — depth belongs in the reference files, not the
router. `test_route_usecase.py` enforces that every reference file has a `route_usecase.py`
entry and a `SKILL.md` link (or, for `gliner1-*.md`, a link from `SKILL.md` or
`gliner1-intro.md`) — run it after adding a file.

## Common issues

- **Agent isn't using the skill.** Name it explicitly ("use the GLiNER2 skill"), or check that
  your phrasing matches a keyword in `SKILL.md`'s frontmatter `description`.
- **`route_usecase.py` routes to the wrong file.** See `SKILL.md`'s "Programmatic routing"
  caveat — multi-label descriptions interact; retune and re-run the full test battery, not just
  the one failing case.
- **A fine-tune isn't beating zero-shot.** Usually a data issue: run `dataset.validate()` first,
  confirm entity mentions exist verbatim in the input text, and check you're evaluating both
  models on the same held-out split.
