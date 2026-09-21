# GLiNER2 Agent Skill

Drop-in [Agent Skill](https://docs.cursor.com) for Cursor / Claude Code / Codex — teaches your
agent GLiNER2 (`gliner2`) and GLiNER (`gliner`): schema selection, entity/JSON/relation
extraction, span attributes, Joint IE, long-context, training, and known zero-shot failure modes.

## Install

```bash
npx skills add jaanavit/gliner2-skill
```

Non-interactive / CI-friendly (installs to Cursor in the current project):

```bash
npx skills add jaanavit/gliner2-skill -a cursor -y
```

Use `-a claude-code` or `-a codex` for other agents, `-g` to install globally instead of
per-project, or `--list` to preview the skill without installing. See
[skills.sh](https://www.npmjs.com/package/skills) for the full CLI.

### Manual install (no npx)

```bash
git clone https://github.com/jaanavit/gliner2-skill.git /tmp/gliner2-skill
mkdir -p .agents/skills
cp -R /tmp/gliner2-skill/.agents/skills/gliner2 .agents/skills/gliner2
rm -rf /tmp/gliner2-skill
```

Claude Code reads `.claude/skills/` instead — use that path, or symlink:
`ln -s .agents/skills/gliner2 .claude/skills/gliner2`.

## Use

Tell your agent: *"Using the GLiNER2 skill, ..."* — it follows the routing table in
[`SKILL.md`](.agents/skills/gliner2/SKILL.md), starting with the Step 0 environment setup:

```bash
python3 --version   # must be 3.10+
python3 -m venv .venv && source .venv/bin/activate
pip install "gliner2[local]" protobuf sentencepiece
```

## License

Documents the third-party `gliner2`/`gliner` packages ([fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2),
Apache 2.0). Not affiliated with Fastino.
