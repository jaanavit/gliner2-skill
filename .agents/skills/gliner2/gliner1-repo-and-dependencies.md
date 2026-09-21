# GLiNER (v1): Repository Layout & Dependencies

What's actually in [urchade/GLiNER](https://github.com/urchade/GLiNER) and its real dependency
manifest (from `pyproject.toml`/`requirements.txt`, not just the docs site's `pip install`
snippets) — most important if you're installing **both** `gliner` and `gliner2` in the same
project, since they pull different, independently-versioned dependency trees.

## Repository layout

```
GLiNER/
├── gliner/          # the installed package (models, config, decoding, onnx/openvino runtime, serve)
├── tests/           # pytest suite
├── scripts/         # convert_to_onnx.py, convert_to_openvino.py — see gliner1-onnx-export.md
├── configs/         # training YAML configs (config.yaml, config_streaming_span.yaml, ...)
├── data/            # dataset preprocessing scripts (Pile-NER, NuNER)
├── examples/        # notebooks, incl. synthetic_data_generation.ipynb
├── benchmarks/       # benchmark scripts/results
├── docs/            # Sphinx source for https://urchade.github.io/GLiNER/
├── train.py         # training entry point — see gliner1-training.md
├── eval.py          # evaluation entry point
├── demo.py          # interactive Gradio demo
├── compose.yml      # Docker Compose (demo/serve)
├── pyproject.toml   # canonical dependency manifest (see below)
├── requirements.txt # minimal pin set used by the "install from source" path
├── uv.lock          # locked dependency versions (dev workflow uses uv)
├── LICENSE          # Apache-2.0
└── RELEASE.md        # release process notes
```

`gliner/` is the only directory that ships in the installed package (`[tool.setuptools.packages.
find] include = ["gliner", "gliner.*"]`); everything else is repo-only (training data prep,
scripts, docs, examples) and only matters if you clone the repo rather than `pip install gliner`.

## Dependencies (from `pyproject.toml`, source of truth)

```toml
requires-python = ">=3.10"

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.51.3,<5.17.0",
    "huggingface_hub>=0.21.4",
    "numpy",
    "packaging",
    "safetensors",
    "tqdm",
    "sentencepiece",
]
```

**Torch and transformers are base (non-optional) dependencies.** `pip install gliner` always pulls
PyTorch — there is no lightweight/cloud-only install path (contrast with `gliner2`, below).

| Extra | Adds | Use |
|---|---|---|
| `onnx` | `onnxruntime` | CPU ONNX Runtime inference — [gliner1-onnx-export.md](gliner1-onnx-export.md) |
| `gpu` | `onnxruntime-gpu` | CUDA ONNX Runtime inference (pick `onnx` **or** `gpu`, never both — same Python module) |
| `openvino` | `openvino` | OpenVINO IR export/inference |
| `tokenizers` | `langdetect`, `python-mecab-ko`, `janome`, `jieba3`, `camel_tools`, `indic-nlp-library`, `spacy`, `stanza` | Non-whitespace word splitters for multilingual checkpoints — see `words_splitter_type` in [gliner1-configs-architectures.md](gliner1-configs-architectures.md) |
| `training` | `accelerate` | Fine-tuning — [gliner1-training.md](gliner1-training.md) |
| `stanza` | `stanza`, `langdetect` | Stanza-tokenizer-specific models |
| `serve` | `ray[serve]>=2.9.0` | Production HTTP serving — [gliner1-serving.md](gliner1-serving.md) |
| `dev` (dependency-group, not a pip extra) | `pytest`, `pytest-asyncio`, `ruff` | Contributing to the repo itself |

`requirements.txt` (used by the "install from source" path in
[gliner1-intro.md](gliner1-intro.md)) pins a **different, currently stricter** floor —
`transformers>=4.57.3` vs. `pyproject.toml`'s `>=4.51.3`. If you hit a transformers-version error
installing from source that a plain `pip install gliner` didn't produce, this is why; treat
`requirements.txt` as the more current constraint when the two disagree.

Repo tooling: `ruff` (line-length 120, target py310, Google-convention docstrings) for lint/format,
`pytest` + `pytest-asyncio` for tests. Contributing: `pip install -e ".[dev]"`, then
`ruff check . --fix && ruff format .` before a PR.

## `gliner` vs `gliner2`: dependency comparison

The two packages version their dependencies **independently** and don't coordinate releases —
expect drift over time, not just at this snapshot.

| | `gliner` (v1) | `gliner2` |
|---|---|---|
| Base deps | `torch`, `transformers`, `huggingface_hub`, `numpy`, `packaging`, `safetensors`, `tqdm`, `sentencepiece` — **torch is mandatory** | `pydantic`, `requests`, `tqdm`, `urllib3` — **no torch**; the base install works with the cloud API alone |
| Local inference | Always available (base install) | Needs the `[local]` extra: `numpy`, `peft`, `safetensors`, `torch>=2.1,<3`, `transformers>=4.38,<5` |
| Training | `[training]` extra: `accelerate` | `[train]` extra: adds `peft`, `PyYAML`, plus the `[local]` set |
| `transformers` floor | `>=4.51.3,<5.17.0` (pyproject) / `>=4.57.3` (requirements.txt) | `>=4.38,<5` (via `[local]`/`[train]`) |
| `torch` floor | `>=2.0.0`, no upper bound | `>=2.1,<3` (via `[local]`/`[train]`) |
| Python | `>=3.10` | `>=3.10` |

**Practical implication:** if a project needs both packages, `pip`'s resolver can usually satisfy
both `transformers` ranges at once (GLiNER's floor is the binding constraint since it's higher and
GLiNER2's ceiling is looser), but this isn't guaranteed to stay conflict-free as either project
moves its pins — and the two pull genuinely different secondary dependency trees (`ray[serve]`,
`onnxruntime*`, `openvino`, language-specific tokenizers for `gliner`; `peft`-only for `gliner2`).
**Prefer separate virtual environments per package** (e.g. two `uv` projects, or a `gliner-env`
and a `gliner2-env`) unless you've specifically verified one shared environment resolves cleanly
for the extras you need from each. This matters more than it looks: `gliner`'s base install is
already "heavy" (torch + transformers unconditionally), while `gliner2`'s is deliberately "light"
(no torch until `[local]`) specifically so callers who only need the cloud API
([api-access.md](api-access.md)) can skip the heavy stack entirely — installing both into one env
by default erases that distinction for no benefit if you don't need both loaded at once.

## Ecosystem (from GLiNER's own README)

GLiNER's README lists GLiNER2 as one of several downstream/sibling projects — useful context for
"which package" questions beyond this skill's two:

| Project | What it is |
|---|---|
| [GLiNER2](https://github.com/fastino-ai/GLiNER2) | This skill's other half — unified multi-task NER/classification/structured extraction |
| [GLiClass](https://github.com/Knowledgator/GLiClass) | Zero-shot text classification, GLiNER-style architecture |
| [GLinker](https://github.com/Knowledgator/GLinker) | Entity linking |
| [GLiNER.cpp](https://github.com/Knowledgator/GLiNER.cpp) | C++ inference implementation |
| [gline-rs](https://github.com/fbilhaut/gline-rs) | Rust implementation |
| [vllm-factory](https://github.com/ddickmann/vllm-factory) | vLLM integration for GLiNER serving |
| [gliner-spacy](https://github.com/theirstory/gliner-spacy) | spaCy integration — see [gliner1-pipelines.md](gliner1-pipelines.md) |

None of these are covered by this skill beyond the mention above — treat this table as a pointer,
not documented behavior.
