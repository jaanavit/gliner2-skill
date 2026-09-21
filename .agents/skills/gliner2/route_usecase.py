#!/usr/bin/env python3
"""Route a free-text GLiNER2 task description to the right reference file, using GLiNER2 itself.

Turns the "Route by use case" table in SKILL.md into a live multi-label classification
schema: each reference file becomes a label, and its "I need to..." row becomes the label
description. This gives every caller -- an agent, a CLI, a docs bot -- a deterministic,
model-based alternative to eyeballing the table, and doubles as a live example of
`classify_text`-style routing with label descriptions (see classification.md).

Usage:
    python route_usecase.py "pull structured fields out of scanned contracts"
    python route_usecase.py "classify support tickets and also extract order IDs" --top 3

Tuning notes (read before editing USE_CASES):
    Verified end-to-end against fastino/gliner2.5-base-v1 on a 19-task battery (one clear-cut
    task per reference file): 16/19 correct as the top match, 17/19 within the top 3 at the
    default --threshold 0.25. Label description quality drives this directly -- a sparse
    description ("long-context.md": "Process a document longer than the model's context
    window...") scored the right label at confidence 0.004 on an on-the-nose task; rewriting it
    with concrete trigger phrases ("...such as a 50-page PDF or a multi-hour transcript")
    brought it to 0.73-0.97.

    classify_text-style multi-label decoding scores every label in one joint pass, not
    independently -- tightening one label's description to fix a miss measurably shifts *other*
    labels' scores (fixing a Chinese-word-segmentation phrasing that scored
    performance-tuning.md at 0.08 pushed it to 0.81, but the same edit dropped an unrelated
    regex-validators.md case from 0.4 to below threshold). Retuning descriptions is
    whack-a-mole, not monotonic improvement -- after any edit, re-run the full battery, not just
    the one case you're fixing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gliner2 import AutoExtractor

DEFAULT_MODEL = "fastino/gliner2.5-small-v1"  # fast/CPU checkpoint; routing only needs coarse topic classification
DEFAULT_THRESHOLD = 0.25

# One entry per reference file in this skill directory. Keep this in sync with the
# "Route by use case" table in SKILL.md -- each description here is exactly that row's
# "I need to..." text, reused verbatim as the classification label description.
USE_CASES: dict[str, str] = {
    "classification.md": "Bucket a whole document or message into one or more categories, such as sentiment, topic, spam vs not spam, or customer intent",
    "entity-extraction.md": "Pull named spans out of text, such as people, organizations, dates, locations, product names, or other custom entity types",
    "json-extraction.md": "Parse a structured record or JSON object out of unstructured text, such as a product listing, invoice, resume, or contact card, with typed fields",
    "combined-schemas.md": "Extract two or more of entities, classification, structured records, or relations from the same text in a single pass, instead of calling the model once per task",
    "relation-extraction.md": "Extract relationship pairs between two entities in text, such as who works for which company, who is located in which place, who is married to whom, or who founded what",
    "regex-validators.md": "Filter or validate already-extracted spans against a regular expression, such as confirming a span looks like a valid email address, phone number, or URL",
    "api-access.md": "Call GLiNER2 through a hosted cloud API endpoint instead of downloading and running a model locally",
    "span-attributes.md": "Attach a sub-label, like sentiment or severity, to each individual extracted entity span rather than classifying the document as a whole",
    "constrained-classification.md": "Enforce a hard logical rule between two classification tasks, such as one label implying another must also be present, or two labels being mutually exclusive",
    "joint-ie.md": "Extract entities and relations together as one consistent graph with typed endpoints, so entities and their relationships stay mutually consistent, such as each person having at most one employer",
    "long-context.md": "Process a document, report, contract, or transcript that is too long to fit in the model's context window in a single pass, such as a 50-page PDF or a multi-hour transcript",
    "safety-pii.md": "Detect and redact personally identifiable information such as names, social security numbers, or credit card numbers, or moderate a prompt or response for unsafe content",
    "training-data-format.md": "Build or format a JSONL training dataset with input text and annotated output labels for fine-tuning a GLiNER2 model",
    "training.md": "Fine-tune or train a GLiNER2 model from scratch, or continue training an existing model on a custom dataset",
    "lora-adapters.md": "Train a small, parameter-efficient LoRA adapter specialized for one narrow domain instead of fine-tuning the whole model",
    "adapter-switching.md": "Load or swap between multiple already-trained domain-specific adapters at inference time without reloading the base model",
    "performance-tuning.md": "Speed up model inference with quantization, torch.compile, or FlashDeBERTa, or change how text is tokenized or word-segmented to handle languages without whitespace between words, such as Chinese",
    "evaluation.md": "Measure whether a schema or model is actually performing well on your data, such as computing precision, recall, or F1 on a held-out labeled set, or comparing a zero-shot model against a fine-tuned one before deciding to ship it",
    "pioneer-api.md": "Call Pioneer's own hosted api.pioneer.ai REST inference endpoint directly over HTTP, such as choosing a model_id, calling a fine-tuned training job, or hitting the OpenAI-compatible chat completions surface",
}


class Extractor(Protocol):
    """Structural type for the subset of `AutoExtractor` this module calls.

    Lets tests pass a stub without importing (or installing) `gliner2`/`torch`.
    """

    def create_schema(self) -> object: ...

    def extract(self, text: str, schema: object, *, include_confidence: bool = False) -> dict[str, object]: ...


@dataclass(frozen=True)
class RouteMatch:
    """One candidate reference file for a task description, ranked by confidence."""

    reference: str
    confidence: float


def load_model(model_name: str = DEFAULT_MODEL) -> "AutoExtractor":
    """Load the GLiNER2.5 checkpoint used for routing.

    Args:
        model_name: Hub checkpoint id. Defaults to the small, CPU-fast GLiNER2.5
            checkpoint since routing only needs coarse topic classification.

    Returns:
        A loaded `AutoExtractor` instance. Requires the `gliner2[local]` extra.
    """
    from gliner2 import AutoExtractor

    return AutoExtractor.from_pretrained(model_name)


def route(task: str, model: Extractor, threshold: float = DEFAULT_THRESHOLD) -> list[RouteMatch]:
    """Classify a free-text task description against this skill's reference files.

    Args:
        task: Free-text description of what the caller wants to do with GLiNER2.
        model: A loaded extractor (`AutoExtractor` in production, a stub in tests).
        threshold: Multi-label confidence cutoff passed to `cls_threshold`. Lower it
            (e.g. 0.15) for more recall when nothing clears the default.

    Returns:
        Matching reference files sorted by descending confidence. Empty when nothing
        clears `threshold` -- callers should retry with a lower threshold or fall back
        to reading the SKILL.md table directly, not treat this as an error.
    """
    schema = model.create_schema().classification(
        "gliner2_topic", USE_CASES, multi_label=True, cls_threshold=threshold
    )
    result = model.extract(task, schema, include_confidence=True)
    matches = result.get("gliner2_topic", [])
    return sorted(
        (RouteMatch(reference=m["label"], confidence=m["confidence"]) for m in matches),
        key=lambda m: m.confidence,
        reverse=True,
    )


def main() -> int:
    """CLI entry point: print ranked reference-file matches for a task description."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="free-text description of the GLiNER2 task")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"checkpoint to route with (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"multi-label confidence cutoff (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument("--top", type=int, default=None, help="limit output to the top N matches")
    args = parser.parse_args()

    matches = route(args.task, model=load_model(args.model), threshold=args.threshold)
    if not matches:
        print(f"No reference file cleared threshold={args.threshold}. Try a lower --threshold or read SKILL.md directly.")
        return 1

    for match in matches[: args.top]:
        print(f"{match.confidence:.2f}  {match.reference}  -- {USE_CASES[match.reference]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
