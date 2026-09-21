"""Tests for route_usecase.py. No gliner2/torch dependency -- the model is stubbed."""

from __future__ import annotations

from pathlib import Path

import pytest

from route_usecase import USE_CASES, RouteMatch, route


class StubSchema:
    """Records the classification call so tests can assert on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], bool, float]] = []

    def classification(self, name: str, labels: dict[str, str], multi_label: bool, cls_threshold: float) -> "StubSchema":
        self.calls.append((name, labels, multi_label, cls_threshold))
        return self


class StubModel:
    """Minimal stand-in for `AutoExtractor` returning a canned classification result."""

    def __init__(self, matches: list[dict[str, object]]) -> None:
        self._matches = matches
        self.schema = StubSchema()

    def create_schema(self) -> StubSchema:
        return self.schema

    def extract(self, text: str, schema: object, *, include_confidence: bool = False) -> dict[str, object]:
        assert schema is self.schema
        assert include_confidence is True
        return {"gliner2_topic": self._matches}


def test_use_cases_cover_every_gliner2_reference_file_in_the_skill_directory() -> None:
    """Every GLiNER2 *.md file (no `gliner1-` prefix) except SKILL.md/README.md must have a
    routing entry, and vice versa. `gliner1-*.md` files are the separate GLiNER (v1) sub-tree --
    see "GLiNER vs GLiNER2" in SKILL.md -- and are deliberately out of scope for this
    GLiNER2-method router; they're covered by
    `test_gliner1_files_are_linked_from_skill_or_its_own_router` below. `README.md` is
    human-facing install/distribution documentation, not a GLiNER2 topic file, so it carries no
    routing entry either -- same category as `SKILL.md` itself.
    """
    skill_dir = Path(__file__).parent
    meta_files = {"SKILL.md", "README.md"}
    reference_files = {
        p.name
        for p in skill_dir.glob("*.md")
        if p.name not in meta_files and not p.name.startswith("gliner1-")
    }
    assert set(USE_CASES) == reference_files


def test_use_cases_are_linked_from_skill_md() -> None:
    """Catches a routing entry for a file that SKILL.md's table forgot to link, or vice versa."""
    skill_md = (Path(__file__).parent / "SKILL.md").read_text()
    for reference in USE_CASES:
        assert f"]({reference})" in skill_md, f"{reference} is not linked from SKILL.md"


def test_gliner1_files_are_linked_from_skill_or_its_own_router() -> None:
    """Every `gliner1-*.md` file must be reachable from SKILL.md's "GLiNER vs GLiNER2" table or
    from gliner1-intro.md's own routing table, so nothing in that sub-tree goes orphaned.
    """
    skill_dir = Path(__file__).parent
    gliner1_files = {p.name for p in skill_dir.glob("gliner1-*.md")}
    assert gliner1_files, "expected at least one gliner1-*.md file"

    skill_md = (skill_dir / "SKILL.md").read_text()
    intro_md = (skill_dir / "gliner1-intro.md").read_text()
    for reference in gliner1_files:
        linked = f"]({reference})" in skill_md or f"]({reference})" in intro_md
        assert linked, f"{reference} is not linked from SKILL.md or gliner1-intro.md"


def test_route_sorts_by_descending_confidence() -> None:
    model = StubModel(
        [
            {"label": "entity-extraction.md", "confidence": 0.4},
            {"label": "combined-schemas.md", "confidence": 0.9},
            {"label": "classification.md", "confidence": 0.6},
        ]
    )

    result = route("classify support tickets and pull out order IDs", model=model)

    assert result == [
        RouteMatch("combined-schemas.md", 0.9),
        RouteMatch("classification.md", 0.6),
        RouteMatch("entity-extraction.md", 0.4),
    ]


def test_route_returns_empty_list_when_nothing_clears_threshold() -> None:
    model = StubModel([])

    assert route("some ambiguous task", model=model) == []


def test_route_passes_use_cases_and_threshold_into_the_schema() -> None:
    model = StubModel([])

    route("task", model=model, threshold=0.42)

    [(name, labels, multi_label, cls_threshold)] = model.schema.calls
    assert name == "gliner2_topic"
    assert labels == USE_CASES
    assert multi_label is True
    assert cls_threshold == pytest.approx(0.42)
