"""Submission gates — portal mechanics and deliverable completeness.

The submission is one-shot and unmodifiable, so these run as part of `make check`
rather than as a manual checklist.
"""

from __future__ import annotations

import sys

import pytest

from aed_siting.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))

DESCRIPTION = ROOT / "docs" / "submission_description.md"
MAX_CHARS = 1000

#: The brief's six required deliverables, and where each lives.
DELIVERABLES = {
    "working prototype": ROOT / "app.py",
    "problem and user definition": ROOT / "README.md",
    "method card": ROOT / "docs" / "method_card.md",
    "baseline and evaluation report": ROOT / "docs" / "evaluation.md",
    "data and reproducibility package": ROOT / "data" / "manifest.json",
    "safety and privacy statement": ROOT / "docs" / "safety.md",
}


@pytest.mark.gate
def test_all_six_deliverable_docs_present():
    for label, path in DELIVERABLES.items():
        assert path.exists(), f"missing deliverable: {label} ({path.name})"
        assert path.stat().st_size > 200, f"deliverable looks like a stub: {label}"


@pytest.mark.gate
def test_submission_description_under_1000_chars():
    text = DESCRIPTION.read_text().strip()
    assert len(text) <= MAX_CHARS, f"{len(text)} chars exceeds the portal limit of {MAX_CHARS}"
    assert len(text) > 400, "description is too thin to cover four points"


@pytest.mark.gate
def test_submission_description_covers_four_points():
    """What was built, the problem, how it works, what would improve with more time."""
    lowered = DESCRIPTION.read_text().lower()
    assert "install queue" in lowered or "ranked" in lowered, "does not say what was built"
    assert "problem" in lowered, "does not state the problem"
    assert "greedy" in lowered or "coverage" in lowered, "does not say how it works"
    assert "improve" in lowered, "does not say what would improve with more time"


@pytest.mark.gate
def test_no_secrets_in_repo():
    from check_secrets import scan

    hits = scan()
    assert not hits, f"credential-like strings in tracked files: {hits[:5]}"


@pytest.mark.gate
def test_supplemental_rules_table_present():
    """The brief's six supplemental-data rules must be mapped, not assumed."""
    text = (ROOT / "docs" / "method_card.md").read_text()
    assert "supplemental-data compliance" in text.lower()
    for phrase in ("Freeze API-derived inputs", "Aggregate demand", "no individual risk"):
        assert phrase.lower() in text.lower(), f"supplemental rule not addressed: {phrase!r}"


@pytest.mark.gate
def test_readme_has_what_i_would_improve():
    """Required by the portal, not optional."""
    text = (ROOT / "README.md").read_text().lower()
    assert "what i would improve" in text
    assert "phase 7" not in text.split("what i would improve")[1][:400], (
        "the improvements section is still a placeholder"
    )


@pytest.mark.gate
def test_evaluation_reports_negative_results():
    """The brief asks for failure cases and negative as well as positive results."""
    text = (ROOT / "docs" / "evaluation.md").read_text().lower()
    for phrase in ("negative results", "consensus set is empty", "unchanged"):
        assert phrase in text, f"evaluation report omits: {phrase!r}"


@pytest.mark.gate
def test_reproduction_command_documented():
    text = (ROOT / "README.md").read_text()
    assert "uv sync" in text
    assert "aed_siting.pipeline" in text
