"""Safety-gate tests.

The brief's safety gate is pass/fail and applied before scoring, so these are the
highest-consequence tests in the repository.
"""

from __future__ import annotations

import re

import pytest

from aed_siting.config import ROOT
from aed_siting.safety import AHA_CAVEAT, BANNER, DISCLOSURES, safe_failure

# Transcribed from the challenge brief. Any drift from this string is a gate failure.
EXPECTED_BANNER = (
    "Prototype for planning and simulation only—not for emergency use. "
    "In an emergency in Singapore, call 995 immediately and follow SCDF "
    "instructions. Use official SCDF/myResponder channels. Do not delay "
    "emergency action to use this prototype."
)


@pytest.mark.gate
def test_banner_is_verbatim():
    assert BANNER == EXPECTED_BANNER
    # The four load-bearing elements, in case the constant is ever reflowed.
    for fragment in ("planning and simulation only", "995", "SCDF", "myResponder"):
        assert fragment in BANNER


@pytest.mark.gate
def test_banner_in_readme_first_lines():
    readme = (ROOT / "README.md").read_text()
    head = "".join(readme.splitlines(keepends=True)[:12])
    assert "not for emergency use" in head, "banner must appear at the top of the README"
    assert "995" in head


@pytest.mark.gate
def test_aha_citations_carry_caveat():
    """Any AHA citation must sit adjacent to the brief's no-survival-claim caveat.

    The brief is explicit that these facts establish the importance of
    preparedness and do not license a claim of improved survival.
    """
    pattern = re.compile(r"American Heart Association|\bAHA\b")
    caveat_core = "do not mean"
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        text = path.read_text()
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 700) : match.end() + 700]
            assert caveat_core in window, (
                f"{path.name}: AHA citation at offset {match.start()} is not "
                "adjacent to the required caveat"
            )


@pytest.mark.gate
def test_disclosures_cover_required_points():
    """Point-of-use disclosures must name the snapshot date and refuse readiness claims."""
    joined = " ".join(DISCLOSURES)
    assert "February 2020" in joined
    assert "not proof" in joined or "not a live copy" in joined
    for term in ("modelled", "planning estimates"):
        assert term in joined


def test_aha_caveat_text_is_usable():
    assert "do not mean" in AHA_CAVEAT


def test_safe_failure_abstains():
    message = safe_failure("operating hours could not be parsed.")
    assert "No recommendation" in message
    assert "abstains" in message
