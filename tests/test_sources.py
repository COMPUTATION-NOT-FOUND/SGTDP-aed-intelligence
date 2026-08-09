"""Gates on the source register and the brief's official resources."""

from __future__ import annotations

import re

import pytest

from aed_siting.config import DATASET_IDS, ROOT

SOURCES_DOC = ROOT / "docs" / "sources.md"

# The six official resources named in the challenge brief, each with the URL the
# citation must resolve to.
OFFICIAL_RESOURCES = {
    "Public Access AEDs": "https://data.gov.sg/datasets/d_4e6b82c58a8a832f6f1fee5dfa6d47ea/view",
    "myResponder app": "https://www.scdf.gov.sg/home/community-and-volunteers/community-resources/myresponder-app",
    "national AED registry": "https://www.scdf.gov.sg/savealife",
    "Emergency Medical Services": "https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services",
    "Adult Basic Life Support": "https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/adult-basic-life-support",
    "CPR facts and statistics": "https://cpr.heart.org/en/resources/cpr-facts-and-stats",
}


@pytest.fixture(scope="module")
def sources_text() -> str:
    assert SOURCES_DOC.exists(), "docs/sources.md is missing"
    return SOURCES_DOC.read_text()


@pytest.mark.gate
def test_all_six_official_resources_cited(sources_text):
    """Every official resource the brief lists must appear with a working URL."""
    for label, url in OFFICIAL_RESOURCES.items():
        assert url in sources_text, f"official resource not cited by URL: {label} ({url})"


@pytest.mark.gate
def test_official_resources_have_stated_roles(sources_text):
    """Citations must carry a role, not sit in a bare link list."""
    for phrase in (
        "does not connect to, replace, duplicate, or compete with",
        "No integration with the national registry",
        "basis of the mandated safety banner",
    ):
        assert phrase in sources_text, f"missing stated role: {phrase!r}"


@pytest.mark.gate
def test_snapshot_staleness_is_quantified(sources_text):
    """The staleness argument must cite SCDF's own registry size, not hand-wave.

    SCDF states the live national registry holds 11,000+ AEDs; the snapshot used
    here has 9,644 records from February 2020.
    """
    assert "11,000" in sources_text
    assert "9,644" in sources_text
    assert "February 2020" in sources_text


@pytest.mark.gate
def test_dataset_ids_match_config(sources_text):
    """The register and the pinned constants must not drift apart."""
    for key, dataset_id in DATASET_IDS.items():
        assert dataset_id in sources_text, f"{key} dataset id absent from docs/sources.md"


@pytest.mark.gate
def test_odbl_obligations_stated(sources_text):
    """OSM is ODbL: attribution and share-alike must both be addressed."""
    assert "ODbL" in sources_text
    assert "share-alike" in sources_text.lower()
    assert "openstreetmap.org/copyright" in sources_text.lower()


@pytest.mark.gate
def test_excluded_fields_documented(sources_text):
    """Undocumented fields must be named as deliberately unused."""
    for field in ("XVAL", "YVAL", "INC_CRC", "FMEL_UPD_D", "AED_LOCATION_DESCRIPTION"):
        assert field in sources_text, f"{field} not documented as excluded"


@pytest.mark.gate
def test_no_bare_aha_statistic_in_repo():
    """An AHA statistic must never appear without its caveat nearby.

    Complements test_aha_citations_carry_caveat by scanning for the specific
    survival claim the brief singles out.
    """
    pattern = re.compile(r"double or triple", re.IGNORECASE)
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        text = path.read_text()
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 800) : match.end() + 800]
            assert "do not mean" in window, (
                f"{path.name}: survival statistic without adjacent caveat"
            )
