"""Manifest and provenance gates — deliverable 5 and supplemental rules 1-3."""

from __future__ import annotations

import pytest

from aed_siting.config import DATASET_IDS
from aed_siting.io.manifest import REQUIRED_ENTRY_FIELDS, load_manifest, verify_checksums


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.mark.gate
def test_manifest_checksums_match():
    results = verify_checksums()
    assert results, "manifest lists no sources"
    bad = [r for r in results if not r.ok]
    assert not bad, f"checksum mismatch: {[(r.key, r.path.name) for r in bad]}"


@pytest.mark.gate
def test_manifest_fields_complete(manifest):
    """Supplemental rule 2: URL, publisher, date, licence, fields used, transformations."""
    for entry in manifest["sources"]:
        missing = [f for f in REQUIRED_ENTRY_FIELDS if f not in entry]
        assert not missing, f"{entry.get('key')}: manifest missing {missing}"
        assert entry["publisher"], f"{entry['key']}: no publisher"
        assert entry["licence"], f"{entry['key']}: no licence"
        assert entry["retrieved_utc"], f"{entry['key']}: no retrieval timestamp"


@pytest.mark.gate
def test_aed_dataset_id_pinned(manifest):
    """The AED layer has near-duplicate siblings on data.gov.sg; pin the right one."""
    entry = next(e for e in manifest["sources"] if e["key"] == "aed")
    assert entry["dataset_id"] == DATASET_IDS["aed"]
    assert entry["name"] == "Public Access AEDs"


@pytest.mark.gate
def test_no_sea_variant_used(manifest):
    """Sea-inclusive boundaries would inflate the study area and flood the grid."""
    for key in ("planning_areas", "subzones"):
        entry = next(e for e in manifest["sources"] if e["key"] == key)
        assert entry["dataset_id"] == DATASET_IDS[key]
        assert "No Sea" in entry["name"], f"{key}: not the No Sea variant"


@pytest.mark.gate
def test_both_dataset_dates_recorded(manifest):
    """The brief distinguishes the Feb 2020 data vintage from the page update date."""
    entry = next(e for e in manifest["sources"] if e["key"] == "aed")
    assert entry["coverage_date"] == "2020-02"
    assert entry["page_last_updated"] == "2025-11-13"


@pytest.mark.gate
def test_licence_table_includes_odbl(manifest):
    """OSM is ODbL: attribution and share-alike must be stated, not implied.

    One entry per town, keyed ``osm:<slug>``; every one must carry the licence.
    """
    entries = [e for e in manifest["sources"] if e["key"].startswith("osm")]
    assert entries, "no OSM source in the manifest"
    for entry in entries:
        licence = entry["licence"]
        assert "ODbL" in licence
        assert "attribution" in licence.lower()
        assert "share-alike" in licence.lower()


@pytest.mark.gate
def test_osm_extract_frozen_with_timestamp(manifest):
    """OSM is unversioned, so each frozen extract is the only reproducible reference."""
    entries = [e for e in manifest["sources"] if e["key"].startswith("osm")]
    assert len(entries) >= 1
    for entry in entries:
        assert entry["retrieved_utc"]
        assert entry["sha256"]
        assert "unversioned" in entry.get("notes", "").lower()
