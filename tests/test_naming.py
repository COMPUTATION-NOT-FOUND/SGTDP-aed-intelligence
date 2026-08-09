"""Gates on site identity and labelling.

"(unnamed building)" is not an answer a planner can act on, so label resolution
is treated as a correctness requirement rather than presentation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aed_siting.config import DEFAULT_TOWN, town_prefix
from aed_siting.naming import (
    LABEL_SOURCES,
    label_quality,
    resolve_label,
    site_ids,
    sites_reference,
)


@pytest.fixture(scope="module")
def candidates(town_artifacts):
    return town_artifacts["candidates"]


@pytest.mark.gate
def test_every_site_has_a_resolvable_label(candidates):
    """No site may fall through the ladder without a usable label."""
    labels = candidates["label"]
    assert labels.notna().all()
    assert (labels.astype(str).str.strip() != "").all()
    assert not labels.astype(str).str.contains("unnamed", case=False).any()


@pytest.mark.gate
def test_site_ids_are_stable_and_prefixed(candidates):
    """Ids key every artifact, so they must be unique, ordered and town-tagged."""
    ids = candidates["site_id"].tolist()
    assert len(set(ids)) == len(ids), "site ids are not unique"
    prefix = town_prefix(DEFAULT_TOWN)
    assert all(str(i).startswith(f"{prefix}-") for i in ids)
    assert ids == sorted(ids), "ids do not sort in candidate order"


@pytest.mark.gate
def test_label_source_is_always_recorded(candidates):
    """A reader must be able to tell a real name from a constructed one."""
    sources = set(candidates["label_source"].unique())
    assert sources <= set(LABEL_SOURCES), f"unknown label source: {sources - set(LABEL_SOURCES)}"


@pytest.mark.gate
def test_address_rungs_actually_fire(candidates):
    """The address ladder must produce labels, not collapse onto its weakest rung.

    Regression gate: `build_candidates` once dropped the `addr:*` columns before
    naming ran, so every unnamed block fell through to "Residential TP-084".
    """
    sources = candidates["label_source"].value_counts()
    from_address = int(sources.get("address_block", 0) + sources.get("address_street", 0))
    assert from_address > 0, "no label came from an address; addr:* tags are being dropped"
    assert from_address / len(candidates) > 0.20, (
        f"only {from_address}/{len(candidates)} labels came from addresses"
    )


def test_resolve_label_ladder_order():
    """Each rung is used only when the ones above it are unavailable."""
    assert resolve_label(pd.Series({"name": "Toa Payoh Mall"}), "TP-001") == (
        "Toa Payoh Mall",
        "osm_name",
    )
    assert resolve_label(pd.Series({"addr:housename": "Kim Keat Palm"}), "TP-002") == (
        "Kim Keat Palm",
        "osm_housename",
    )
    assert resolve_label(
        pd.Series({"addr:housenumber": "265", "addr:street": "Kim Keat Avenue"}), "TP-003"
    ) == ("Blk 265 Kim Keat Avenue", "address_block")
    label, source = resolve_label(
        pd.Series({"addr:street": "Braddell Road", "building": "school"}), "TP-004"
    )
    assert label == "School on Braddell Road" and source == "address_street"
    assert resolve_label(pd.Series({"building": "yes"}), "TP-005") == ("TP-005", "identifier")


def test_nullish_tags_do_not_become_labels():
    """OSM placeholders must not leak through as names."""
    for junk in ("nan", "None", "", "  ", "<NA>"):
        label, source = resolve_label(pd.Series({"name": junk, "building": "yes"}), "TP-009")
        assert label == "TP-009" and source == "identifier"


def test_site_ids_pad_to_a_consistent_width():
    ids = site_ids(1200, DEFAULT_TOWN)
    assert ids[0].endswith("-0001") and ids[-1].endswith("-1200")
    assert ids == sorted(ids)


@pytest.mark.gate
def test_sites_reference_resolves_ids_to_coordinates(candidates):
    """Any id in the queue must be lookup-able to a place on the ground."""
    reference = sites_reference(candidates)
    assert set(reference["site_id"]) == set(candidates["site_id"])
    assert reference["lat"].between(1.1, 1.5).all(), "latitudes outside Singapore"
    assert reference["lon"].between(103.5, 104.2).all(), "longitudes outside Singapore"


def test_label_quality_sums_to_the_candidate_count(candidates):
    quality = label_quality(candidates)
    assert quality["sites"].sum() == len(candidates)
    assert abs(quality["share"].sum() - 1.0) < 0.01
