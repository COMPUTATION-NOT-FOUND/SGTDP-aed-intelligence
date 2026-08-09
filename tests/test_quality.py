"""Registry data-quality gates (Lane 3 techniques, Lane 2 purpose)."""

from __future__ import annotations

import pytest

from aed_siting.quality import find_colocated_sites, validate_postal_codes


@pytest.mark.gate
def test_aed_id_is_the_authoritative_device_key(aed_town):
    """Identity comes from AED_ID, not geometry.

    If this ever fails, the co-location logic below is unsafe: records sharing a
    coordinate could no longer be assumed to be distinct devices.
    """
    frame, _ = aed_town
    assert frame["AED_ID"].is_unique, "AED_ID is not unique; identity model is invalid"


@pytest.mark.gate
def test_colocated_records_are_retained_not_dropped(aed_town):
    """Distinct devices geocoded to one building centroid must survive loading.

    Dropping them would delete real AEDs, understate the existing network, and
    inflate the apparent value of every recommended placement.
    """
    frame, report = aed_town
    assert report.dropped_duplicate_ids == 0, (
        "records were dropped as duplicates despite unique AED_IDs"
    )
    assert report.colocated_records > 0, "expected co-located devices in a dense town"
    assert frame.geometry.duplicated().any(), "co-located records did not survive"


@pytest.mark.gate
def test_postal_repair_is_confirmed_by_independent_field(aed_town):
    """The zero-pad repair is verified against the AED_ID prefix, not merely inferred."""
    frame, report = aed_town
    result = validate_postal_codes(frame)
    assert result.agreement_rate == 1.0, (
        f"cross-field postal disagreement: {result.mismatches.to_dict('records')[:5]}"
    )
    assert result.repaired_and_confirmed >= 0
    assert len(result.mismatches) == 0
    # Every repair the loader made must be one this check vouches for.
    assert result.repaired_and_confirmed <= report.postal_codes_zero_padded


@pytest.mark.gate
def test_colocation_report_is_not_a_defect_claim(aed_town):
    """Co-location is a modelling fact; the report must expose devices per site."""
    frame, _ = aed_town
    report = find_colocated_sites(frame)
    assert report.n_multi_device_sites > 0
    assert (report.sites["devices"] >= 2).all()
    assert report.n_records_colocated <= len(frame)


def test_postal_validation_detects_a_planted_mismatch(aed_town):
    """The check must be capable of failing, not vacuously true."""
    frame, _ = aed_town
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[0], "POSTAL_CODE"] = "999999"
    result = validate_postal_codes(corrupted)
    assert result.agreement_rate < 1.0
    assert len(result.mismatches) == 1
