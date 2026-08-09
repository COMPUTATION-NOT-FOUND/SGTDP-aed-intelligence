"""Data-contract gates for the AED registry and the study-area boundary."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from aed_siting.config import (
    ADDRESS_FIELDS,
    AED_PATH,
    AED_REQUIRED_FIELDS,
    BANNED_FIELDS,
    CRS_SVY21,
    CRS_WGS84,
    POSTAL_CODE_LENGTH,
    ROOT,
    threshold_m,
)
from aed_siting.io.aed import parse_floor_penalty


@pytest.mark.gate
def test_schema_16_attributes():
    """All 16 attributes named in the brief must be present in the frozen file."""
    raw = gpd.read_file(AED_PATH)
    missing = [c for c in AED_REQUIRED_FIELDS if c not in raw.columns]
    assert not missing, f"missing published attributes: {missing}"
    assert len(AED_REQUIRED_FIELDS) == 16


@pytest.mark.gate
def test_address_fields_read_as_str(aed_town):
    """Address-like fields must never be numeric — that is how zeros are lost."""
    frame, _ = aed_town
    for column in ADDRESS_FIELDS:
        assert not pd.api.types.is_numeric_dtype(frame[column]), (
            f"{column} is numeric; leading zeros and nulls are at risk"
        )


@pytest.mark.gate
def test_postal_code_zero_pad_documented(aed_town):
    """Padding is an inferred repair, so it must be counted and reported."""
    frame, report = aed_town
    present = frame["POSTAL_CODE"].dropna()
    assert (present.str.len() == POSTAL_CODE_LENGTH).all(), "postal codes remain short after repair"
    assert report.postal_codes_zero_padded >= 0
    assert any("Left-padded" in t for t in report.as_transformations())


@pytest.mark.gate
def test_banned_fields_never_read(aed_town):
    """Fields with no authoritative data dictionary must not survive loading.

    The brief forbids assigning operational meaning to INC_CRC or FMEL_UPD_D.
    XVAL/YVAL were verified to be SVY21 but are dropped on the same principle:
    geometry comes from LATITUDE/LONGITUDE only.
    """
    frame, _ = aed_town
    leaked = BANNED_FIELDS.intersection(frame.columns)
    assert not leaked, f"banned fields reached the pipeline: {sorted(leaked)}"


@pytest.mark.gate
def test_crs_agreement_xval_yval():
    """Verify XVAL/YVAL are SVY21 — recorded as a finding, never depended upon.

    This test reads the raw file directly *because* the loader drops these fields.
    It documents the inference; it does not license using them.
    """
    raw = gpd.read_file(AED_PATH).head(3000)
    sub = raw.dropna(subset=["LATITUDE", "LONGITUDE", "XVAL", "YVAL"])
    projected = gpd.GeoSeries(
        gpd.points_from_xy(sub["LONGITUDE"].astype(float), sub["LATITUDE"].astype(float)),
        crs=CRS_WGS84,
    ).to_crs(CRS_SVY21)
    offset = np.hypot(
        projected.x.values - sub["XVAL"].astype(float).values,
        projected.y.values - sub["YVAL"].astype(float).values,
    )
    assert np.median(offset) < 1.0, (
        f"XVAL/YVAL are not SVY21 as assumed (median offset {np.median(offset):.2f} m); "
        "the projection assumption documented in docs/sources.md must be corrected"
    )


@pytest.mark.gate
def test_planning_area_selects_one_polygon(boundary):
    assert len(boundary) == 1
    assert boundary.crs.to_epsg() == CRS_SVY21
    area_km2 = boundary.geometry.area.sum() / 1e6
    assert 7.0 < area_km2 < 10.0, f"Toa Payoh area implausible: {area_km2:.2f} km^2"


@pytest.mark.gate
def test_town_clip_nonempty(aed_town, boundary):
    frame, report = aed_town
    assert len(frame) > 0, "no AEDs inside the study area"
    assert report.records_in > len(frame), "clip did not reduce the all-Singapore set"
    polygon = boundary.union_all()
    assert frame.geometry.within(polygon).all(), "a retained AED lies outside the town"


@pytest.mark.gate
def test_dedup_documented(aed_town):
    frame, report = aed_town
    assert frame["AED_ID"].is_unique, "duplicate AED_IDs survived"
    assert report.dropped_duplicate_ids >= 0
    assert any("share a coordinate" in t for t in report.as_transformations())


@pytest.mark.gate
def test_primary_metric_prenominated():
    """The brief requires nomination before final evaluation."""
    path = ROOT / "docs" / "primary_metric.md"
    assert path.exists(), "primary metric was not pre-nominated"
    text = path.read_text()
    assert "Demand-weighted coverage" in text
    assert "3-minute round-trip" in text


@pytest.mark.gate
def test_threshold_derives_from_time_budget():
    """126 m one-way for a 3-minute round trip, with no double-counted detour."""
    assert threshold_m(3.0) == pytest.approx(126.0)
    assert threshold_m(2.0) == pytest.approx(84.0)
    assert threshold_m(4.0) == pytest.approx(168.0)


@pytest.mark.gate
def test_floor_penalty_ground_is_free_and_unparseable_is_conservative():
    ground, ok = parse_floor_penalty("1")
    assert ground == 0.0 and ok

    upper, ok = parse_floor_penalty("5")
    assert upper > 0 and ok

    basement, ok = parse_floor_penalty("B1")
    assert basement > 0 and ok, "basements need the lift allowance too"

    unknown, ok = parse_floor_penalty("mezzanine-ish")
    assert unknown > 0 and not ok, "unparseable floors must be penalised and flagged"
