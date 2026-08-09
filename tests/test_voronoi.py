"""Gates on the Voronoi layout diagnostics."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from aed_siting.config import CRS_SVY21
from aed_siting.voronoi import build_cells, gini, largest_empty_circle, measure


@pytest.fixture(scope="module")
def square():
    """A 1 km square study area."""
    return gpd.GeoDataFrame({"n": [1]}, geometry=[box(0, 0, 1000, 1000)], crs=CRS_SVY21)


@pytest.fixture(scope="module")
def corners():
    """Four devices, one near each corner."""
    return gpd.GeoDataFrame(
        {"label": ["a", "b", "c", "d"]},
        geometry=[Point(100, 100), Point(900, 100), Point(100, 900), Point(900, 900)],
        crs=CRS_SVY21,
    )


@pytest.mark.gate
def test_voronoi_cells_partition_the_town(corners, square):
    """Cells must tile the study area without overlapping.

    A service area that double-counts ground would double-count demand, making
    load figures meaningless.
    """
    cells = build_cells(corners, square)
    total = float(cells.geometry.area.sum())
    area = float(square.geometry.area.sum())
    assert total == pytest.approx(area, rel=0.02), "cells do not cover the area"
    merged = cells.geometry.union_all().area
    assert merged == pytest.approx(total, rel=0.02), "cells overlap"


@pytest.mark.gate
def test_cells_align_with_device_order(corners, square):
    """Row order must match the device frame so loads can be attached positionally."""
    cells = build_cells(corners, square)
    assert len(cells) == len(corners)
    assert cells["device_index"].tolist() == list(range(len(corners)))
    for index, cell in enumerate(cells.geometry):
        assert cell.contains(corners.geometry.iloc[index]), (
            "a device does not fall inside its own service area"
        )


@pytest.mark.gate
def test_largest_empty_circle_finds_the_centre(corners, square):
    """With devices at four corners, the emptiest spot is the middle.

    The centre of a 1 km square is ~566 m from a device 100 m in from each corner.
    """
    radius, point = largest_empty_circle(corners, square)
    assert point is not None
    assert point.x == pytest.approx(500, abs=60)
    assert point.y == pytest.approx(500, abs=60)
    assert radius == pytest.approx(566, rel=0.1)


@pytest.mark.gate
def test_largest_empty_circle_shrinks_when_the_gap_is_filled(corners, square):
    """Placing a device in the gap must reduce the worst-served distance."""
    before, _ = largest_empty_circle(corners, square)
    filled = gpd.GeoDataFrame(
        {"label": [*corners["label"], "centre"]},
        geometry=[*corners.geometry, Point(500, 500)],
        crs=CRS_SVY21,
    )
    after, _ = largest_empty_circle(filled, square)
    assert after < before, "filling the largest gap did not reduce the worst case"


@pytest.mark.gate
def test_load_gini_bounds():
    """Gini must be 0 for even load and approach 1 as load concentrates."""
    assert gini(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    assert gini(np.zeros(4)) == 0.0
    concentrated = gini(np.array([0.0, 0.0, 0.0, 100.0]))
    assert 0.7 < concentrated < 1.0
    assert gini(np.array([1.0, 2.0, 3.0, 4.0])) > 0.0


@pytest.mark.gate
def test_measure_reports_a_complete_layout_score(corners, square):
    cells, metrics = measure(corners, square, np.array([0.4, 0.3, 0.2, 0.1]))
    assert metrics.devices == 4
    assert metrics.largest_empty_circle_m > 0
    assert 0.0 <= metrics.load_gini <= 1.0
    assert metrics.area_spread >= 1.0
    assert cells["load"].sum() == pytest.approx(1.0)
    assert "worst-served point" in metrics.summary()


@pytest.mark.gate
def test_voronoi_artifacts_are_consistent(town_artifacts):
    """The shipped artifacts must agree with the run summary."""
    metrics = town_artifacts["voronoi_metrics"]
    summary = town_artifacts["summary"]
    assert set(metrics["stage"]) == {"before", "after"}
    before = metrics[metrics["stage"] == "before"].iloc[0]
    assert before["devices"] == summary["existing_aeds"]
    assert before["largest_empty_circle_m"] == pytest.approx(
        summary["worst_radius_before"], abs=0.5
    )
    cells = town_artifacts["voronoi_before"]
    assert cells["load"].sum() == pytest.approx(1.0, abs=0.01)


def test_fewer_than_four_devices_does_not_crash(square):
    """Voronoi is undefined below four points; the fallback must still return cells."""
    tiny = gpd.GeoDataFrame({"label": ["only"]}, geometry=[Point(500, 500)], crs=CRS_SVY21)
    cells = build_cells(tiny, square)
    assert len(cells) == 1
    assert cells.geometry.iloc[0].area > 0
