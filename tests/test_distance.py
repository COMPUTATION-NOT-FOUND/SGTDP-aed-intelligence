"""Grid, distance-model and hours gates."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest

from aed_siting.config import CRS_SVY21, CRS_WGS84, DETOUR_FACTOR, threshold_m
from aed_siting.distance import UNIT, DetourEuclidean, DistanceModel, nearest_distance
from aed_siting.grid import build_grid
from aed_siting.hours import always_open_mask, bucket_hours, hours_report


@pytest.fixture(scope="module")
def grid(boundary):
    return build_grid(boundary)


# --- grid ------------------------------------------------------------------


@pytest.mark.gate
def test_grid_covers_town_without_gross_overshoot(grid, boundary):
    """Tiling must cover the town closely; boundary cells overhang symmetrically."""
    assert len(grid) > 300
    assert grid.crs.to_epsg() == CRS_SVY21
    tiled = gpd.GeoSeries(grid["hex_geometry"], crs=CRS_SVY21).area.sum() / 1e6
    town = boundary.geometry.area.sum() / 1e6
    assert 0.95 < tiled / town < 1.15, f"tiled {tiled:.2f} km² vs town {town:.2f} km²"


@pytest.mark.gate
def test_grid_is_deterministic(boundary):
    """Same boundary, same cells, same order — required by the determinism gate."""
    assert build_grid(boundary)["h3"].tolist() == build_grid(boundary)["h3"].tolist()


# --- distance model --------------------------------------------------------


@pytest.mark.gate
def test_detour_euclidean_satisfies_protocol():
    assert isinstance(DetourEuclidean(), DistanceModel)
    assert DetourEuclidean.unit == UNIT


@pytest.mark.gate
def test_distance_matrix_detour_ratio(grid, aed_town):
    """Matrix ÷ raw straight-line must equal the factor exactly, never its square.

    This is the gate for the double-counted-detour defect: if the factor were
    applied twice the ratio would be 1.8225, and if it were also folded into the
    threshold the effective radius would be 35% too tight.
    """
    frame, _ = aed_town
    matrix = DetourEuclidean().matrix(grid, frame)
    origins = np.c_[grid.geometry.x, grid.geometry.y]
    dests = np.c_[frame.geometry.x, frame.geometry.y]
    raw = np.linalg.norm(origins[:, None, :] - dests[None, :, :], axis=-1)
    ratio = matrix[raw > 0] / raw[raw > 0]
    assert np.allclose(ratio, DETOUR_FACTOR), f"detour ratio {ratio.mean():.4f}"


@pytest.mark.gate
def test_distance_matrix_is_one_way(grid, aed_town, boundary):
    """No round-trip doubling may be baked into the matrix."""
    frame, _ = aed_town
    matrix = DetourEuclidean().matrix(grid, frame)
    bounds = boundary.total_bounds
    diameter = float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    assert matrix.max() < diameter * DETOUR_FACTOR * 1.05, "matrix looks round-trip"
    assert matrix.shape == (len(grid), len(frame))


@pytest.mark.gate
def test_distance_refuses_degrees(grid, aed_town):
    """Working in 4326 must fail loudly rather than silently produce degrees."""
    frame, _ = aed_town
    with pytest.raises(ValueError, match="EPSG:3414"):
        DetourEuclidean().matrix(grid.to_crs(CRS_WGS84), frame)


@pytest.mark.gate
def test_detour_factor_cannot_beat_straight_line():
    with pytest.raises(ValueError):
        DetourEuclidean(factor=0.9)


def test_nearest_distance_applies_penalties(grid, aed_town):
    """A penalty must be able to make a nearer site lose to a further one."""
    frame, _ = aed_town
    plain = nearest_distance(grid, frame)
    penalised = nearest_distance(grid, frame, penalties_m=np.full(len(frame), 50.0))
    assert np.all(penalised >= plain - 1e-9)
    assert np.allclose(penalised, plain + 50.0)


def test_nearest_distance_with_no_destinations_is_infinite(grid, aed_town):
    frame, _ = aed_town
    result = nearest_distance(grid, frame.iloc[:0])
    assert np.isinf(result).all(), "no sites must mean no coverage, not zero distance"


# --- hours -----------------------------------------------------------------


@pytest.mark.gate
def test_hours_buckets_exhaustive(aed_town):
    """Every record lands in exactly one known bucket."""
    frame, _ = aed_town
    report = hours_report(frame)
    assert report.total == len(frame)
    assert set(report.counts.index) <= {
        "always_open",
        "extended",
        "daytime",
        "closed",
        "unparseable",
    }
    assert report.counts.sum() == len(frame)


@pytest.mark.gate
def test_unparseable_hours_excluded_not_guessed():
    """Unrecognised input must abstain, and must never count as 24/7."""
    for value in ("", None, "nan", "whenever the caretaker is around"):
        assert bucket_hours(value) == "unparseable"


@pytest.mark.gate
def test_always_open_is_conservative(aed_town):
    """Only positively-identified 24/7 devices may enter the night scenario."""
    frame, _ = aed_town
    mask = always_open_mask(frame)
    assert mask.sum() > 0
    assert mask.sum() < len(frame), "not every device can be 24/7"
    stated = frame.loc[mask.to_numpy(), "OPERATING_HOURS"].astype(str)
    assert stated.str.contains("00:00").all()


@pytest.mark.gate
def test_hours_parse_rate_reported(aed_town):
    """The parse rate is the nominated safety/error metric, so it must exist."""
    frame, _ = aed_town
    report = hours_report(frame)
    assert 0.0 <= report.parse_rate <= 1.0
    assert "parse rate" in report.summary()


def test_bucket_hours_known_shapes():
    assert bucket_hours("Mon - Sun 00:00-23:59;") == "always_open"
    assert bucket_hours("Mon - Sun Closed;") == "closed"
    assert bucket_hours("Mon - Sun 07:00-17:00;") == "daytime"
    assert bucket_hours("Mon - Sun 05:30-23:59;") == "extended"


def test_night_coverage_is_never_higher_than_all_day(grid, aed_town):
    """Removing devices cannot increase coverage — a monotonicity sanity check."""
    frame, _ = aed_town
    mask = always_open_mask(frame).to_numpy()
    limit = threshold_m(3.0)
    all_day = (nearest_distance(grid, frame) <= limit).mean()
    night = (nearest_distance(grid, frame[mask]) <= limit).mean()
    assert night <= all_day + 1e-12
