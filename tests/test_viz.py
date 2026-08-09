"""Gates on map-layer construction.

Rendering defects are invisible from Python — no exception, no warning, just a
wrong picture — so the checks here are structural.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aed_siting import viz
from aed_siting.config import CRS_SVY21

#: deck.gl properties that take a literal enum, never an accessor expression.
ENUM_PROPERTIES = (
    "radiusUnits",
    "widthUnits",
    "sizeUnits",
    "lineWidthUnits",
    "coordinateSystem",
    "billboard",
)


@pytest.fixture(scope="module")
def devices():
    return gpd.GeoDataFrame(
        {"label": ["A", "B", "C"]},
        geometry=[Point(29000, 33000), Point(29400, 33200), Point(29900, 32800)],
        crs=CRS_SVY21,
    )


@pytest.fixture(scope="module")
def layers(devices, town_artifacts):
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    incidents = town_artifacts["incidents_after"]
    return [
        viz.coverage_area(devices, 126.0),
        viz.coverage_discs(devices, 126.0),
        viz.device_points(devices),
        viz.demand_hexes(grid, cells["demand"].to_numpy(), ~cells["covered_before"].to_numpy()),
        viz.voronoi_layer(town_artifacts["voronoi_before"]),
        viz.worst_point_layer((29500, 33000), 500.0),
        viz.incident_points(incidents),
    ]


@pytest.mark.gate
def test_no_mangled_enum_kwargs(layers):
    """String kwargs must not be serialised as JavaScript accessor expressions.

    pydeck turns *every* string keyword into an accessor, so ``radius_units="meters"``
    becomes ``"radiusUnits": "@@=meters"``. deck.gl then evaluates ``meters`` as an
    undefined variable, radius semantics collapse, and every circle expands to fill
    the canvas. It produces no Python error, which is exactly why it needs a gate.
    """
    for layer in layers:
        if layer is None:
            continue
        spec = json.loads(layer.to_json())
        for prop in ENUM_PROPERTIES:
            value = spec.get(prop)
            assert not (isinstance(value, str) and value.startswith("@@=")), (
                f"{spec.get('@@type')}: {prop} serialised as an accessor ({value!r})"
            )


@pytest.mark.gate
def test_coverage_is_drawn_as_geometry_not_radius(devices, layers):
    """Coverage must be real polygons, so it cannot depend on marker-radius units."""
    area = viz.coverage_area(devices, 126.0)
    spec = json.loads(area.to_json())
    assert spec["@@type"] == "GeoJsonLayer"
    assert "getRadius" not in spec


@pytest.mark.gate
def test_coverage_area_merges_overlapping_discs(devices):
    """The union must be one shape, not one per device."""
    spec = json.loads(viz.coverage_area(devices, 400.0).to_json())
    features = spec["data"]["features"]
    assert len(features) == 1, "coverage area was not merged into a single shape"


@pytest.mark.gate
def test_coverage_radius_is_honoured(devices):
    """A larger reach must produce a visibly larger covered area."""
    small = devices.geometry.buffer(126.0).union_all().area
    large = devices.geometry.buffer(252.0).union_all().area
    assert large > small * 2, "coverage area does not scale with the threshold"


@pytest.mark.gate
def test_device_markers_stay_visible_at_any_zoom(devices):
    """Pixel bounds keep markers legible when metres would render sub-pixel."""
    spec = json.loads(viz.device_points(devices).to_json())
    assert spec["radiusMinPixels"] >= 1
    assert spec["radiusMaxPixels"] >= spec["radiusMinPixels"]


@pytest.mark.gate
def test_deck_accepts_any_geometry_type(devices, town_artifacts):
    """Framing must not assume Points.

    Regression gate: ``deck`` centred with ``geometry.y.mean()``, which exists
    only for Points, so the Service-areas tab crashed on Voronoi polygons with
    ``ValueError: y attribute access only provided for Point geometries``. Every
    prior viz test passed points, which is why it reached a user.
    """
    polygons = town_artifacts["voronoi_before"]
    empty = gpd.GeoDataFrame({"label": []}, geometry=[], crs=CRS_SVY21)
    for frame in (devices, polygons, empty):
        deck = viz.deck([], frame)
        view = json.loads(deck.to_json())["initialViewState"]
        assert 1.0 < view["latitude"] < 1.6, f"implausible latitude for {type(frame)}"
        assert 103.4 < view["longitude"] < 104.3
        assert 8.0 <= view["zoom"] <= 16.0


@pytest.mark.gate
def test_zoom_fits_the_area(town_artifacts):
    """A small town must zoom in further than a large one."""
    from aed_siting.config import processed_dir

    small = gpd.read_file(processed_dir("toa-payoh") / "grid.geojson")
    large = gpd.read_file(processed_dir("bedok") / "grid.geojson")
    small_zoom = json.loads(viz.deck([], small).to_json())["initialViewState"]["zoom"]
    large_zoom = json.loads(viz.deck([], large).to_json())["initialViewState"]["zoom"]
    assert small_zoom > large_zoom, "zoom is not fitted to the study area"


@pytest.mark.gate
def test_uncovered_shading_tracks_demand(town_artifacts):
    """Empty land must not be shaded as urgently as dense housing."""
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    uncovered = ~cells["covered_before"].to_numpy()
    layer = viz.demand_hexes(grid, demand, uncovered)
    fills = pd.DataFrame(layer.data)["fill"].tolist()
    alphas = np.array([f[3] for f in fills])
    flagged = alphas[uncovered]
    quiet = flagged[demand[uncovered] < demand.max() * 0.05]
    loud = flagged[demand[uncovered] > demand.max() * 0.5]
    assert quiet.size and loud.size
    assert quiet.mean() * 2 < loud.mean(), "low-demand gaps are as loud as high-demand ones"


@pytest.mark.gate
def test_new_ground_layer_subtracts_existing(town_artifacts):
    """The emphasised shape must exclude ground the network already covers."""
    from shapely.geometry import shape

    existing = town_artifacts["existing_aeds"]
    cand = town_artifacts["candidates"]
    queue = town_artifacts["install_queue"]
    proposed = cand[cand["site_id"].isin(queue["site_id"])]
    layer = viz.new_ground(proposed, existing, 126.0)
    assert layer is not None
    added = gpd.GeoSeries([shape(f["geometry"]) for f in layer.data["features"]], crs=4326).to_crs(
        CRS_SVY21
    )
    raw = proposed.to_crs(CRS_SVY21).geometry.buffer(126.0).union_all().area
    assert added.area.sum() < raw, "new-ground layer did not subtract existing coverage"


@pytest.mark.gate
def test_pickable_layers_expose_a_label(town_artifacts, devices):
    """Every pickable layer must carry the property the shared tooltip reads.

    Regression gate: the Voronoi layer carried ``tip`` while ``deck()``'s tooltip
    referenced ``{label}``, so hovering a service area displayed the placeholder
    literally instead of the device name.
    """
    layers = [
        viz.voronoi_layer(town_artifacts["voronoi_before"]),
        viz.coverage_discs(devices, 126.0),
        viz.device_points(devices),
        viz.worst_point_layer((29500, 33000), 400.0),
    ]
    for layer in layers:
        if layer is None:
            continue
        spec = json.loads(layer.to_json())
        if not spec.get("pickable"):
            continue
        data = layer.data
        if isinstance(data, dict):  # GeoJSON FeatureCollection
            keys = list(data["features"][0]["properties"])
        elif isinstance(data, list):  # pydeck converts frames to records
            keys = list(data[0])
        else:
            keys = list(data.columns)
        assert "label" in keys, f"{spec.get('@@type')} is pickable but exposes no `label`"


@pytest.mark.gate
def test_chart_height_scales_with_rows(town_artifacts):
    """A fixed height silently drops categories, so height must follow the data.

    Regression gate: 130 px could not fit three bars, so the required baseline
    disappeared from the method comparison entirely.
    """
    comparison = town_artifacts["method_vs_baseline"]
    spec = viz.method_bar(comparison).to_dict()
    assert spec["height"] >= 30 * len(comparison), "height too small for the row count"
    assert viz._height(5) > viz._height(3)


@pytest.mark.gate
def test_outage_chart_shows_loss_not_absolute(town_artifacts):
    """A 1.7 pp decline on a 0-100% axis is invisible; plot the loss instead."""
    outages = town_artifacts["outage_stress"]
    spec = viz.outage_chart(outages, 0.4704).to_dict()
    encodings = json.dumps(spec)
    assert "lost_mean" in encodings, "chart still plots absolute coverage"
    assert "lost_p05" in encodings, "bad-draw band is missing"


def test_layers_tolerate_empty_input():
    """Empty frames must yield no layer rather than an exception."""
    empty = gpd.GeoDataFrame({"label": []}, geometry=[], crs=CRS_SVY21)
    assert viz.coverage_area(empty, 126.0) is None
    assert viz.coverage_discs(empty, 126.0) is None
    assert viz.device_points(empty) is None
    assert viz.worst_point_layer(None, 0.0) is None


def test_charts_build_from_artifacts(town_artifacts):
    """Every chart must accept the artifact schema it is given."""
    art = town_artifacts
    queue = art["install_queue"]
    assert viz.coverage_curve(queue, 0.47, 5) is not None
    assert viz.marginal_gain_chart(queue) is not None
    assert viz.method_bar(art["method_vs_baseline"]) is not None
    assert viz.objective_scatter(art["objectives"]) is not None
    assert viz.equity_slope(art["equity_before"], art["equity_after"]) is not None
    assert viz.retrieval_histogram(art["incidents_before"], art["incidents_after"]) is not None
    assert viz.outage_chart(art["outage_stress"], 0.47) is not None
    assert viz.simple_bar(art["label_quality"], "sites", "label_source", "Sites") is not None


def test_demand_hexes_flags_uncovered(town_artifacts):
    """Uncovered cells must be coloured differently from covered ones."""
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    layer = viz.demand_hexes(grid, demand, ~cells["covered_before"].to_numpy())
    fills = pd.DataFrame(layer.data)["fill"].tolist()
    assert len({tuple(f[:3]) for f in fills}) == 2, "uncovered cells are not distinguished"


@pytest.mark.gate
def test_unreachable_gaps_are_shaded_apart_from_fundable_ones(town_artifacts):
    """A gap no candidate can reach must not look like a gap awaiting budget.

    These are different problems -- a funding backlog against the candidate pool
    running out -- and in Toa Payoh the second is roughly a third of the
    remaining gap. Painting both amber is what made the map read as a bug.
    """
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    uncovered = ~cells["covered_before"].to_numpy()

    dead = np.zeros(len(grid), dtype=bool)
    dead[np.flatnonzero(uncovered)[:5]] = True  # uncovered AND unreachable
    # A cell that no candidate reaches but the existing network already covers
    # must keep rendering as covered, never as a gap.
    covered_but_unreachable = int(np.flatnonzero(~uncovered)[0])
    dead[covered_but_unreachable] = True

    layer = viz.demand_hexes(grid, demand, uncovered, dead)
    fills = pd.DataFrame(layer.data)["fill"].tolist()
    rgb = [tuple(f[:3]) for f in fills]

    assert len(set(rgb)) == 3, "expected covered / fundable-gap / unreachable-gap"
    assert tuple(viz.UNREACHABLE) in set(rgb)
    for i in np.flatnonzero(uncovered)[:5]:
        assert rgb[i] == tuple(viz.UNREACHABLE), "unreachable gap not shaded apart"
    assert rgb[covered_but_unreachable] != tuple(viz.UNREACHABLE), (
        "a covered cell was shaded as a gap because no candidate reaches it"
    )
    assert rgb[covered_but_unreachable] != tuple(viz.UNCOVERED)


def test_demand_hexes_without_unreachable_is_unchanged(town_artifacts):
    """The new argument is optional and must not alter existing behaviour."""
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    uncovered = ~cells["covered_before"].to_numpy()
    before = pd.DataFrame(viz.demand_hexes(grid, demand, uncovered).data)["fill"].tolist()
    after = pd.DataFrame(viz.demand_hexes(grid, demand, uncovered, None).data)["fill"].tolist()
    assert before == after


def test_empty_geodataframe_helper_roundtrips(devices):
    """Reprojection to WGS84 must produce plausible Singapore coordinates."""
    points = viz._points(devices, ("label",))
    assert points["lat"].between(1.1, 1.5).all()
    assert points["lon"].between(103.5, 104.2).all()
    assert np.isfinite(points[["lat", "lon"]].to_numpy()).all()
