"""Map layers and charts.

Everything here exists to make the model's claims visible: a coverage shape is
what the threshold *means*, a choropleth is where demand actually is, and a
Voronoi cell is who serves whom.

**Coverage is drawn as real geometry, not as a marker radius.** Buffering device
points in EPSG:3414 and reprojecting is deterministic and composes with the other
polygon layers. It also sidesteps a live trap: pydeck serialises *every* string
keyword as a JavaScript accessor, so ``radius_units="meters"`` becomes
``"radiusUnits": "@@=meters"``, deck.gl evaluates ``meters`` as an undefined
variable, and every circle expands to fill the canvas. See
``tests/test_viz.py::test_no_mangled_enum_kwargs``.

Colour is consistent throughout:

======================  =========================================
grey                    the existing network
red                     what this tool proposes
amber                   demand still uncovered, and the worst-served point
blue                    simulated incidents
======================  =========================================
"""

from __future__ import annotations

import json

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely.geometry import Point

from aed_siting.config import CRS_SVY21, CRS_WGS84


def _height(rows: int, floor: int = 140) -> int:
    """Chart height that grows with the data.

    A fixed height silently drops categories: 130 px could not fit three bars, so
    the required baseline vanished from the method comparison and two of five
    presets vanished from the policy chart.
    """
    return max(floor, 30 * int(rows) + 70)


EXISTING = [140, 145, 155]
PROPOSED = [214, 69, 65]
UNCOVERED = [222, 158, 54]
INCIDENT = [90, 155, 220]
SERVED = [88, 166, 255]
PINNED = [46, 160, 110]
VETOED = [120, 120, 128]
#: A gap no candidate can reach. Desaturated on purpose -- it is still a gap, but
#: nothing in the budget can act on it, so it must not compete for attention with
#: the amber gaps a planner can actually close.
UNREACHABLE = [148, 132, 116]


def _geojson(frame: gpd.GeoDataFrame) -> dict:
    """Reproject to WGS84 and serialise for a deck.gl GeoJsonLayer."""
    return json.loads(frame.to_crs(CRS_WGS84).to_json())


def _points(frame: gpd.GeoDataFrame, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Project to lat/long and keep only what a tooltip needs."""
    wgs84 = frame.to_crs(CRS_WGS84)
    out = pd.DataFrame({"lon": wgs84.geometry.x, "lat": wgs84.geometry.y})
    for column in keep:
        if column in frame.columns:
            out[column] = frame[column].astype(str).to_numpy()
    return out.reset_index(drop=True)


# --- coverage --------------------------------------------------------------


def coverage_area(devices: gpd.GeoDataFrame, radius_m: float, colour=EXISTING) -> pdk.Layer | None:
    """The union of everything the existing network covers, as one shape.

    Merging the discs is the point: 284 overlapping rings read as noise, whereas
    a single served area reads as "here is what is covered, and here are the
    holes" — which is the question the tool answers.
    """
    if devices is None or len(devices) == 0:
        return None
    merged = devices.to_crs(CRS_SVY21).geometry.buffer(radius_m).union_all()
    frame = gpd.GeoDataFrame({"kind": ["covered"]}, geometry=[merged], crs=CRS_SVY21)
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame),
        filled=True,
        stroked=True,
        get_fill_color=[*colour, 55],
        get_line_color=[*colour, 170],
        line_width_min_pixels=1,
        pickable=False,
    )


def coverage_discs(devices: gpd.GeoDataFrame, radius_m: float, colour=PROPOSED) -> pdk.Layer | None:
    """One shape per proposed device, so each addition's contribution is visible."""
    if devices is None or len(devices) == 0:
        return None
    projected = devices.to_crs(CRS_SVY21)
    frame = gpd.GeoDataFrame(
        {"label": projected.get("label", pd.Series(range(len(projected)))).astype(str)},
        geometry=projected.geometry.buffer(radius_m),
        crs=CRS_SVY21,
    )
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame),
        filled=True,
        stroked=True,
        get_fill_color=[*colour, 70],
        get_line_color=[*colour, 220],
        line_width_min_pixels=2,
        pickable=True,
    )


def new_ground(
    proposed: gpd.GeoDataFrame, existing: gpd.GeoDataFrame, radius_m: float
) -> pdk.Layer | None:
    """Only the ground the proposals add, with existing coverage subtracted.

    Overlapping translucent discs read as waste even when they are not: measured,
    the ten proposals overlap each other by 2.3% and 87.9% of their combined reach
    is new. Emphasising the difference makes that visible instead of asserted.
    """
    if proposed is None or len(proposed) == 0:
        return None
    added = proposed.to_crs(CRS_SVY21).geometry.buffer(radius_m).union_all()
    if existing is not None and len(existing):
        added = added.difference(existing.to_crs(CRS_SVY21).geometry.buffer(radius_m).union_all())
    if added.is_empty:
        return None
    frame = gpd.GeoDataFrame({"label": ["New ground"]}, geometry=[added], crs=CRS_SVY21)
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame),
        filled=True,
        stroked=True,
        get_fill_color=[*PROPOSED, 120],
        get_line_color=[*PROPOSED, 245],
        line_width_min_pixels=2,
        pickable=True,
    )


def device_points(
    devices: gpd.GeoDataFrame, colour=EXISTING, radius_m: float = 25
) -> pdk.Layer | None:
    """The devices themselves, drawn over their coverage.

    ``radius_units`` is deliberately not passed: metres is deck.gl's default, and
    passing the string would be mangled into an accessor expression.
    """
    if devices is None or len(devices) == 0:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        _points(devices, ("label",)),
        get_position="[lon, lat]",
        get_fill_color=[*colour, 235],
        get_radius=radius_m,
        radius_min_pixels=3,
        radius_max_pixels=14,
        pickable=True,
    )


def demand_hexes(
    grid: gpd.GeoDataFrame,
    demand: np.ndarray,
    uncovered: np.ndarray | None = None,
    unreachable: np.ndarray | None = None,
) -> pdk.Layer:
    """Demand as a choropleth, with uncovered cells picked out in amber.

    Args:
        grid: One row per H3 cell, aligned with ``demand``.
        demand: Per-cell demand weight.
        uncovered: Cells not covered at the budget being drawn.
        unreachable: Cells **no candidate can reach at all**. Where these overlap
            ``uncovered`` the cell is drawn inert rather than amber.

    Painting both kinds of gap the same amber is what made the map look broken:
    a viewer cannot tell "fund more devices" from "no eligible building exists
    here", and the second is a third of the gap in Toa Payoh. Note the ``&``
    below -- a cell no candidate reaches may still be covered by an *existing*
    device, and must keep rendering as covered.
    """
    weights = np.asarray(demand, dtype=float)
    # Square-root scaling widens the visible range: linear alpha leaves the
    # long tail of small cells indistinguishable, so a park read as urgently as
    # a housing block. Intensity is now how many people, not how much land.
    peak = weights.max()
    scale = np.sqrt(weights / peak) if peak > 0 else weights
    flagged = (
        np.zeros(len(grid), dtype=bool) if uncovered is None else np.asarray(uncovered, dtype=bool)
    )
    inert = (
        np.zeros(len(grid), dtype=bool)
        if unreachable is None
        else np.asarray(unreachable, dtype=bool) & flagged
    )

    def _fill(s: float, flag: bool, dead: bool) -> list[int]:
        if dead:
            return [*UNREACHABLE, int(10 + 150 * s)]
        if flag:
            return [*UNCOVERED, int(12 + 228 * s)]
        return [110, 130, 170, int(6 + 70 * s)]

    frame = pd.DataFrame(
        {
            "h3": grid["h3"].astype(str),
            "fill": [
                _fill(s, flag, dead) for s, flag, dead in zip(scale, flagged, inert, strict=True)
            ],
        }
    )
    return pdk.Layer(
        "H3HexagonLayer",
        frame,
        get_hexagon="h3",
        get_fill_color="fill",
        extruded=False,
        stroked=False,
        pickable=False,
    )


def flash_hexes(grid: gpd.GeoDataFrame, newly: np.ndarray) -> pdk.Layer | None:
    """Cells gained on this animation step, drawn bright.

    Makes the causal link visible: without it a viewer sees a device appear and
    a number change, but not which ground the device actually won.
    """
    mask = np.asarray(newly, dtype=bool)
    if not mask.any():
        return None
    return pdk.Layer(
        "H3HexagonLayer",
        pd.DataFrame({"h3": grid.loc[mask, "h3"].astype(str)}),
        get_hexagon="h3",
        get_fill_color=[120, 220, 140, 235],
        extruded=False,
        stroked=True,
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=2,
        pickable=False,
    )


# --- voronoi ---------------------------------------------------------------


def voronoi_layer(cells: gpd.GeoDataFrame) -> pdk.Layer | None:
    """Service areas shaded by how much demand each device is nearest to.

    Dark cells are over-subscribed devices; pale cells are redundant ones. This
    is catchment load, which the pipeline has computed since Phase 3 and which
    has never been visible until now.
    """
    if cells is None or len(cells) == 0:
        return None
    frame = cells[~cells.geometry.is_empty].copy()
    load = frame["load"].to_numpy(dtype=float)
    scale = load / load.max() if load.max() > 0 else load
    frame["fill"] = [[196, 74, 62, int(35 + 165 * s)] for s in scale]
    # Property must be named `label`: deck()'s shared tooltip reads {label}, and a
    # mismatch renders the placeholder literally instead of the device name.
    frame["label"] = [
        f"{name} — serves {value:.2%} of demand"
        for name, value in zip(frame["label"].astype(str), load, strict=True)
    ]
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame[["fill", "label", "geometry"]]),
        filled=True,
        stroked=True,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
        pickable=True,
    )


def worst_point_layer(point_xy: tuple[float, float] | None, radius_m: float) -> pdk.Layer | None:
    """Mark the largest empty circle — the worst-served spot in the study area."""
    if point_xy is None:
        return None
    frame = gpd.GeoDataFrame(
        {"label": [f"Worst-served point — {radius_m:.0f} m from any device"]},
        geometry=[Point(point_xy).buffer(radius_m)],
        crs=CRS_SVY21,
    )
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame),
        filled=False,
        stroked=True,
        get_line_color=[*UNCOVERED, 245],
        line_width_min_pixels=3,
        pickable=True,
    )


def criticality_points(
    existing: gpd.GeoDataFrame, criticality: pd.DataFrame, top: int = 40
) -> pdk.Layer | None:
    """Devices that are the *only* cover for some demand, sized by how much.

    "How fragile is the network on average" is answered by the random-outage
    curve; "which devices in particular" is answered here, and it is a different
    and more actionable question. These are single points of failure — losing one
    strands demand nothing else reaches.
    """
    if criticality is None or existing is None or criticality.empty:
        return None
    ranked = criticality[criticality["sole_cover_demand"] > 0].nlargest(top, "sole_cover_demand")
    if ranked.empty:
        return None
    merged = existing.merge(ranked[["AED_ID", "sole_cover_demand"]], on="AED_ID", how="inner")
    if merged.empty:
        return None
    peak = float(merged["sole_cover_demand"].max()) or 1.0
    frame = _points(merged, ("label",))
    frame["radius"] = 40 + 260 * (merged["sole_cover_demand"].to_numpy() / peak)
    frame["label"] = [
        f"{name} — sole cover for {value:.2%} of demand"
        for name, value in zip(merged["label"], merged["sole_cover_demand"], strict=True)
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        frame,
        get_position="[lon, lat]",
        get_fill_color=[*UNCOVERED, 205],
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=26,
        pickable=True,
    )


def subzone_choropleth(subzones: gpd.GeoDataFrame, column: str) -> pdk.Layer | None:
    """Subzones shaded by coverage on a shared red-to-green scale.

    Replaces a twelve-line slope chart that no one could read. Colour is fixed to
    an absolute 0-100% scale so before and after are directly comparable — a
    per-panel scale would make an unchanged subzone look different between them.
    """
    if subzones is None or len(subzones) == 0 or column not in subzones.columns:
        return None
    frame = subzones.copy()
    value = frame[column].to_numpy(dtype=float).clip(0, 1)
    frame["fill"] = [[int(214 - 150 * v), int(70 + 110 * v), int(60 + 40 * v), 205] for v in value]
    return pdk.Layer(
        "GeoJsonLayer",
        _geojson(frame[["fill", "label", "geometry"]]),
        filled=True,
        stroked=True,
        get_fill_color="properties.fill",
        get_line_color=[255, 255, 255, 90],
        line_width_min_pixels=1,
        pickable=True,
    )


def stability_points(
    candidates: gpd.GeoDataFrame, stability: pd.DataFrame, top: int = 30
) -> pdk.Layer | None:
    """Candidate sites sized by how often they survive the scenario sweep."""
    if candidates is None or stability is None or stability.empty:
        return None
    ranked = stability.nlargest(top, "share")
    merged = candidates.merge(ranked[["site_id", "share"]], on="site_id", how="inner")
    if merged.empty:
        return None
    frame = _points(merged, ("label",))
    frame["radius"] = 40 + 220 * merged["share"].to_numpy()
    frame["label"] = [
        f"{name} — chosen in {share:.0%} of scenarios"
        for name, share in zip(merged["label"], merged["share"], strict=True)
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        frame,
        get_position="[lon, lat]",
        get_fill_color=[*PROPOSED, 215],
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=22,
        pickable=True,
    )


#: Layer id the click handler looks for in the pydeck selection payload.
CANDIDATE_LAYER_ID = "candidate-sites"


def candidate_markers(
    candidates: gpd.GeoDataFrame,
    proposed_ids: tuple[str, ...] = (),
    pinned: tuple[str, ...] = (),
    vetoed: tuple[str, ...] = (),
    limit: int = 400,
) -> pdk.Layer | None:
    """Eligible sites as click targets, coloured by their role in the plan.

    Candidates are the click target rather than the demand hexes because a hex
    can contain no eligible building (nothing to pin) or several (ambiguous).
    Clicking the marker itself removes both failure modes without inventing a
    nearest-neighbour rule the planner cannot see.

    ``limit`` keeps the layer legible and the payload small; the proposed,
    pinned and vetoed sites are always kept, and the remaining slots are filled
    in candidate order so the set is deterministic.
    """
    if candidates is None or len(candidates) == 0:
        return None
    named = set(proposed_ids) | set(pinned) | set(vetoed)
    keep = candidates[candidates["site_id"].isin(named)]
    if len(keep) < limit:
        rest = candidates[~candidates["site_id"].isin(named)].head(limit - len(keep))
        keep = pd.concat([keep, rest])
    if keep.empty:
        return None

    frame = _points(keep, ("site_id", "label"))
    states, colours = [], []
    for site_id in keep["site_id"]:
        if site_id in vetoed:
            states.append("vetoed")
            colours.append([*VETOED, 170])
        elif site_id in pinned:
            states.append("pinned")
            colours.append([*PINNED, 245])
        elif site_id in proposed_ids:
            states.append("proposed")
            colours.append([*PROPOSED, 235])
        else:
            states.append("available")
            colours.append([*EXISTING, 130])
    frame["state"] = states
    frame["colour"] = colours
    frame["radius"] = [55 if s in ("pinned", "vetoed", "proposed") else 30 for s in states]
    frame["label"] = [
        f"{name} ({site}) — {state}"
        for name, site, state in zip(keep["label"], keep["site_id"], states, strict=True)
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        frame,
        # A stable id is what lets the click handler find this layer's picks in
        # the selection payload, which is keyed by layer id.
        id=CANDIDATE_LAYER_ID,
        get_position="[lon, lat]",
        get_fill_color="colour",
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=16,
        pickable=True,
        auto_highlight=True,
    )


CANDIDATE_LEGEND = [
    ("In the plan at this budget", [*PROPOSED, 235]),
    ("You pinned it — must include", [*PINNED, 245]),
    ("You vetoed it — must exclude", [*VETOED, 170]),
    ("Eligible, not currently chosen", [*EXISTING, 130]),
]


def incident_points(incidents: pd.DataFrame) -> pdk.Layer:
    """Simulated incidents, coloured by whether a device was in range."""
    frame = incidents.copy()
    frame["fill"] = [
        [*INCIDENT, 200] if covered else [*UNCOVERED, 230] for covered in frame["covered"]
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        frame[["lon", "lat", "fill", "distance_m"]],
        get_position="[lon, lat]",
        get_fill_color="fill",
        get_radius=30,
        radius_min_pixels=2,
        radius_max_pixels=6,
        pickable=True,
    )


def _fit(frame: gpd.GeoDataFrame) -> tuple[float, float, float]:
    """Centre and zoom that make a frame fill the viewport.

    Uses ``total_bounds`` rather than ``geometry.y``: the latter exists only for
    Point geometries, so centring on polygons — Voronoi cells, coverage areas —
    raised ``ValueError: y attribute access only provided for Point geometries``.

    Zoom is derived from the bounding-box span so each town frames itself; Bedok
    is 21.7 km² and Toa Payoh 8.2 km², and one hardcoded zoom cannot serve both.
    """
    if frame is None or len(frame) == 0:
        return 1.3521, 103.8198, 11.0  # Singapore, wide
    minx, miny, maxx, maxy = frame.to_crs(CRS_WGS84).total_bounds
    span = max(maxx - minx, maxy - miny, 1e-4)
    # deck.gl zoom is log2: each level halves the visible span. 360 deg at zoom 0.
    zoom = float(np.clip(np.log2(360.0 / span) - 0.45, 9.0, 15.0))
    return float((miny + maxy) / 2), float((minx + maxx) / 2), zoom


def legend(items: list[tuple[str, list[int]]]) -> str:
    """Inline HTML colour key.

    None of the maps explained their colours, which made an otherwise readable
    picture unreadable: grey, red and amber all carried meaning and none of it
    was stated.
    """
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:1.1rem;'
        f'white-space:nowrap;font-size:0.82rem;">'
        f'<span style="width:0.85rem;height:0.85rem;border-radius:3px;margin-right:0.35rem;'
        f"background:rgba({c[0]},{c[1]},{c[2]},{(c[3] if len(c) > 3 else 235) / 255:.2f});"
        f'border:1px solid rgba(255,255,255,0.28);"></span>{text}</span>'
        for text, c in items
    )
    return f'<div style="margin:0.2rem 0 0.6rem 0;line-height:1.9;">{chips}</div>'


PLAN_LEGEND = [
    ("Covered today", [*EXISTING, 120]),
    ("Ground the plan adds", [*PROPOSED, 190]),
    ("Uncovered demand a device could reach (darker = more people)", [*UNCOVERED, 210]),
    ("Uncovered — no eligible building within reach", [*UNREACHABLE, 170]),
    ("Existing device", [*EXISTING, 255]),
    ("Proposed device", [*PROPOSED, 255]),
]

SERVICE_LEGEND = [
    ("Service area, darker = more demand carried", [196, 74, 62, 200]),
    ("Largest gap containing no device", [*UNCOVERED, 245]),
]

INCIDENT_LEGEND = [
    ("Device in reach", [*INCIDENT, 235]),
    ("Out of reach", [*UNCOVERED, 235]),
]


def deck(layers: list, centre: gpd.GeoDataFrame, zoom: float | None = None) -> pdk.Deck:
    """Assemble a map framed on the study area, whatever its geometry type."""
    latitude, longitude, fitted = _fit(centre)
    return pdk.Deck(
        layers=[layer for layer in layers if layer is not None],
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=fitted if zoom is None else zoom,
        ),
        tooltip={"text": "{label}"},
        map_style="road",
    )


# --- charts ----------------------------------------------------------------


def knee_of(cumulative: np.ndarray) -> int:
    """Budget at which returns start flattening — the elbow of the coverage curve.

    Maximum perpendicular distance from the chord joining the first and last
    points, the standard geometric elbow. Answers "where does money stop being
    worth it" with a number rather than an eyeball.
    """
    values = np.asarray(cumulative, dtype=float)
    if values.size < 3:
        return int(values.size)
    x = np.arange(1, values.size + 1, dtype=float)
    x0, y0, x1, y1 = x[0], values[0], x[-1], values[-1]
    span = float(np.hypot(y1 - y0, x1 - x0)) or 1.0
    offsets = np.abs((y1 - y0) * x - (x1 - x0) * values + x1 * y0 - y1 * x0) / span
    return int(x[int(offsets.argmax())])


def coverage_curve(
    queue: pd.DataFrame, baseline: float, selected: int, knee: int | None = None
) -> alt.Chart:
    """Cumulative coverage against devices funded, with the budget and knee marked."""
    frame = pd.concat(
        [
            pd.DataFrame({"devices": [0], "coverage": [baseline]}),
            queue[["rank", "cumulative_coverage"]].rename(
                columns={"rank": "devices", "cumulative_coverage": "coverage"}
            ),
        ]
    )
    line = (
        alt.Chart(frame)
        .mark_line(point=True, color="#d64541")
        .encode(
            x=alt.X("devices:Q", title="Devices funded"),
            y=alt.Y(
                "coverage:Q",
                title="Modelled coverage",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("devices:Q", title="Devices"),
                alt.Tooltip("coverage:Q", format=".1%", title="Coverage"),
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"devices": [selected]}))
        .mark_rule(color="#999", strokeDash=[4, 4])
        .encode(x="devices:Q")
    )
    layers = [line, marker]
    if knee:
        layers.append(
            alt.Chart(pd.DataFrame({"devices": [knee]}))
            .mark_rule(color="#e0a33c", size=2)
            .encode(x="devices:Q")
        )
    return alt.layer(*layers).properties(height=260)


def marginal_gain_chart(queue: pd.DataFrame) -> alt.Chart:
    """Coverage added by each successive device — the diminishing-returns story."""
    return (
        alt.Chart(queue)
        .mark_bar(color="#d64541")
        .encode(
            x=alt.X("rank:O", title="Install order"),
            y=alt.Y("coverage_added_pp:Q", title="Adds (pp)"),
            tooltip=[
                alt.Tooltip("label:N", title="Site"),
                alt.Tooltip("coverage_added_pp:Q", format=".2f", title="Adds (pp)"),
            ],
        )
        .properties(height=190)
    )


def retrieval_histogram(before: pd.DataFrame, after: pd.DataFrame) -> alt.Chart:
    """Simulated retrieval distances, before against after. The tail is the point."""
    frame = pd.concat(
        [before.assign(scenario="before"), after.assign(scenario="after")], ignore_index=True
    ).dropna(subset=["distance_m"])
    return (
        alt.Chart(frame)
        .mark_bar(opacity=0.65)
        .encode(
            x=alt.X(
                "distance_m:Q", bin=alt.Bin(maxbins=40), title="Retrieval distance (m, one way)"
            ),
            y=alt.Y("count():Q", title="Simulated incidents", stack=None),
            color=alt.Color(
                "scenario:N",
                title="",
                scale=alt.Scale(domain=["before", "after"], range=["#8b8f99", "#d64541"]),
            ),
        )
        .properties(height=230)
    )


def priority_comparison(frame: pd.DataFrame) -> alt.Chart:
    """Orderings compared on the axes that actually separate them.

    Comparing coverage at saturation is meaningless — every ordering converges on
    the same 80.2%, which is why the previous chart showed five identical bars.
    At a fixed budget they differ, and they differ most on worst-case distance and
    the worst-served subzone rather than on headline coverage.
    """
    melted = frame.melt(
        id_vars="priority", value_vars=[c for c in frame.columns if c != "priority"]
    )
    return (
        alt.Chart(melted)
        .mark_bar(color="#d64541")
        .encode(
            y=alt.Y("priority:N", title="", sort="-x"),
            x=alt.X("value:Q", title=""),
            column=alt.Column("variable:N", title="", header=alt.Header(labelFontSize=12)),
            tooltip=[alt.Tooltip("priority:N"), alt.Tooltip("value:Q", format=".3g")],
        )
        .resolve_scale(x="independent")
        .properties(height=_height(len(frame), floor=120), width=170)
    )


def method_bar(comparison: pd.DataFrame) -> alt.Chart:
    """Coverage by method — the baseline comparison, as a picture."""
    frame = comparison.copy()
    frame["short"] = (
        frame["method"]
        .str.replace(r"\s*\(.*\)", "", regex=True)
        .str.replace("existing network only", "existing only")
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("short:N", title="", sort="-x"),
            x=alt.X("coverage:Q", title="Modelled coverage", axis=alt.Axis(format="%")),
            color=alt.Color(
                "short:N",
                legend=None,
                scale=alt.Scale(range=["#8b8f99", "#b0b4bd", "#d64541"]),
            ),
            tooltip=[alt.Tooltip("coverage:Q", format=".2%", title="Coverage")],
        )
        .properties(height=_height(len(frame)))
    )


def objective_scatter(objectives: pd.DataFrame) -> alt.Chart:
    """Coverage against worst-case distance — the equity trade-off, visible."""
    frame = objectives[objectives["objective"] != "do nothing"].copy()
    return (
        alt.Chart(frame)
        .mark_circle(size=190)
        .encode(
            x=alt.X(
                "max_m:Q",
                title="Worst-case distance (m) — lower is better",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("coverage:Q", title="Coverage — higher is better", axis=alt.Axis(format="%")),
            color=alt.Color("objective:N", title="Objective"),
            tooltip=[
                alt.Tooltip("objective:N"),
                alt.Tooltip("coverage:Q", format=".1%"),
                alt.Tooltip("max_m:Q", format=".0f", title="Max (m)"),
                alt.Tooltip("mean_m:Q", format=".0f", title="Mean (m)"),
            ],
        )
        .properties(height=250)
    )


def equity_slope(before: pd.DataFrame, after: pd.DataFrame) -> alt.Chart:
    """Subzone coverage before against after — where the gain actually lands."""
    frame = pd.concat(
        [before.assign(stage="before"), after.assign(stage="after")], ignore_index=True
    )
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("stage:N", title="", sort=["before", "after"]),
            y=alt.Y("coverage:Q", title="Coverage", axis=alt.Axis(format="%")),
            color=alt.Color("subzone:N", title="Subzone"),
            tooltip=[alt.Tooltip("subzone:N"), alt.Tooltip("coverage:Q", format=".1%")],
        )
        .properties(height=280)
    )


def simple_bar(frame: pd.DataFrame, value: str, label: str, title: str) -> alt.Chart:
    """Generic horizontal bar for calibration and label-provenance blocks."""
    return (
        alt.Chart(frame)
        .mark_bar(color="#d64541")
        .encode(
            y=alt.Y(f"{label}:N", title="", sort="-x"),
            x=alt.X(f"{value}:Q", title=title),
            tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q")],
        )
        .properties(height=_height(len(frame)))
    )


def outage_chart(stress: pd.DataFrame, baseline: float) -> alt.Chart:
    """Coverage *lost* as devices go out of service, with the bad-draw band.

    Plotting absolute coverage on a 0-100% axis rendered a 1.7 pp decline as a
    flat line. Showing the loss, on its own scale, is what makes the shape of the
    degradation visible at all.
    """
    frame = stress.copy()
    frame["lost_mean"] = (baseline - frame["coverage_mean"]) * 100
    frame["lost_p05"] = (baseline - frame["coverage_p05"]) * 100
    frame["devices"] = frame["devices_lost"].astype(int).astype(str)

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.25, color="#d64541")
        .encode(
            x=alt.X("devices:N", title="Devices out of service", sort=None),
            y=alt.Y("lost_mean:Q", title="Coverage lost (pp)"),
            y2="lost_p05:Q",
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(point=True, color="#d64541")
        .encode(
            x=alt.X("devices:N", sort=None),
            y="lost_mean:Q",
            tooltip=[
                alt.Tooltip("devices:N", title="Devices lost"),
                alt.Tooltip("lost_mean:Q", format=".2f", title="Typical loss (pp)"),
                alt.Tooltip("lost_p05:Q", format=".2f", title="Bad draw (pp)"),
            ],
        )
    )
    return (band + line).properties(height=230)
