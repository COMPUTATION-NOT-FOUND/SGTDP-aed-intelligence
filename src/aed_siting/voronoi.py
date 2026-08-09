"""Voronoi diagnostics: who serves whom, and where the worst gap is.

Voronoi is used here as a **diagnostic, never as a decision procedure**. Cell
centroids are not candidate sites — a Voronoi vertex frequently lands in the
middle of a road, which is exactly the failure this project's candidate-set design
exists to prevent. What the tessellation is good for is measuring the *quality of
a layout* in ways a coverage percentage structurally cannot:

``largest empty circle``
    The vertex farthest from any device is, by construction, the worst-served
    point in the study area. It gives both a distance and a location.
``load Gini``
    Inequality of demand per device. Two networks can report identical coverage
    while one has a device serving 6,000 people next to one serving 400.
``area spread``
    Geographic lumpiness — the ratio of the largest cell to the median.

Together these answer "is this a well-balanced network?", which coverage alone
never asks.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon

from aed_siting.config import CRS_SVY21

#: Padding around the study area used to bound otherwise-infinite Voronoi regions.
_BOUNDS_PAD_M = 20_000.0


@dataclass(frozen=True)
class VoronoiMetrics:
    """Layout-quality measures derived from the tessellation."""

    devices: int
    largest_empty_circle_m: float
    largest_empty_circle_point: tuple[float, float] | None
    load_gini: float
    max_load: float
    median_load: float
    area_spread: float

    def summary(self) -> str:
        """One-line summary for the tracker and the UI."""
        return (
            f"{self.devices} service areas · worst-served point "
            f"{self.largest_empty_circle_m:.0f} m from any device · "
            f"load Gini {self.load_gini:.2f} · area spread {self.area_spread:.1f}x"
        )

    def as_row(self) -> dict:
        """Flat dict for artifact tables."""
        return {
            "devices": self.devices,
            "largest_empty_circle_m": round(self.largest_empty_circle_m, 1),
            "load_gini": round(self.load_gini, 4),
            "max_load": round(self.max_load, 5),
            "median_load": round(self.median_load, 5),
            "area_spread": round(self.area_spread, 2),
        }


def build_cells(devices: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Voronoi service areas for each device, clipped to the study area.

    Args:
        devices: Point geometries in EPSG:3414.
        boundary: Study-area polygon in EPSG:3414.

    Returns:
        One row per device with a ``cell`` polygon and its ``area_m2``, in device
        order. Devices whose cell falls entirely outside the boundary get an empty
        polygon rather than being dropped, so row order stays aligned with the
        distance matrix.

    Assumptions:
        Straight-line Voronoi. The service areas therefore assume travel as the
        crow flies, matching the shipped distance model but not a walking network.

    Limits:
        A cell shows which device is *nearest*, not which is reachable. A cell can
        be entirely outside every device's coverage threshold and still be
        assigned; that is the point of reading it alongside coverage rather than
        instead of it.
    """
    if len(devices) < 4:
        # Voronoi is undefined below 4 points in 2-D; fall back to the whole area.
        polygon = boundary.union_all()
        cells = [polygon if i == 0 else Polygon() for i in range(len(devices))]
        return _frame(devices, cells)

    coords = np.c_[devices.geometry.x.to_numpy(), devices.geometry.y.to_numpy()]
    minx, miny, maxx, maxy = boundary.total_bounds
    # Sentinel points far outside the area force every real region to be finite.
    sentinels = np.array(
        [
            [minx - _BOUNDS_PAD_M, miny - _BOUNDS_PAD_M],
            [minx - _BOUNDS_PAD_M, maxy + _BOUNDS_PAD_M],
            [maxx + _BOUNDS_PAD_M, miny - _BOUNDS_PAD_M],
            [maxx + _BOUNDS_PAD_M, maxy + _BOUNDS_PAD_M],
        ]
    )
    diagram = Voronoi(np.vstack([coords, sentinels]))
    area = boundary.union_all()

    cells = []
    for index in range(len(coords)):
        region = diagram.regions[diagram.point_region[index]]
        if not region or -1 in region:
            cells.append(Polygon())
            continue
        polygon = Polygon(diagram.vertices[region])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        cells.append(polygon.intersection(area))
    return _frame(devices, cells)


def _frame(devices: gpd.GeoDataFrame, cells: list) -> gpd.GeoDataFrame:
    """Assemble the cell frame, preserving device order and labels."""
    frame = gpd.GeoDataFrame(
        {
            "device_index": np.arange(len(devices)),
            "label": devices.get("label", pd.Series(range(len(devices)))).to_numpy(),
        },
        geometry=gpd.GeoSeries(cells, crs=CRS_SVY21),
        crs=CRS_SVY21,
    )
    frame["area_m2"] = frame.geometry.area.round(1)
    return frame


def largest_empty_circle(
    devices: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> tuple[float, Point | None]:
    """Radius and centre of the largest circle containing no device.

    The centre is always a Voronoi vertex (or a boundary point), which is what
    makes this cheap: the worst-served location cannot be anywhere else.

    Returns:
        ``(radius_m, point)``; ``(inf, None)`` when there are no devices.

    Limits:
        Straight-line, and constrained to the study area. It answers "how far is
        the worst spot from the nearest device", not "how long would it take to
        walk there", and it ignores whether anyone is actually at that spot --
        read it next to the demand surface, not alone.
    """
    if len(devices) == 0:
        return float("inf"), None
    area = boundary.union_all()
    coords = np.c_[devices.geometry.x.to_numpy(), devices.geometry.y.to_numpy()]

    candidates: list[Point] = []
    if len(devices) >= 4:
        diagram = Voronoi(coords)
        for vertex in diagram.vertices:
            point = Point(vertex)
            if area.contains(point):
                candidates.append(point)
    # Boundary-constrained optima are not Voronoi vertices, so sample the edge too.
    perimeter = area.exterior if hasattr(area, "exterior") else area.boundary
    if perimeter is not None and perimeter.length > 0:
        step = max(perimeter.length / 400.0, 1.0)
        candidates += [perimeter.interpolate(d) for d in np.arange(0, perimeter.length, step)]
    if not candidates:
        return 0.0, None

    device_points = MultiPoint([Point(c) for c in coords])
    best_point, best_radius = None, -1.0
    for point in candidates:
        radius = point.distance(device_points)
        if radius > best_radius:
            best_radius, best_point = radius, point
    return float(best_radius), best_point


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution.

    0.0 is perfectly even load across devices; values approaching 1.0 mean a few
    devices carry almost all the demand.
    """
    array = np.sort(np.asarray(values, dtype=float))
    total = array.sum()
    if array.size == 0 or total <= 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    return float((2 * (index * array).sum()) / (array.size * total) - (array.size + 1) / array.size)


def measure(
    devices: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    loads: np.ndarray,
) -> tuple[gpd.GeoDataFrame, VoronoiMetrics]:
    """Build cells and score the layout.

    Args:
        devices: Device points, EPSG:3414.
        boundary: Study-area polygon, EPSG:3414.
        loads: Demand assigned to each device, aligned to ``devices`` -- reuse
            :func:`aed_siting.metrics.catchment_load` rather than recomputing.

    Returns:
        ``(cells, metrics)``. ``cells`` carries a ``load`` column for shading.
    """
    cells = build_cells(devices, boundary)
    load = np.asarray(loads, dtype=float)
    if load.size != len(cells):
        load = np.zeros(len(cells), dtype=float)
    cells["load"] = load.round(6)

    radius, point = largest_empty_circle(devices, boundary)
    areas = cells["area_m2"].to_numpy()
    populated = areas[areas > 0]
    metrics = VoronoiMetrics(
        devices=len(devices),
        largest_empty_circle_m=radius,
        largest_empty_circle_point=(point.x, point.y) if point is not None else None,
        load_gini=gini(load),
        max_load=float(load.max()) if load.size else 0.0,
        median_load=float(np.median(load)) if load.size else 0.0,
        area_spread=float(populated.max() / np.median(populated)) if populated.size else 0.0,
    )
    return cells, metrics
