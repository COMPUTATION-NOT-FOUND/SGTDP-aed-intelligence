"""H3 demand grid over the study area.

Demand and coverage are both evaluated at hex centroids. Hexagons are used rather
than a square lattice because every neighbour is equidistant, so a coverage disc
is approximated without the diagonal bias a square grid introduces.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon

from aed_siting.config import CRS_SVY21, CRS_WGS84, H3_RESOLUTION


def build_grid(
    boundary: gpd.GeoDataFrame,
    resolution: int = H3_RESOLUTION,
) -> gpd.GeoDataFrame:
    """Tile the study area with H3 cells, returned as centroids in SVY21.

    Args:
        boundary: Single-row study-area frame, any CRS.
        resolution: H3 resolution. 10 gives ~0.0150 km² per cell, so roughly 545
            cells over Toa Payoh's 8.18 km².

    Returns:
        A frame indexed 0..n-1 with columns ``h3`` (cell id), ``geometry`` (the
        centroid, EPSG:3414) and ``hex_geometry`` (the cell polygon, EPSG:3414),
        sorted by cell id so the row order is deterministic.

    Assumptions:
        H3 works in WGS84, so the boundary is reprojected to 4326 to tile and the
        results are projected back to 3414. Every distance downstream is metres in
        3414; nothing is measured in degrees.

    Limits:
        A cell is included when H3 judges it to intersect the polygon, so cells on
        the boundary are partly outside the town. At resolution 10 the resulting
        area error is small relative to the modelling assumptions it feeds, and it
        is symmetric — it does not bias any site over another.
    """
    shape = h3.geo_to_h3shape(boundary.to_crs(CRS_WGS84).union_all().__geo_interface__)
    cells = sorted(h3.h3shape_to_cells(shape, resolution))
    if not cells:
        raise RuntimeError("H3 returned no cells for the study-area polygon")

    centroids = [h3.cell_to_latlng(cell) for cell in cells]
    polygons = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]) for cell in cells]

    frame = gpd.GeoDataFrame(
        {"h3": cells},
        geometry=gpd.points_from_xy([lng for _, lng in centroids], [lat for lat, _ in centroids]),
        crs=CRS_WGS84,
    ).to_crs(CRS_SVY21)

    frame["hex_geometry"] = gpd.GeoSeries(polygons, crs=CRS_WGS84).to_crs(CRS_SVY21).to_numpy()
    return frame.reset_index(drop=True)


def grid_summary(grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> pd.Series:
    """Descriptive stats for the tracker and the method card."""
    hexes = gpd.GeoSeries(grid["hex_geometry"], crs=CRS_SVY21)
    town_km2 = boundary.to_crs(CRS_SVY21).geometry.area.sum() / 1e6
    return pd.Series(
        {
            "cells": len(grid),
            "resolution": H3_RESOLUTION,
            "mean_cell_km2": round(float(hexes.area.mean() / 1e6), 5),
            "tiled_km2": round(float(hexes.area.sum() / 1e6), 3),
            "town_km2": round(float(town_km2), 3),
            "overshoot_pct": round(float(hexes.area.sum() / 1e6 / town_km2 - 1) * 100, 1),
        }
    )
