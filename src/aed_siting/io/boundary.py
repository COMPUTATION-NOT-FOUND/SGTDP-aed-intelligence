"""Planning-area and subzone boundary loading."""

from __future__ import annotations

import geopandas as gpd

from aed_siting.config import (
    CRS_SVY21,
    PLANNING_AREAS_PATH,
    SUBZONES_PATH,
    TOWN,
)

_AREA_NAME_COL = "PLN_AREA_N"
_SUBZONE_NAME_COL = "SUBZONE_N"


def _resolve_name_column(frame: gpd.GeoDataFrame, wanted: str) -> str:
    """Find an attribute column, recovering it from an HTML table if needed.

    Several URA layers on data.gov.sg publish their attributes as an HTML
    ``<table>`` inside a single ``Description`` property rather than as fields.
    The MP2019 boundary layers were verified to carry clean columns, so this is a
    contingency; it exists because the format is not guaranteed across refreshes.

    Raises:
        RuntimeError: If the column cannot be found or recovered.
    """
    for column in frame.columns:
        if column.upper() == wanted:
            return column

    if "Description" in frame.columns:
        import pandas as pd

        parsed = [
            dict(zip(*pd.read_html(html, index_col=0)[0].T.to_numpy(), strict=False))
            for html in frame["Description"]
        ]
        recovered = pd.DataFrame(parsed, index=frame.index)
        for column in recovered.columns:
            if str(column).upper() == wanted:
                frame[wanted] = recovered[column]
                return wanted

    raise RuntimeError(
        f"{wanted} not found and not recoverable from an HTML Description column; "
        f"available columns: {list(frame.columns)}"
    )


def _repair(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Fix invalid rings in place, leaving valid geometry untouched.

    Some URA planning-area polygons carry ring self-intersections — BEDOK does,
    at (103.9321, 1.3055). Left alone they break H3 tiling and make ``within``
    tests unreliable, and osmnx refuses them outright. A zero-width buffer closes
    the ring without moving the boundary.

    Limits:
        Silently changes geometry, so the area change is asserted to be
        negligible rather than assumed; a real distortion raises instead.
    """
    invalid = ~frame.geometry.is_valid
    if not invalid.any():
        return frame
    before = float(frame.geometry.area.sum())
    frame = frame.copy()
    frame.loc[invalid, frame.geometry.name] = frame.loc[invalid, frame.geometry.name].buffer(0)
    after = float(frame.geometry.area.sum())
    if before > 0 and abs(after - before) / before > 1e-6:
        raise RuntimeError(
            f"repairing {label!r} changed its area by "
            f"{abs(after - before) / before:.2%} — refusing to continue silently"
        )
    return frame


def load_town_boundary(path=PLANNING_AREAS_PATH, town: str = TOWN) -> gpd.GeoDataFrame:
    """Load the study-area polygon, projected to SVY21.

    Args:
        path: Frozen MP2019 planning-area GeoJSON ("No Sea" variant).
        town: Planning-area name, matched case- and whitespace-insensitively.

    Returns:
        A single-row frame in EPSG:3414.

    Assumptions:
        The "No Sea" variant is required: sea-inclusive boundaries would inflate
        the study area and place demand cells over water.

    Raises:
        RuntimeError: If the name matches zero or more than one polygon.
    """
    areas = gpd.read_file(path)
    column = _resolve_name_column(areas, _AREA_NAME_COL)
    match = areas[areas[column].astype(str).str.upper().str.strip() == town.upper()]
    if len(match) != 1:
        raise RuntimeError(f"expected exactly 1 polygon for {town!r}, got {len(match)}")
    return _repair(match.to_crs(CRS_SVY21).reset_index(drop=True), town)


def load_subzones(path=SUBZONES_PATH, town: str = TOWN) -> gpd.GeoDataFrame:
    """Load subzone polygons for the study area, projected to SVY21.

    Subzones back the worst-served-subzone equity metric. They are selected by
    the planning-area name they carry, not by spatial join, so a subzone is never
    split between towns.
    """
    subzones = gpd.read_file(path)
    area_column = _resolve_name_column(subzones, _AREA_NAME_COL)
    name_column = _resolve_name_column(subzones, _SUBZONE_NAME_COL)
    match = subzones[subzones[area_column].astype(str).str.upper().str.strip() == town.upper()]
    return _repair(
        match.to_crs(CRS_SVY21).rename(columns={name_column: "subzone"}).reset_index(drop=True),
        f"{town} subzones",
    )
