"""Demand proxies over the H3 grid.

No observed demand exists: the brief forbids using incident data, and none is
supplied. Everything here is a **proxy**, and the honest response to that is not
to pick the best-sounding one but to run several and report where they disagree.

Three proxies plus a uniform reference:

``area``
    Building footprint area per cell. Fully observed -- no imputation anywhere.
``area_levels``
    Footprint area x storeys, capturing vertical population density. Requires
    ``building:levels``, which is present on only about a third of footprints, so
    the rest is imputed from the median for that building type.
``poi``
    Count of activity tags per cell, standing in for daytime footfall rather
    than residence.
``uniform``
    Every cell equal. Not a belief about the world -- a reference case that shows
    what the other proxies are actually contributing.

The disagreement between ``area`` and ``area_levels`` is the *measurement* of the
imputation's influence. That is why both exist: burying the imputation inside a
single proxy would hide exactly the uncertainty the brief asks to be exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from aed_siting.config import CRS_SVY21
from aed_siting.io.osm import buildings, points_of_interest, tag_coverage

PROXIES = ("area", "area_levels", "poi", "uniform")

#: Below this, a levels-based proxy is imputation-dominated and must be reported
#: as such rather than presented as observation.
LEVELS_COVERAGE_FLOOR = 0.60

_DEFAULT_LEVELS = 3.0
_MAX_PLAUSIBLE_LEVELS = 60.0


@dataclass
class DemandReport:
    """Provenance for the demand surfaces."""

    cells: int
    buildings: int
    levels_coverage: float = 0.0
    levels_imputed: int = 0
    poi_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def levels_proxy_is_imputation_dominated(self) -> bool:
        """True when most storey counts had to be invented."""
        return self.levels_coverage < LEVELS_COVERAGE_FLOOR

    def summary(self) -> str:
        """One-line summary for the tracker and the method card."""
        flag = " (IMPUTATION-DOMINATED)" if self.levels_proxy_is_imputation_dominated else ""
        return (
            f"{self.cells} cells, {self.buildings} buildings, {self.poi_count} POIs; "
            f"building:levels present on {self.levels_coverage:.1%}{flag}, "
            f"{self.levels_imputed} imputed."
        )


def _assign_to_cells(
    features: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, weights: np.ndarray
) -> np.ndarray:
    """Sum feature weights into the grid cell whose hexagon contains each feature."""
    totals = np.zeros(len(grid), dtype=float)
    if len(features) == 0:
        return totals

    hexes = gpd.GeoDataFrame(
        {"cell": np.arange(len(grid))},
        geometry=gpd.GeoSeries(grid["hex_geometry"], crs=CRS_SVY21),
        crs=CRS_SVY21,
    )
    points = features.set_geometry(features.geometry.representative_point())
    joined = gpd.sjoin(points, hexes, predicate="within", how="inner")
    if joined.empty:
        return totals

    weight_series = pd.Series(weights, index=features.index)
    grouped = weight_series.loc[joined.index].groupby(joined["cell"].to_numpy()).sum()
    totals[grouped.index.to_numpy()] = grouped.to_numpy()
    return totals


def _storeys(footprints: gpd.GeoDataFrame, report: DemandReport) -> np.ndarray:
    """Storey counts, imputed by building-type median where absent."""
    # Cast to float explicitly: pd.to_numeric infers nullable Int64 when every
    # storey count is a clean integer, and filling that with a fractional median
    # raises. Toa Payoh, Ang Mo Kio and Bedok have whole-number medians and hid
    # this; Jurong East does not.
    raw = pd.to_numeric(footprints.get("building:levels"), errors="coerce").astype("Float64")
    raw = raw.where((raw > 0) & (raw <= _MAX_PLAUSIBLE_LEVELS))
    report.levels_coverage = float(raw.notna().mean()) if len(raw) else 0.0

    kind = (
        footprints["building"]
        if "building" in footprints.columns
        else pd.Series("yes", index=footprints.index)
    )
    by_type = raw.groupby(kind).transform("median")
    filled = raw.fillna(by_type).fillna(raw.median()).fillna(_DEFAULT_LEVELS)
    report.levels_imputed = int(raw.isna().sum())
    return filled.to_numpy(dtype=float)


def build_demand(
    grid: gpd.GeoDataFrame, osm: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, DemandReport]:
    """Build every demand proxy over the grid.

    Args:
        grid: H3 grid with ``hex_geometry``, EPSG:3414.
        osm: Frozen OSM extract clipped to the study area, EPSG:3414.

    Returns:
        ``(demand, report)``. ``demand`` has one column per proxy, each summing
        to 1.0 so proxies are directly comparable.

    Assumptions:
        Presence of built floor area stands in for presence of people. This is a
        proxy, not a measurement, and its weaknesses are the reason more than one
        is computed.

    Limits:
        None of these is population. They cannot distinguish a busy mall from an
        empty warehouse of equal footprint, and they carry no time-of-day
        variation. Conclusions are only trustworthy where the proxies agree --
        which the cross-proxy sensitivity analysis reports explicitly.
    """
    footprints = buildings(osm)
    pois = points_of_interest(osm)
    report = DemandReport(
        cells=len(grid),
        buildings=len(footprints),
        poi_count=len(pois),
    )
    report.notes.append(
        f"building:levels tag coverage {tag_coverage(footprints, 'building:levels'):.1%}"
    )

    area = footprints.geometry.area.to_numpy() if len(footprints) else np.zeros(0)
    storeys = _storeys(footprints, report) if len(footprints) else np.zeros(0)

    surfaces = {
        "area": _assign_to_cells(footprints, grid, area),
        "area_levels": _assign_to_cells(footprints, grid, area * storeys),
        "poi": _assign_to_cells(pois, grid, np.ones(len(pois))),
        "uniform": np.ones(len(grid), dtype=float),
    }

    demand = pd.DataFrame(
        {name: _normalise(values) for name, values in surfaces.items()},
        index=pd.Index(grid["h3"], name="h3"),
    )
    return demand, report


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale a surface to sum to 1.0, falling back to uniform if it is empty."""
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / len(values)) if len(values) else values
    return values / total


def proxy_agreement(demand: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between every pair of proxies.

    High agreement means the choice of proxy is not driving the answer; low
    agreement is a warning that it is, and belongs in the evaluation report
    rather than in a footnote.

    Constant surfaces (``uniform``) are dropped: rank correlation is undefined
    against a column with no variation, and reporting NaN would read as a bug
    rather than as the tautology it is.
    """
    varying = demand.loc[:, demand.nunique() > 1]
    return varying.corr(method="spearman").round(3)
