"""Reading the frozen OpenStreetMap extract.

The extract is a snapshot taken once by ``scripts/fetch_sources.py`` and
checksummed into the manifest. OSM is unversioned, so this frozen file -- never
the live API -- is the source of truth for every judged result.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from aed_siting.config import CRS_SVY21, OSM_PATH

_NULLISH = {"", "nan", "none", "null", "<na>"}


def _clean(series: pd.Series) -> pd.Series:
    """Normalise OSM tag values, mapping placeholder strings to NA."""
    text = series.astype("string").str.strip()
    return text.mask(text.str.lower().isin(_NULLISH))


def load_osm(path=None, boundary: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Load the frozen OSM extract, projected to SVY21.

    Args:
        path: Frozen Overpass GeoJSON.
        boundary: Optional study-area frame; features whose representative point
            falls outside are dropped.

    Returns:
        Features in EPSG:3414 with tag columns normalised to nullable strings.

    Limits:
        OSM completeness is uneven and unverifiable from within this pipeline. Tag
        coverage is measured and reported (see :func:`tag_coverage`) rather than
        assumed, because a missing tag is indistinguishable from an absent
        feature.
    """
    source = path if path is not None else OSM_PATH
    reader = gpd.read_parquet if str(source).endswith(".parquet") else gpd.read_file
    frame = reader(source).to_crs(CRS_SVY21)
    for column in frame.columns:
        if column != "geometry":
            frame[column] = _clean(frame[column])

    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    if boundary is not None:
        polygon = boundary.to_crs(CRS_SVY21).union_all()
        inside = frame.geometry.representative_point().within(polygon)
        frame = frame[inside.to_numpy()]
    return frame.reset_index(drop=True)


def buildings(osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Polygonal features carrying a ``building`` tag."""
    if "building" not in osm.columns:
        return osm.iloc[:0]
    has_tag = osm["building"].notna()
    is_polygon = osm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    return osm[has_tag & is_polygon].reset_index(drop=True)


def points_of_interest(osm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Features carrying any activity tag, used for the POI-density proxy."""
    columns = [c for c in ("amenity", "shop", "leisure", "public_transport") if c in osm.columns]
    if not columns:
        return osm.iloc[:0]
    mask = osm[columns].notna().any(axis=1)
    return osm[mask].reset_index(drop=True)


def tag_coverage(frame: gpd.GeoDataFrame, tag: str) -> float:
    """Share of rows carrying a non-null value for ``tag``.

    Reported rather than assumed: ``building:levels`` coverage in particular
    decides whether a levels-based demand proxy can be trusted.
    """
    if tag not in frame.columns or len(frame) == 0:
        return 0.0
    return float(frame[tag].notna().mean())
