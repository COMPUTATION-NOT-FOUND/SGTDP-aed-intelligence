"""Stable site identifiers and human-readable labels.

A planner cannot act on "(unnamed building)". Only 50 of 791 candidate buildings
in Toa Payoh carry an OSM ``name``, and 544 of the unnamed ones are HDB
residential blocks -- but ``addr:housenumber`` and ``addr:street`` are present on
two thirds of footprints, which is exactly how Singaporeans refer to a block:
*Blk 265 Kim Keat Avenue*.

Two separate things are produced, and the distinction matters:

**The id** (``TP-041``) is stable, opaque, and is what every artifact keys on. It
survives a relabelling.

**The label** is for humans and is resolved by a ladder of decreasing confidence.
``label_source`` records which rung was used, so a reader can tell a real name
from a constructed one.
"""

from __future__ import annotations

from typing import Final

import geopandas as gpd
import pandas as pd

from aed_siting.config import CRS_WGS84, town_prefix

_NULLISH: Final[frozenset[str]] = frozenset({"", "nan", "none", "null", "<na>", "n/a"})

#: Ladder rungs, most to least trustworthy.
LABEL_SOURCES: Final[tuple[str, ...]] = (
    "osm_name",
    "osm_housename",
    "address_block",
    "address_street",
    "building_type",
    "identifier",
)


def _clean(value: object) -> str:
    """Normalise a tag value; return '' for anything null-ish."""
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in _NULLISH else text


def _titlecase_type(value: object) -> str:
    """Render an OSM building/amenity tag as prose: ``sports_centre`` -> ``Sports centre``."""
    text = _clean(value).replace("_", " ")
    return text.capitalize() if text else "Building"


def site_ids(count: int, slug: str) -> list[str]:
    """Stable, zero-padded identifiers for one town's candidate sites.

    Args:
        count: Number of sites, already in deterministic order.
        slug: Town slug; supplies the prefix.

    Returns:
        ``["TP-001", "TP-002", ...]``. Width follows the town's size so ids sort
        lexicographically in the same order as numerically.

    Limits:
        Identity is positional, so it is stable only while the upstream sort is.
        ``candidates.build_candidates`` sorts by footprint area then name, and
        ``test_site_ids_are_stable_across_runs`` locks that.
    """
    width = max(3, len(str(count)))
    prefix = town_prefix(slug)
    return [f"{prefix}-{i + 1:0{width}d}" for i in range(count)]


def resolve_label(row: pd.Series, fallback_id: str) -> tuple[str, str]:
    """Resolve one site's display label.

    Args:
        row: A candidate row carrying OSM tag columns.
        fallback_id: The site's stable id, used when nothing else resolves.

    Returns:
        ``(label, label_source)`` where ``label_source`` is one of
        :data:`LABEL_SOURCES`.

    Limits:
        Rungs 3-5 *construct* a label rather than reporting one. A constructed
        label describes where a building is, not what it is called, and
        ``label_source`` is carried through to the UI so the difference stays
        visible.
    """
    if name := _clean(row.get("name")):
        return name, "osm_name"
    if housename := _clean(row.get("addr:housename")):
        return housename, "osm_housename"

    number = _clean(row.get("addr:housenumber"))
    street = _clean(row.get("addr:street"))
    if number and street:
        return f"Blk {number} {street}", "address_block"
    if street:
        return f"{_titlecase_type(row.get('building'))} on {street}", "address_street"

    kind = _clean(row.get("amenity")) or _clean(row.get("shop")) or _clean(row.get("building"))
    if kind and kind.lower() != "yes":
        return f"{_titlecase_type(kind)} {fallback_id}", "building_type"
    return fallback_id, "identifier"


def label_candidates(candidates: gpd.GeoDataFrame, slug: str) -> gpd.GeoDataFrame:
    """Attach ``site_id``, ``label`` and ``label_source`` to a candidate frame.

    Replaces the positional integer ``site_id`` with a stable string id while
    keeping the original integer as ``site_index`` -- the distance matrix is
    column-indexed, so the numeric position is still needed internally.
    """
    frame = candidates.copy().reset_index(drop=True)
    frame["site_index"] = frame.index
    ids = site_ids(len(frame), slug)
    frame["site_id"] = ids
    resolved = [resolve_label(row, ids[i]) for i, row in enumerate(frame.to_dict("records"))]
    frame["label"] = [label for label, _ in resolved]
    frame["label_source"] = [source for _, source in resolved]
    return frame


def label_existing(existing: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach a display label to existing devices.

    The registry supplies ``BUILDING_NAME`` for most records; where it is absent
    the same block-address construction applies, using the registry's own
    ``HOUSE_NUMBER`` and ``ROAD_NAME`` fields.
    """
    frame = existing.copy().reset_index(drop=True)
    labels, sources = [], []
    for row in frame.to_dict("records"):
        if building := _clean(row.get("BUILDING_NAME")):
            labels.append(building)
            sources.append("registry_building_name")
            continue
        number, road = _clean(row.get("HOUSE_NUMBER")), _clean(row.get("ROAD_NAME"))
        if number and road:
            labels.append(f"Blk {number} {road}")
            sources.append("address_block")
        elif road:
            labels.append(road)
            sources.append("address_street")
        else:
            labels.append(_clean(row.get("AED_ID")) or "unknown device")
            sources.append("identifier")
    frame["label"] = labels
    frame["label_source"] = sources
    return frame


def sites_reference(candidates: gpd.GeoDataFrame) -> pd.DataFrame:
    """Lookup table mapping every site id to a label and a coordinate.

    Shipped as ``sites.csv`` and offered for download in the UI so any id in the
    install queue can be resolved to a place on the ground.
    """
    wgs84 = candidates.to_crs(CRS_WGS84)
    columns = {
        "site_id": candidates["site_id"],
        "label": candidates["label"],
        "label_source": candidates["label_source"],
        "lat": wgs84.geometry.y.round(6),
        "lon": wgs84.geometry.x.round(6),
        "footprint_m2": candidates.get("footprint_m2"),
        "building": candidates.get("building"),
    }
    return pd.DataFrame(columns).reset_index(drop=True)


def label_quality(candidates: gpd.GeoDataFrame) -> pd.DataFrame:
    """How many labels came from each rung of the ladder.

    Reported rather than hidden: a submission claiming readable recommendations
    should show what share of them are real names.
    """
    counts = candidates["label_source"].value_counts()
    return (
        counts.rename("sites")
        .to_frame()
        .assign(share=lambda d: (d["sites"] / len(candidates)).round(3))
        .reindex([s for s in LABEL_SOURCES if s in counts.index])
    )
