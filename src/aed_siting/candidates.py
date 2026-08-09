"""Building the candidate-site set.

A candidate is a real OSM building footprint that passes the eligibility filter.
Its representative point -- not its centroid -- is used as the site location,
because a centroid can fall outside a concave or L-shaped footprint, and a site
outside its own building is exactly the failure mode this design exists to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from aed_siting.eligibility import EligibilityFilter, TagAllowlist
from aed_siting.io.osm import buildings


@dataclass(frozen=True)
class CandidateReport:
    """What the filter admitted, excluded and abstained on."""

    buildings_considered: int
    eligible: int
    abstained: int
    excluded: int

    @property
    def abstention_rate(self) -> float:
        """Share of buildings the filter refused to judge."""
        if not self.buildings_considered:
            return 0.0
        return self.abstained / self.buildings_considered

    def summary(self) -> str:
        """One-line summary for the tracker and the method card."""
        return (
            f"{self.buildings_considered} building footprints → {self.eligible} eligible, "
            f"{self.abstained} abstained ({self.abstention_rate:.1%}, flagged for field "
            f"verification), {self.excluded} excluded."
        )


def build_candidates(
    osm: gpd.GeoDataFrame,
    eligibility: EligibilityFilter | None = None,
) -> tuple[gpd.GeoDataFrame, CandidateReport]:
    """Select candidate AED sites from OSM building footprints.

    Args:
        osm: Frozen OSM extract, EPSG:3414, already clipped to the study area.
        eligibility: Filter to apply; defaults to :class:`TagAllowlist`.

    Returns:
        ``(candidates, report)``. ``candidates`` carries ``site_id``, ``name``,
        ``footprint_m2``, ``p``, ``decision``, ``geometry`` (representative point
        inside the footprint) and ``footprint`` (the polygon itself).

    Assumptions:
        A new device is installed at ground level, so candidates carry no floor
        penalty. This is the asymmetry with existing devices disclosed in the
        method card: it flatters new installs.

    Limits:
        Only buildings OSM knows about can be candidates. A real, suitable site
        absent from OSM is invisible here, and no amount of optimisation
        recovers it.
    """
    footprints = buildings(osm)
    if len(footprints) == 0:
        empty = footprints.assign(site_id=pd.Series(dtype="int64"))
        return empty, CandidateReport(0, 0, 0, 0)

    verdict = (eligibility or TagAllowlist()).decide(footprints)
    report = CandidateReport(
        buildings_considered=len(footprints),
        eligible=int((verdict["decision"] == "eligible").sum()),
        abstained=int((verdict["decision"] == "abstain").sum()),
        excluded=int((verdict["decision"] == "excluded").sum()),
    )

    keep = footprints[verdict["decision"] == "eligible"].copy()
    keep["p"] = verdict.loc[keep.index, "p"]
    keep["decision"] = verdict.loc[keep.index, "decision"]
    keep["footprint_m2"] = keep.geometry.area.round(1)
    keep["name"] = keep["name"] if "name" in keep.columns else pd.NA
    keep["footprint"] = keep.geometry

    # Representative point is guaranteed to lie inside its own polygon; centroid
    # is not. The candidate-inside-footprint gate depends on this choice.
    candidates = keep.set_geometry(keep.geometry.representative_point())

    # Deterministic ordering, then a stable id used by every downstream artifact.
    candidates = candidates.sort_values(
        ["footprint_m2", "name"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    candidates["site_id"] = candidates.index

    columns = ["site_id", "name", "footprint_m2", "p", "decision", "geometry", "footprint"]
    # Address tags must survive: they are how an unnamed HDB block gets a label a
    # planner can act on ("Blk 265 Kim Keat Avenue"). Dropping them here silently
    # collapses the whole naming ladder onto its weakest rung.
    tags = [
        c
        for c in (
            "amenity",
            "shop",
            "leisure",
            "public_transport",
            "building",
            "addr:housename",
            "addr:housenumber",
            "addr:street",
        )
        if c in candidates.columns
    ]
    return candidates[columns + tags], report


def abstained_sites(
    osm: gpd.GeoDataFrame, eligibility: EligibilityFilter | None = None
) -> gpd.GeoDataFrame:
    """Buildings the filter refused to judge, for the field-verification queue.

    Surfacing these is the abstention requirement in practice: an uncertain site
    is neither used nor discarded silently, it is handed to a human.
    """
    footprints = buildings(osm)
    if len(footprints) == 0:
        return footprints
    verdict = (eligibility or TagAllowlist()).decide(footprints)
    out = footprints[verdict["decision"] == "abstain"].copy()
    out["reason"] = "tags do not establish public access; needs field verification"
    return out.reset_index(drop=True)
