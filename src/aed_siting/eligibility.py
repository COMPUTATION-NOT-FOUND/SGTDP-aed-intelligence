"""Deciding which OSM sites may host an AED.

This is where the pipeline's central commitment lives: **constraints belong in
the feasible set, not in a corrector applied afterwards.** Because the optimisers
can only choose from what this module admits, no optimiser can ever propose a
site in the middle of a road -- no such site exists to be chosen.

The shipped implementation is a deterministic tag allowlist. A learned classifier
implementing the same protocol is a documented upgrade path; it would change how
``p`` is computed, not what the rest of the pipeline consumes.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import geopandas as gpd
import pandas as pd

Decision = Literal["eligible", "excluded", "abstain"]

#: Tag values that plausibly indicate a publicly enterable, staffed or monitored
#: building. Conservative by design: a false positive puts an infeasible site in
#: a planner's queue, whereas a false negative costs one candidate out of
#: thousands.
ALLOWLIST: dict[str, frozenset[str]] = {
    "amenity": frozenset(
        {
            "community_centre",
            "townhall",
            "library",
            "school",
            "college",
            "university",
            "kindergarten",
            "hospital",
            "clinic",
            "doctors",
            "pharmacy",
            "place_of_worship",
            "marketplace",
            "food_court",
            "police",
            "fire_station",
            "bank",
            "post_office",
            "social_facility",
        }
    ),
    "shop": frozenset({"mall", "department_store", "supermarket"}),
    "leisure": frozenset({"sports_centre", "fitness_centre", "stadium", "swimming_pool"}),
    "public_transport": frozenset({"station"}),
    "railway": frozenset({"station"}),
    "building": frozenset(
        {
            "retail",
            "commercial",
            "public",
            "civic",
            "school",
            "hospital",
            "train_station",
            "sports_hall",
            "government",
            "college",
            "university",
            # Admitted on evidence, not intuition. A first version abstained on
            # residential blocks, which would have refused the single most common
            # real AED location in Singapore: 121 of the 284 devices already in
            # Toa Payoh sit in a building tagged `residential`, typically an HDB
            # void deck. The observed network is the calibration set --
            # see validate_against_existing().
            "residential",
            "apartments",
        }
    ),
}

#: Tag values that positively rule a site out regardless of other signals.
DENYLIST: dict[str, frozenset[str]] = {
    "building": frozenset(
        {"house", "detached", "garage", "garages", "shed", "roof", "carport", "hut"}
    ),
    "amenity": frozenset({"parking", "parking_space", "shelter", "bicycle_parking"}),
    "access": frozenset({"private", "no"}),
}

#: ``building=yes`` means "a building, unspecified" -- it carries no information
#: about public access by definition, so it can be neither admitted nor denied on
#: the tag alone. Such sites abstain and are surfaced for field verification.
ABSTAIN_BUILDING = frozenset({"yes", "dormitory", "construction"})


@runtime_checkable
class EligibilityFilter(Protocol):
    """Decide which sites may enter the candidate set."""

    def decide(self, sites: gpd.GeoDataFrame) -> pd.DataFrame:
        """Return columns ``[p, decision]`` aligned to ``sites.index``."""
        ...


class TagAllowlist:
    """Deterministic rules over OSM tags.

    Assumptions:
        A tag that is present is correct. OSM tagging is uneven but rarely
        actively wrong for the categories used here.

    Limits:
        Cannot see whether a door is actually unlocked, whether a lobby is
        staffed, or whether a building still exists. Sites it abstains on are
        excluded from optimisation and surfaced as "needs field verification" --
        never silently admitted or silently dropped.
    """

    def __init__(
        self,
        allowlist: dict[str, frozenset[str]] | None = None,
        denylist: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Store the rule tables, defaulting to the module-level lists."""
        self.allowlist = allowlist or ALLOWLIST
        self.denylist = denylist or DENYLIST

    def decide(self, sites: gpd.GeoDataFrame) -> pd.DataFrame:
        """Classify each site as eligible, excluded, or abstain."""
        denied = pd.Series(False, index=sites.index)
        for tag, values in self.denylist.items():
            if tag in sites.columns:
                denied |= sites[tag].isin(values).fillna(False)

        allowed = pd.Series(False, index=sites.index)
        for tag, values in self.allowlist.items():
            if tag in sites.columns:
                allowed |= sites[tag].isin(values).fillna(False)

        uncertain = pd.Series(False, index=sites.index)
        if "building" in sites.columns:
            uncertain = sites["building"].isin(ABSTAIN_BUILDING).fillna(False)

        decision = pd.Series("excluded", index=sites.index, dtype="object")
        decision[uncertain & ~denied] = "abstain"
        decision[allowed & ~denied] = "eligible"

        probability = pd.Series(0.0, index=sites.index, dtype=float)
        probability[decision == "eligible"] = 1.0
        probability[decision == "abstain"] = 0.5

        return pd.DataFrame({"p": probability, "decision": decision})


def validate_against_existing(
    footprints: gpd.GeoDataFrame,
    existing: gpd.GeoDataFrame,
    eligibility: EligibilityFilter | None = None,
) -> pd.DataFrame:
    """Score the eligibility rules against the network SCDF actually built.

    The 284 devices already sited in Toa Payoh are the only ground truth
    available for "a building that can host a public-access AED". Each one is a
    worked example, so the rules can be calibrated rather than guessed.

    Args:
        footprints: Building polygons, EPSG:3414.
        existing: Existing AED points, EPSG:3414.
        eligibility: Filter under test; defaults to :class:`TagAllowlist`.

    Returns:
        One row per verdict with the count of existing devices whose host
        building received it. A large ``abstain`` or ``excluded`` count is a
        calibration failure: the rules are refusing places that demonstrably work.

    Limits:
        Recall against the existing network, not precision. It cannot show that
        an admitted building would be a *good* site, only that the rules do not
        contradict observed practice. Devices in buildings absent from OSM appear
        as ``no OSM building`` and are a data-coverage gap, not a rule failure.
    """
    verdict = (eligibility or TagAllowlist()).decide(footprints)
    tagged = footprints.assign(decision=verdict["decision"].to_numpy())
    joined = gpd.sjoin(
        existing[["geometry"]], tagged[["decision", "geometry"]], predicate="within", how="left"
    )
    counts = joined["decision"].fillna("no OSM building").value_counts()
    return (
        counts.rename("existing_aeds")
        .to_frame()
        .assign(share=lambda d: (d["existing_aeds"] / len(existing)).round(3))
        .sort_values("existing_aeds", ascending=False)
    )
