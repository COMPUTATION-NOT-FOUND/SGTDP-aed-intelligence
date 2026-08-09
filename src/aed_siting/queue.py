"""Turning a site ordering into a planner-readable install queue.

The product of this tool is not a map of pins. It is an ordered list a planner
can act on one budget cycle at a time, where every row states what it buys and
why it was chosen.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from aed_siting.metrics import catchment_load, nearest


def build_queue(
    candidates: gpd.GeoDataFrame,
    order: list[int],
    marginal_gain: list[float],
    cumulative: list[float],
    demand: np.ndarray,
    dist: np.ndarray,
    dist_existing: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Assemble the ranked install queue with per-site justification.

    Args:
        candidates: Candidate sites, aligned to the columns of ``dist``.
        order: Site indices in install order.
        marginal_gain: Demand newly covered by each pick, as a fraction.
        cumulative: Running coverage after each pick.
        demand: Per-cell weight.
        dist: One-way metres to candidates.
        dist_existing: One-way metres to existing devices.
        threshold: One-way metres.

    Returns:
        One row per recommended device, in install order, with the coverage it
        adds, the running total, how far the nearest existing device is, and the
        worst-case distance it closes.

    Limits:
        Every column is a modelled planning estimate. None of it asserts that a
        site is available, permitted, or that a device placed there would be
        maintained.
    """
    if not order:
        return pd.DataFrame(
            columns=[
                "rank",
                "site_id",
                "site_index",
                "label",
                "label_source",
                "coverage_added_pp",
                "cumulative_coverage",
                "demand_served",
                "new_ground_share",
                "nearest_existing_m",
                "worst_case_closed_m",
            ]
        )

    weights = np.asarray(demand, dtype=float)
    # Share of each proposal's reach that the existing network does not already
    # cover. Overlapping translucent circles read as redundancy on a map even
    # when they are not; this is the number that settles it.
    existing_reach = (
        (dist_existing <= threshold).any(axis=1)
        if dist_existing.shape[1]
        else np.zeros(len(weights), dtype=bool)
    )
    existing_nearest = (
        nearest(dist_existing, range(dist_existing.shape[1]))
        if dist_existing.shape[1]
        else np.full(len(weights), np.inf)
    )
    load = catchment_load(weights, dist, order)

    rows = []
    for rank, (site, gain, running) in enumerate(
        zip(order, marginal_gain, cumulative, strict=True), 1
    ):
        newly = (dist[:, site] <= threshold) & (existing_nearest > threshold)
        row = candidates.iloc[site]
        rows.append(
            {
                "rank": rank,
                "site_id": row.get("site_id", site),
                "site_index": int(site),
                "label": row.get("label", str(site)),
                "label_source": row.get("label_source", "identifier"),
                "coverage_added_pp": round(gain * 100, 3),
                "cumulative_coverage": round(running, 4),
                "demand_served": round(float(load.get(site, 0.0)), 4),
                "new_ground_share": round(
                    float(
                        (dist[:, site] <= threshold).sum()
                        and ((dist[:, site] <= threshold) & ~existing_reach).sum()
                        / max((dist[:, site] <= threshold).sum(), 1)
                    ),
                    3,
                ),
                "nearest_existing_m": round(float(existing_nearest[dist[:, site].argmin()]), 1),
                "worst_case_closed_m": round(
                    float(existing_nearest[newly].max()) if newly.any() else 0.0, 1
                ),
            }
        )
    return pd.DataFrame(rows)


def format_justification(row: pd.Series) -> str:
    """One sentence explaining why a site earned its rank.

    Feeds the UI's traceable source-to-output reasoning. Deliberately hedged:
    every verb is modelled, never asserted.
    """
    return (
        f"Rank {int(row['rank'])}: {row['label']} adds a modelled "
        f"{row['coverage_added_pp']:.2f} percentage points of demand-weighted coverage, "
        f"bringing the running total to {row['cumulative_coverage']:.1%}. The nearest "
        f"existing device is about {row['nearest_existing_m']:.0f} m away; this site "
        f"closes a worst-case gap of roughly {row['worst_case_closed_m']:.0f} m. "
        "Planning estimate only — not a statement that this site is available or permitted."
    )
