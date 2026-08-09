"""Scenario and sensitivity analysis.

The brief asks how recommendations change when demand assumptions or constraints
change. Three axes answer that:

* **Demand proxy** -- solve under each proxy and score under all of them. The
  diagonal of that matrix is the optimistic case; the off-diagonal is the honest
  one. This is a *sensitivity* analysis, not held-out validation: there are no
  labels, so nothing here is validated real-world performance.
* **Threshold** -- the retrieval-time budget is swept so a headline cannot be
  chosen after the fact.
* **Hours of access** -- a night scenario admitting only devices that state
  round-the-clock cover.

Sites chosen under every scenario are the robust recommendations; sites that
appear and disappear are the uncertainty, and are reported as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aed_siting.config import RETRIEVAL_BUDGET_SWEEP_MIN, threshold_m
from aed_siting.metrics import coverage
from aed_siting.optimize.greedy import existing_coverage_mask, greedy_select


def cross_proxy_sensitivity(
    demand: pd.DataFrame,
    dist: np.ndarray,
    dist_existing: np.ndarray,
    k: int,
    threshold: float,
    proxies: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Solve under each demand proxy, then score the result under every proxy.

    Returns:
        A matrix indexed by the proxy solved under, with one column per proxy
        scored under. Off-diagonal cells answer "if we optimised for the wrong
        demand surface, how much coverage would we still get?".

    Limits:
        No ground truth exists, so this measures *robustness to an assumption*,
        never accuracy. Describing it as validation would overclaim.
    """
    names = proxies or tuple(demand.columns)
    rows = {}
    for solved_under in names:
        weights = demand[solved_under].to_numpy()
        already = existing_coverage_mask(dist_existing, threshold)
        picked = greedy_select(weights, dist, k, threshold, already).order
        rows[solved_under] = {
            scored_under: round(
                coverage(demand[scored_under].to_numpy(), dist, picked, threshold), 4
            )
            for scored_under in names
        }
    return pd.DataFrame(rows).T.rename_axis(index="solved_under", columns="scored_under")


def threshold_sweep(
    demand: np.ndarray,
    dist: np.ndarray,
    dist_existing: np.ndarray,
    k: int,
    budgets_min: tuple[float, ...] = RETRIEVAL_BUDGET_SWEEP_MIN,
) -> pd.DataFrame:
    """Re-solve at each retrieval-time budget.

    A tighter budget shrinks every coverage figure; what matters is whether the
    *recommendation* survives, which the site lists here allow a reader to check.
    """
    rows = []
    for budget in budgets_min:
        limit = threshold_m(budget)
        already = existing_coverage_mask(dist_existing, limit)
        result = greedy_select(demand, dist, k, limit, already)
        rows.append(
            {
                "budget_min": budget,
                "threshold_m": round(limit, 1),
                "baseline_coverage": round(result.baseline_coverage, 4),
                "coverage_after": round(
                    result.cumulative_coverage[-1]
                    if result.cumulative_coverage
                    else result.baseline_coverage,
                    4,
                ),
                "sites_placed": len(result.order),
                "sites": result.order,
            }
        )
    return pd.DataFrame(rows)


def hours_scenario(
    demand: np.ndarray,
    dist: np.ndarray,
    dist_existing_all: np.ndarray,
    dist_existing_247: np.ndarray,
    k: int,
    threshold: float,
) -> pd.DataFrame:
    """Compare all-hours cover against devices stating round-the-clock access.

    Limits:
        These are *stated* hours from a February 2020 snapshot, not verified
        present access. Nothing here may be rendered as "open now".
    """
    rows = []
    for label, existing in (("all_hours", dist_existing_all), ("stated_24_7", dist_existing_247)):
        already = existing_coverage_mask(existing, threshold)
        result = greedy_select(demand, dist, k, threshold, already)
        rows.append(
            {
                "scenario": label,
                "existing_devices": existing.shape[1],
                "baseline_coverage": round(result.baseline_coverage, 4),
                "coverage_after": round(
                    result.cumulative_coverage[-1]
                    if result.cumulative_coverage
                    else result.baseline_coverage,
                    4,
                ),
                "sites": result.order,
            }
        )
    return pd.DataFrame(rows)


def consensus_sites(selections: dict[str, list[int]], min_share: float = 0.9) -> list[int]:
    """Sites chosen in at least ``min_share`` of the given scenarios.

    These are the recommendations to lead with; everything else is reported as
    scenario-dependent rather than presented as settled.
    """
    if not selections:
        return []
    counts: dict[int, int] = {}
    for picked in selections.values():
        for site in picked:
            counts[site] = counts.get(site, 0) + 1
    needed = min_share * len(selections)
    return sorted(site for site, n in counts.items() if n >= needed)


def stability_table(selections: dict[str, list[int]]) -> pd.DataFrame:
    """How often each site is chosen across scenarios."""
    counts: dict[int, int] = {}
    for picked in selections.values():
        for site in picked:
            counts[site] = counts.get(site, 0) + 1
    total = max(len(selections), 1)
    return (
        pd.DataFrame({"site_id": list(counts), "times_chosen": list(counts.values())})
        .assign(share=lambda d: (d["times_chosen"] / total).round(3))
        .sort_values(["times_chosen", "site_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
