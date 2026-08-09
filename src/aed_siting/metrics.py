"""Coverage, retrieval-distance and equity metrics.

**Where round-trip lives.** Exactly one place: the ``/ 2`` in
:func:`aed_siting.config.threshold_m`, which turns a round-trip time budget into
the one-way distance a responder can cover. Distance matrices are one-way and
thresholds are one-way, so nothing here doubles anything.

**Why there is no Monte Carlo.** Demand is already a weight per cell and coverage
is evaluated at cell centroids, so sampling incidents from that distribution
would re-derive, with noise, a quantity computable exactly: the demand-weighted
quantiles of per-cell retrieval distance. Every cell centroid *is* a scripted
scenario, weighted by demand. Exact, instant, and deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aed_siting.config import RETRIEVAL_BUDGET_MIN, WALK_SPEED_MS, threshold_m


def covered_mask(
    dist: np.ndarray, sites: Sequence[int] | np.ndarray, threshold: float
) -> np.ndarray:
    """Boolean per demand cell: is any selected site within the threshold?

    Args:
        dist: One-way metres, shape ``(n_cells, n_sites)``.
        sites: Column indices of the selected sites.
        threshold: One-way metres, from :func:`threshold_m`.

    Returns:
        Boolean array of length ``n_cells``. All-False when nothing is selected.
    """
    sites = np.asarray(list(sites), dtype=int)
    if sites.size == 0:
        return np.zeros(dist.shape[0], dtype=bool)
    return (dist[:, sites] <= threshold).any(axis=1)


def coverage(
    demand: np.ndarray,
    dist: np.ndarray,
    sites: Sequence[int] | np.ndarray,
    threshold: float | None = None,
) -> float:
    """Share of demand within round-trip retrieval range of a selected site.

    Args:
        demand: Per-cell weight, length ``n_cells``. Need not be normalised.
        dist: One-way metres, shape ``(n_cells, n_sites)``.
        sites: Column indices of the selected sites.
        threshold: One-way metres; defaults to the pre-registered 3-minute budget.

    Returns:
        Demand-weighted covered fraction in ``[0, 1]``.

    Assumptions:
        Coverage is binary at the threshold -- no distance decay. A cell one metre
        beyond is uncovered, which overstates the sharpness of a real response.

    Limits:
        Says nothing about whether a device is present, working, or accessible. A
        covered cell is a planning estimate, never an availability claim.
    """
    threshold = threshold_m() if threshold is None else threshold
    total = float(np.sum(demand))
    if total <= 0:
        return 0.0
    return float(np.sum(demand[covered_mask(dist, sites, threshold)]) / total)


def nearest(dist: np.ndarray, sites: Sequence[int] | np.ndarray) -> np.ndarray:
    """One-way distance from each cell to its nearest selected site."""
    sites = np.asarray(list(sites), dtype=int)
    if sites.size == 0:
        return np.full(dist.shape[0], np.inf)
    return dist[:, sites].min(axis=1)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float | Sequence[float]
) -> np.ndarray:
    """Exact demand-weighted quantiles.

    Replaces Monte Carlo sampling: with demand as a weight, the quantiles of the
    retrieval-distance distribution are computable in closed form.
    """
    quantiles = np.atleast_1d(np.asarray(q, dtype=float))
    finite = np.isfinite(values)
    if not finite.any() or float(weights[finite].sum()) <= 0:
        return np.full(quantiles.shape, np.inf)

    order = np.argsort(values[finite], kind="stable")
    ordered_values = values[finite][order]
    ordered_weights = weights[finite][order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return np.interp(quantiles, cumulative, ordered_values)


def distance_to_seconds(metres: float | np.ndarray) -> float | np.ndarray:
    """Convert a one-way distance into a round-trip retrieval time in seconds."""
    return 2.0 * np.asarray(metres, dtype=float) / WALK_SPEED_MS


@dataclass(frozen=True)
class CoverageResult:
    """Everything reported about one selection of sites."""

    coverage: float
    median_m: float
    p90_m: float
    p95_m: float
    max_m: float
    uncovered_demand: float

    @property
    def median_seconds(self) -> float:
        """Median round-trip retrieval time."""
        return float(distance_to_seconds(self.median_m))

    @property
    def p90_seconds(self) -> float:
        """90th-percentile round-trip retrieval time."""
        return float(distance_to_seconds(self.p90_m))

    def as_row(self) -> dict[str, float]:
        """Flat dict for assembling evaluation tables."""
        return {
            "coverage": round(self.coverage, 4),
            "median_m": round(self.median_m, 1),
            "p90_m": round(self.p90_m, 1),
            "p95_m": round(self.p95_m, 1),
            "max_m": round(self.max_m, 1),
            "p90_s": round(self.p90_seconds, 1),
            "uncovered_demand": round(self.uncovered_demand, 4),
        }


def evaluate(
    demand: np.ndarray,
    dist: np.ndarray,
    sites: Sequence[int] | np.ndarray,
    threshold: float | None = None,
) -> CoverageResult:
    """Full metric set for one selection.

    Averages alone are insufficient for this challenge, so tail statistics (p90,
    p95, max) are computed alongside the median and reported together.
    """
    threshold = threshold_m() if threshold is None else threshold
    weights = np.asarray(demand, dtype=float)
    distances = nearest(dist, sites)
    quantiles = weighted_quantile(distances, weights, [0.5, 0.9, 0.95])
    finite = distances[np.isfinite(distances)]
    return CoverageResult(
        coverage=coverage(weights, dist, sites, threshold),
        median_m=float(quantiles[0]),
        p90_m=float(quantiles[1]),
        p95_m=float(quantiles[2]),
        max_m=float(finite.max()) if finite.size else float("inf"),
        uncovered_demand=float(
            weights[~covered_mask(dist, sites, threshold)].sum() / max(weights.sum(), 1e-12)
        ),
    )


def catchment_load(
    demand: np.ndarray, dist: np.ndarray, sites: Sequence[int] | np.ndarray
) -> pd.Series:
    """Demand assigned to each site under nearest-facility allocation.

    Binary coverage hides congestion: a cell can be "covered" by a device that
    thousands of other people also depend on. High load is a case for a *second*
    device in an area that already reads as covered.
    """
    sites = np.asarray(list(sites), dtype=int)
    if sites.size == 0:
        return pd.Series(dtype=float)
    assignment = sites[dist[:, sites].argmin(axis=1)]
    return (
        pd.Series(np.asarray(demand, dtype=float))
        .groupby(assignment)
        .sum()
        .reindex(sites, fill_value=0.0)
        .rename("demand_served")
    )


def subzone_coverage(
    demand: np.ndarray,
    dist: np.ndarray,
    sites: Sequence[int] | np.ndarray,
    subzone_of_cell: pd.Series,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Coverage broken down by subzone, for the worst-served-area equity metric.

    A town-wide average can look healthy while one subzone is entirely uncovered.
    The minimum of this table, not its mean, is the equity number that matters.
    """
    threshold = threshold_m() if threshold is None else threshold
    weights = np.asarray(demand, dtype=float)
    covered = covered_mask(dist, sites, threshold)
    frame = pd.DataFrame(
        {"subzone": subzone_of_cell.to_numpy(), "demand": weights, "covered": covered}
    )
    grouped = frame.groupby("subzone", dropna=False).apply(
        lambda g: pd.Series(
            {
                "demand": g["demand"].sum(),
                "coverage": (
                    g.loc[g["covered"], "demand"].sum() / g["demand"].sum()
                    if g["demand"].sum() > 0
                    else 0.0
                ),
            }
        ),
        include_groups=False,
    )
    return grouped.sort_values("coverage").round(4)


def summarise_threshold_sweep(
    demand: np.ndarray,
    dist: np.ndarray,
    sites: Sequence[int] | np.ndarray,
    budgets_min: Sequence[float] = (2.0, 3.0, RETRIEVAL_BUDGET_MIN, 4.0),
) -> pd.DataFrame:
    """Coverage across retrieval-time budgets, so the headline cannot be cherry-picked."""
    rows = []
    for budget in sorted(set(budgets_min)):
        result = evaluate(demand, dist, sites, threshold_m(budget))
        rows.append(
            {"budget_min": budget, "threshold_m": round(threshold_m(budget), 1), **result.as_row()}
        )
    return pd.DataFrame(rows)
