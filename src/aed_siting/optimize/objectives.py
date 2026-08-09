"""Alternative placement objectives: worst-case and average distance.

Marginal-coverage greedy maximises how much demand falls inside a threshold. It
is indifferent to *how far* the rest has to go, which is why the evaluation
reports maximum retrieval distance as unchanged at 785 m however many devices are
added. Two other objectives address that directly:

``k-center``
    Minimise the worst-case distance any demand cell faces. The equity lens.
``p-median``
    Minimise the demand-weighted *average* distance. Between the other two.

All three run over the same discrete candidate set and the same pre-seeded
existing network, so their answers are directly comparable. Where they agree, a
recommendation is robust to the choice of objective -- a second, independent
robustness signal alongside the scenario sweep.

(The plan named separate ``kcenter.py`` and ``pmedian.py`` modules; they share
almost all of their structure, so they live together here.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObjectiveResult:
    """Sites chosen under one objective, with what they achieve."""

    objective: str
    sites: list[int]
    max_m: float
    mean_m: float
    guarantee: str


def _current(dist_existing: np.ndarray, n_cells: int) -> np.ndarray:
    """Distance from each cell to the nearest existing device."""
    if dist_existing.size == 0 or dist_existing.shape[1] == 0:
        return np.full(n_cells, np.inf)
    return dist_existing.min(axis=1)


def kcenter_minimax(dist: np.ndarray, dist_existing: np.ndarray, k: int) -> list[int]:
    """Greedily add the site that most reduces the worst-case distance.

    At each step every remaining candidate is scored by the maximum distance that
    would remain if it were added, and the best is taken.

    Limits:
        No approximation guarantee. The classical 2-approximation belongs to the
        farthest-point variant (:func:`kcenter_gonzalez`); this one is a heuristic
        that in practice does at least as well, and both are reported so the
        difference is visible rather than asserted.
    """
    current = _current(dist_existing, dist.shape[0])
    chosen: list[int] = []
    available = np.ones(dist.shape[1], dtype=bool)

    for _ in range(min(k, dist.shape[1])):
        resulting = np.minimum(current[:, None], dist).max(axis=0)
        resulting[~available] = np.inf
        best = int(np.argmin(resulting))  # ties -> lowest index
        if not np.isfinite(resulting[best]):
            break
        current = np.minimum(current, dist[:, best])
        available[best] = False
        chosen.append(best)
    return chosen


def kcenter_gonzalez(dist: np.ndarray, dist_existing: np.ndarray, k: int) -> list[int]:
    """Farthest-point heuristic: serve the worst-off cell, repeat.

    Finds the demand cell currently farthest from any device, then places at the
    candidate nearest to it.

    Assumptions:
        Carries the classical 2-approximation for k-center in a metric space. The
        bound assumes candidates are dense relative to demand points; with a
        restricted candidate set it is a guide rather than a proof, which is why
        the minimax variant is reported alongside it.
    """
    current = _current(dist_existing, dist.shape[0])
    chosen: list[int] = []
    available = np.ones(dist.shape[1], dtype=bool)

    for _ in range(min(k, dist.shape[1])):
        worst_cell = int(np.argmax(current))
        if not np.isfinite(current[worst_cell]) and not available.any():
            break
        options = np.where(available, dist[worst_cell], np.inf)
        best = int(np.argmin(options))
        if not np.isfinite(options[best]):
            break
        current = np.minimum(current, dist[:, best])
        available[best] = False
        chosen.append(best)
    return chosen


def pmedian_greedy(
    demand: np.ndarray, dist: np.ndarray, dist_existing: np.ndarray, k: int
) -> list[int]:
    """Greedily add the site that most reduces demand-weighted average distance.

    Limits:
        Averages hide tails by construction: a site serving many people slightly
        better outranks one serving a few people far better. That is exactly the
        trade-off against k-center, and reporting both is the point.
    """
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum()) or 1.0
    current = _current(dist_existing, dist.shape[0])
    chosen: list[int] = []
    available = np.ones(dist.shape[1], dtype=bool)

    for _ in range(min(k, dist.shape[1])):
        resulting = (weights[:, None] * np.minimum(current[:, None], dist)).sum(axis=0) / total
        resulting[~available] = np.inf
        best = int(np.argmin(resulting))
        if not np.isfinite(resulting[best]):
            break
        current = np.minimum(current, dist[:, best])
        available[best] = False
        chosen.append(best)
    return chosen


def _score(
    demand: np.ndarray, dist: np.ndarray, dist_existing: np.ndarray, sites: list[int]
) -> tuple[float, float]:
    """Worst-case and demand-weighted mean distance after placing ``sites``."""
    current = _current(dist_existing, dist.shape[0])
    if sites:
        current = np.minimum(current, dist[:, sites].min(axis=1))
    weights = np.asarray(demand, dtype=float)
    finite = np.isfinite(current)
    total = float(weights[finite].sum()) or 1.0
    return (
        float(current[finite].max()) if finite.any() else float("inf"),
        float((weights[finite] * current[finite]).sum() / total),
    )


def compare_objectives(
    demand: np.ndarray,
    dist: np.ndarray,
    dist_existing: np.ndarray,
    k: int,
    threshold: float,
    coverage_sites: list[int],
) -> tuple[pd.DataFrame, list[int]]:
    """Run every objective and report what each one buys.

    Args:
        demand: Per-cell weight.
        dist: One-way metres to candidates, shape ``(n_cells, n_candidates)``.
        dist_existing: One-way metres to the existing network.
        k: Devices to place under each objective.
        threshold: One-way metres, for the coverage column.
        coverage_sites: Sites already chosen by marginal-coverage greedy, so the
            objectives are compared rather than recomputed.

    Returns:
        ``(table, agreed)`` where ``agreed`` lists the sites every objective
        selects -- robust to the choice of objective, not just to the demand
        assumption.
    """
    from aed_siting.metrics import coverage

    runs = {
        "max coverage (greedy)": (coverage_sites, "1-1/e, ILP-certified optimal here"),
        "min worst case (k-center, minimax)": (
            kcenter_minimax(dist, dist_existing, k),
            "heuristic",
        ),
        "min worst case (k-center, Gonzalez)": (
            kcenter_gonzalez(dist, dist_existing, k),
            "2-approx in a metric space",
        ),
        "min average (p-median)": (
            pmedian_greedy(demand, dist, dist_existing, k),
            "heuristic",
        ),
    }

    baseline_max, baseline_mean = _score(demand, dist, dist_existing, [])
    rows = [
        {
            "objective": "do nothing",
            "coverage": round(coverage(demand, dist, [], threshold), 4),
            "max_m": round(baseline_max, 1),
            "mean_m": round(baseline_mean, 1),
            "guarantee": "-",
        }
    ]
    for name, (sites, guarantee) in runs.items():
        worst, mean = _score(demand, dist, dist_existing, sites)
        combined = np.concatenate([dist, dist_existing], axis=1)
        idx = list(sites) + list(range(dist.shape[1], combined.shape[1]))
        rows.append(
            {
                "objective": name,
                "coverage": round(coverage(demand, combined, idx, threshold), 4),
                "max_m": round(worst, 1),
                "mean_m": round(mean, 1),
                "guarantee": guarantee,
            }
        )

    selections = [set(sites) for sites, _ in runs.values()]
    agreed = sorted(set.intersection(*selections)) if selections else []
    return pd.DataFrame(rows), agreed
