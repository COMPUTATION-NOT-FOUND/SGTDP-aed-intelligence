"""Baselines the method must beat.

The challenge brief's minimum baseline for this lane is *the existing network
plus a transparent greedy or population-ranked placement rule*. Two distinct
things are therefore named carefully throughout this project:

``demand_ranked_topk``
    The **baseline**. Picks the *k* candidates sitting in the highest-demand
    cells. It never asks whether that demand is already covered, so it stacks
    devices into dense cores and re-covers covered ground.

marginal-coverage greedy
    The **method** (see :mod:`aed_siting.optimize.greedy`). Each pick maximises
    *newly* covered demand, given the existing network and everything picked
    before it.

Calling both "greedy" would make the headline appear to beat itself. The contrast
between them is a result, not a technicality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aed_siting.config import SEED


def do_nothing() -> list[int]:
    """Add no devices. The floor every other option must clear."""
    return []


def demand_ranked_topk(
    demand: np.ndarray,
    dist: np.ndarray,
    k: int,
    threshold: float,
) -> list[int]:
    """Baseline: the *k* candidates covering the most demand, ignoring overlap.

    Args:
        demand: Per-cell weight.
        dist: One-way metres, shape ``(n_cells, n_candidates)``.
        k: Number of devices to place.
        threshold: One-way metres.

    Returns:
        Candidate indices, best first. Ties break to the lowest index so the
        result is deterministic.

    Limits:
        Deliberately naive: each candidate is scored in isolation against total
        demand, so two adjacent sites serving the same cells both score highly
        and both get picked. That redundancy is exactly what a marginal-coverage
        method avoids, and quantifying the gap is the point of running this.
    """
    if k <= 0 or dist.shape[1] == 0:
        return []
    weights = np.asarray(demand, dtype=float)
    standalone = (dist <= threshold).T @ weights
    order = np.lexsort((np.arange(dist.shape[1]), -standalone))
    return order[: min(k, dist.shape[1])].tolist()


def random_k(
    n_candidates: int,
    k: int,
    seed: int = SEED,
    trials: int = 50,
) -> list[list[int]]:
    """Uncertainty band: ``trials`` independent random selections of *k* sites.

    Gives the spread a method must clear to be distinguishable from luck. Seeded,
    so the band is reproducible.
    """
    if k <= 0 or n_candidates == 0:
        return [[] for _ in range(trials)]
    rng = np.random.default_rng(seed)
    size = min(k, n_candidates)
    return [
        sorted(rng.choice(n_candidates, size=size, replace=False).tolist()) for _ in range(trials)
    ]


def coverage_band(
    coverages: Sequence[float],
) -> dict[str, float]:
    """Summarise a set of random-trial coverages as a band."""
    values = np.asarray(list(coverages), dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "p05": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }
