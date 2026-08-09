"""Exact brute-force oracles for testing the optimisers.

An approximation algorithm needs something exact to be checked against. For very
small budgets the optimum is enumerable, so ``k = 1`` and ``k = 2`` give two
points where greedy's answer must match the true optimum exactly.

Welzl's smallest-enclosing-circle is deliberately **not** used. It solves the
*continuous* 1-centre problem, whereas this model is discrete -- sites must be
real buildings -- so its radius is a lower bound that is never attained. Testing
against it would assert something false.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from aed_siting.metrics import coverage


def brute_force_best(
    demand: np.ndarray,
    dist: np.ndarray,
    k: int,
    threshold: float,
    existing_covered: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Exhaustively find the best ``k`` sites.

    Args:
        demand: Per-cell weight.
        dist: One-way metres, shape ``(n_cells, n_candidates)``.
        k: Budget. Only 1 or 2 are practical.
        threshold: One-way metres.
        existing_covered: Cells already covered by the existing network.

    Returns:
        ``(sites, coverage)`` for the best combination, ties broken by lowest
        index so the result is unique and comparable to greedy.

    Limits:
        ``O(n^k)``. At a few hundred candidates ``k = 2`` is seconds and ``k = 3``
        is not worth it.
    """
    n_candidates = dist.shape[1]
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum()) or 1.0

    if existing_covered is not None:
        base = weights[np.asarray(existing_covered)].sum() / total
        reach = dist <= threshold
        open_cells = ~np.asarray(existing_covered)

        def score(combo: tuple[int, ...]) -> float:
            gained = reach[:, list(combo)].any(axis=1) & open_cells
            return base + float(weights[gained].sum() / total)
    else:

        def score(combo: tuple[int, ...]) -> float:
            return coverage(weights, dist, list(combo), threshold)

    best_sites: list[int] = []
    best_score = -1.0
    for combo in combinations(range(n_candidates), k):
        value = score(combo)
        if value > best_score + 1e-12:
            best_score, best_sites = value, list(combo)
    return best_sites, best_score
