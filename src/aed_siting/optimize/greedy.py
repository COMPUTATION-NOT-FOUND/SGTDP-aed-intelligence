"""Marginal-coverage greedy placement.

The method, as distinct from the demand-ranked baseline. Each pick maximises
demand covered *that is not already covered* -- by the existing network, or by
anything picked earlier in this run.

Two properties matter downstream:

* **Pre-seeded.** The existing AED network is folded in before the first pick, so
  the optimiser is asked "where should the next device go?" rather than "where
  would you put a network from scratch?". Solving from scratch would re-cover
  ground that is already covered and overstate every gain.
* **Nested.** ``select(k)`` is always a prefix of ``select(k + 1)``, so a planner
  who installs the first three this year and two more next year ends up exactly
  where a five-device plan would have put them. The ILP has no such guarantee,
  and the coverage cost of that difference is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedyResult:
    """An ordered install queue with the gain each step buys."""

    order: list[int]
    marginal_gain: list[float]
    cumulative_coverage: list[float]
    baseline_coverage: float

    def top(self, k: int) -> list[int]:
        """The first ``k`` sites. A prefix, by construction."""
        return self.order[:k]


def greedy_select(
    demand: np.ndarray,
    dist: np.ndarray,
    k: int,
    threshold: float,
    existing_covered: np.ndarray | None = None,
) -> GreedyResult:
    """Choose up to ``k`` sites, each maximising newly covered demand.

    Args:
        demand: Per-cell weight, length ``n_cells``.
        dist: One-way metres to candidates, shape ``(n_cells, n_candidates)``.
        k: Maximum devices to place.
        threshold: One-way metres.
        existing_covered: Boolean mask of cells already covered by the existing
            network. Omitting it solves from scratch, which is almost never what
            a planner wants.

    Returns:
        A :class:`GreedyResult` whose ``order`` is the install queue.

    Assumptions:
        Coverage is binary at the threshold, so the objective is submodular and
        greedy is within ``1 - 1/e`` (~63%) of optimal. That bound is a guarantee,
        not an estimate.

    Limits:
        Stops early when no remaining candidate adds any demand -- a queue shorter
        than ``k`` means the town is saturated at this threshold, which is itself
        a finding. Ties break to the lowest candidate index, making the result
        byte-reproducible.
    """
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum())
    n_cells, n_candidates = dist.shape

    covered = np.zeros(n_cells, dtype=bool) if existing_covered is None else existing_covered.copy()
    baseline = float(weights[covered].sum() / total) if total > 0 else 0.0

    reach = dist <= threshold
    chosen: list[int] = []
    gains: list[float] = []
    cumulative: list[float] = []
    available = np.ones(n_candidates, dtype=bool)

    for _ in range(min(k, n_candidates)):
        new_demand = (weights[:, None] * (reach & ~covered[:, None])).sum(axis=0)
        new_demand[~available] = -np.inf
        best = int(np.argmax(new_demand))  # ties -> lowest index
        gain = float(new_demand[best])
        if not np.isfinite(gain) or gain <= 0:
            break
        covered |= reach[:, best]
        available[best] = False
        chosen.append(best)
        gains.append(gain / total if total > 0 else 0.0)
        cumulative.append(float(weights[covered].sum() / total) if total > 0 else 0.0)

    return GreedyResult(
        order=chosen,
        marginal_gain=gains,
        cumulative_coverage=cumulative,
        baseline_coverage=baseline,
    )


def existing_coverage_mask(dist_existing: np.ndarray, threshold: float) -> np.ndarray:
    """Cells already covered by the existing network."""
    if dist_existing.size == 0:
        return np.zeros(dist_existing.shape[0], dtype=bool)
    return (dist_existing <= threshold).any(axis=1)


def leave_one_out_criticality(
    demand: np.ndarray,
    dist_existing: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Demand that becomes uncovered if each existing device is removed.

    Powers replacement-cycle guidance. A device with near-zero criticality is
    fully redundant: at end of life, the same spend elsewhere buys more coverage.
    A high-criticality device is a single point of failure worth prioritising for
    maintenance.

    Limits:
        This is emphatically **not** a recommendation to remove a working device.
        The snapshot is years old, the data carries no ownership or authorisation
        information, and asserting redundancy would imply knowing the device is
        present and working -- which this dataset cannot establish. It informs
        what to do when a device is *already* being replaced.
    """
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum())
    if total <= 0 or dist_existing.shape[1] == 0:
        return np.zeros(dist_existing.shape[1], dtype=float)

    reach = dist_existing <= threshold
    reach_count = reach.sum(axis=1)
    # A cell is lost only where this device is its sole cover.
    sole = reach & (reach_count[:, None] == 1)
    return (weights[:, None] * sole).sum(axis=0) / total
