"""Placement policy as an *ordering of criteria*.

A planner does not want one fixed objective. They want to say "widest gap first,
then demand" in one scenario and "demand first, then gap" in another, and see how
the plan changes. So the policy here is a **list of sort keys**, applied
lexicographically at each greedy step.

Every previous objective is an ordering of this, which is why this module replaces
them rather than sitting beside them:

=========================  ===========================
``["demand", "gap"]``      maximum coverage
``["gap", "demand"]``      worst-served areas first
``["mean", "demand"]``     shortest average trip
``["relief", "demand"]``   relieve over-subscribed devices
=========================  ===========================

**Why banding is not optional.** Ranking continuous scores lexicographically never
reaches the second key: exact float ties essentially never occur, so the sort
silently collapses to single-key and every ordering produces the same plan. Each
key therefore admits a *band* — candidates within a relative tolerance of the best
are treated as tied and separated by the next key.

Banding also repairs a real defect. The old `kcenter_minimax` returned
``[184, 0, 1, 2, 3, 4, 5, 6, 7, 8]``: once the worst-served cell became
unreachable, **all 791 candidates scored identically**, `argmin` returned the
lowest index, and nine of ten picks were arbitrary buildings presented as an
equity plan. With a band, a saturated key hands the decision to the next one
instead of to the index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Candidates within this fraction of the best score on a key count as tied.
DEFAULT_TOLERANCE = 0.02

KEYS: tuple[str, ...] = ("demand", "gap", "relief", "mean", "equity")

KEY_LABELS: dict[str, str] = {
    "demand": "Demand newly covered",
    "gap": "Shrinks the worst gap",
    "relief": "Relieves busy devices",
    "mean": "Shortens average trip",
    "equity": "Helps the worst subzone",
}

#: Named orderings offered in the UI. "Custom" is assembled by the user.
PRESETS: dict[str, tuple[str, ...]] = {
    "Most people reached": ("demand", "gap"),
    "Worst-served areas first": ("gap", "demand"),
    "Shortest average trip": ("mean", "demand"),
    "Relieve busiest devices": ("relief", "demand"),
    "Worst subzone first": ("equity", "demand"),
}


@dataclass
class PolicyResult:
    """An ordered install queue produced by one criterion ordering."""

    keys: tuple[str, ...]
    order: list[int] = field(default_factory=list)
    marginal_gain: list[float] = field(default_factory=list)
    cumulative_coverage: list[float] = field(default_factory=list)
    #: Which key actually decided each pick, after banding.
    deciding_key: list[str] = field(default_factory=list)
    baseline_coverage: float = 0.0

    def top(self, k: int) -> list[int]:
        """The first ``k`` sites. A prefix, by construction."""
        return self.order[:k]


def _scores(
    key: str,
    weights: np.ndarray,
    dist: np.ndarray,
    current: np.ndarray,
    covered: np.ndarray,
    threshold: float,
    subzone_ids: np.ndarray | None,
) -> np.ndarray:
    """Score every candidate on one criterion. Higher is always better."""
    total = float(weights.sum()) or 1.0

    if key == "demand":
        reach = dist <= threshold
        return (weights[:, None] * (reach & ~covered[:, None])).sum(axis=0) / total

    if key == "gap":
        # Reduction in the worst-case distance anyone faces.
        return current.max() - np.minimum(current[:, None], dist).max(axis=0)

    if key == "mean":
        before = float((weights * current).sum() / total)
        return before - (weights[:, None] * np.minimum(current[:, None], dist)).sum(axis=0) / total

    if key == "relief":
        # Demand currently carried by whichever device is nearest, that this
        # candidate would take over -- i.e. load lifted off the busiest server.
        taken = dist < current[:, None]
        return (weights[:, None] * taken).sum(axis=0) / total

    if key == "equity":
        # Silently degrading to a zero score would make the key a no-op; a
        # mismatched label array is a caller error worth surfacing.
        if subzone_ids is None:
            return np.zeros(dist.shape[1])
        if len(subzone_ids) != dist.shape[0]:
            raise ValueError(f"subzone_ids has {len(subzone_ids)} labels for {dist.shape[0]} cells")
        reach = dist <= threshold
        gained = weights[:, None] * (reach & ~covered[:, None])
        # Improvement concentrated in the currently worst-served subzone.
        worst = _worst_subzone(weights, covered, subzone_ids)
        in_worst = (subzone_ids == worst)[:, None]
        return (gained * in_worst).sum(axis=0) / total

    raise ValueError(f"unknown policy key: {key!r} (expected one of {KEYS})")


def _worst_subzone(weights: np.ndarray, covered: np.ndarray, subzone_ids: np.ndarray) -> int:
    """Index of the subzone with the lowest covered share of its own demand."""
    best_id, best_share = -1, np.inf
    for value in np.unique(subzone_ids):
        mask = subzone_ids == value
        demand = weights[mask].sum()
        if demand <= 0:
            continue
        share = weights[mask & covered].sum() / demand
        if share < best_share:
            best_id, best_share = int(value), share
    return best_id


def rank_candidates(
    keys: tuple[str, ...],
    scores: dict[str, np.ndarray],
    available: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[int, str]:
    """Pick the best candidate by applying ``keys`` lexicographically with banding.

    Args:
        keys: Criterion names, most important first.
        scores: One score array per key; higher is better.
        available: Boolean mask of candidates still selectable.
        tolerance: Relative band. Candidates within this fraction of the best
            score on a key are treated as tied and separated by the next key.

    Returns:
        ``(index, deciding_key)`` — the chosen candidate and the key that actually
        narrowed the field to it. ``deciding_key`` is ``"index"`` when every key
        saturated and the lowest-index tie-break resolved it.

    Limits:
        Banding is what makes a second key reachable at all, but it also means the
        winner need not be the strict maximiser of key 1 — it is the best on a
        later key among those *near* the top of key 1. That trade is the point:
        a 1% concession on the primary can buy a large gain on the secondary.
    """
    field_mask = available.copy()
    deciding = "index"
    if not field_mask.any():
        return -1, deciding

    for key in keys:
        values = np.where(field_mask, scores[key], -np.inf)
        best = float(values.max())
        if not np.isfinite(best):
            break
        # Relative band, with an absolute floor so scores near zero still band.
        band = max(abs(best) * tolerance, 1e-12)
        keep = field_mask & (values >= best - band)
        if keep.sum() < field_mask.sum():
            deciding = key
        field_mask = keep
        if field_mask.sum() == 1:
            break

    return int(np.argmax(field_mask)), deciding  # ties -> lowest index


def build_queue(
    keys: tuple[str, ...],
    weights: np.ndarray,
    dist: np.ndarray,
    dist_existing: np.ndarray,
    k: int,
    threshold: float,
    subzone_ids: np.ndarray | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    pinned: tuple[int, ...] = (),
    vetoed: tuple[int, ...] = (),
) -> PolicyResult:
    """Build a ranked install queue under one criterion ordering.

    Args:
        keys: Criterion names, most important first.
        weights: Per-cell demand.
        dist: One-way metres to candidates, ``(n_cells, n_candidates)``.
        dist_existing: One-way metres to the existing network.
        k: Devices to place.
        threshold: One-way metres.
        subzone_ids: Integer subzone label per cell; required by the ``equity`` key.
        tolerance: Banding tolerance.
        pinned: Candidate indices a human requires, taken first in the given order.
        vetoed: Candidate indices a human excludes.

    Human overrides are applied as *constraints on the solve*, not as edits to a
    finished plan: every subsequent pick is then chosen against the state the pins
    actually create. Layering them afterwards would leave the rest of the queue
    optimised for a plan that no longer exists.

    Returns:
        A :class:`PolicyResult`. The sequence is nested — ``top(k)`` is always a
        prefix of ``top(k+1)`` — so a staged budget never invalidates an earlier
        choice, whatever the ordering.

    Assumptions:
        Every criterion is evaluated against the state *after* previous picks, so
        the queue is genuinely marginal rather than a batch ranking.
    """
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum()) or 1.0
    n_cells, n_candidates = dist.shape

    current = (
        dist_existing.min(axis=1)
        if dist_existing.size and dist_existing.shape[1]
        else np.full(n_cells, np.inf)
    )
    covered = current <= threshold
    result = PolicyResult(keys=tuple(keys), baseline_coverage=float(weights[covered].sum() / total))
    available = np.ones(n_candidates, dtype=bool)
    available[list(vetoed)] = False

    # Pins are honoured first and in order, so a planner's non-negotiable sites
    # appear at the top of the queue and everything after is optimised given them.
    for site in pinned:
        if site >= n_candidates or not available[site] or len(result.order) >= k:
            continue
        reach = dist[:, site] <= threshold
        gained = float(weights[reach & ~covered].sum() / total)
        current = np.minimum(current, dist[:, site])
        covered = covered | reach
        available[site] = False
        result.order.append(int(site))
        result.marginal_gain.append(gained)
        result.deciding_key.append("pinned")
        result.cumulative_coverage.append(float(weights[covered].sum() / total))

    for _ in range(min(k, n_candidates) - len(result.order)):
        scores = {
            key: _scores(key, weights, dist, current, covered, threshold, subzone_ids)
            for key in keys
        }
        choice, deciding = rank_candidates(tuple(keys), scores, available, tolerance)
        if choice < 0:
            break

        gained = float(scores.get("demand", np.zeros(n_candidates))[choice])
        if "demand" not in scores:
            reach = dist[:, choice] <= threshold
            gained = float(weights[reach & ~covered].sum() / total)

        # Stop when no criterion can separate the remaining candidates.
        #
        # `deciding == "index"` means every key saturated and the lowest-index
        # tie-break resolved the pick -- i.e. the choice is arbitrary. Stopping
        # there is what makes the queue length meaningful: "Shortest average trip"
        # otherwise ran to all 791 candidates in Toa Payoh (4,185 in Bedok) with
        # 681 zero-gain picks and 511 decided by index, because `mean` always
        # yields a vanishing positive improvement and never trips a `<= 0` test.
        if deciding == "index":
            break
        if gained <= 0 and all(scores[key][choice] <= 0 for key in keys):
            break

        current = np.minimum(current, dist[:, choice])
        covered = covered | (dist[:, choice] <= threshold)
        available[choice] = False
        result.order.append(choice)
        result.marginal_gain.append(gained)
        result.deciding_key.append(deciding)
        result.cumulative_coverage.append(float(weights[covered].sum() / total))

    return result


def covered_at_step(
    order: list[int], dist: np.ndarray, dist_existing: np.ndarray, threshold: float
) -> np.ndarray:
    """For each cell, the queue step at which it first becomes covered.

    ``0`` means the existing network already covers it, ``k`` that the *k*-th
    funded device first does, ``-1`` that nothing in the queue ever does.

    This is what makes the animation a replay of the real result rather than a
    decoration: at step *k*, a cell is covered iff ``0 <= value <= k``.
    """
    n_cells = dist.shape[0]
    already = (
        (dist_existing <= threshold).any(axis=1)
        if dist_existing.size and dist_existing.shape[1]
        else np.zeros(n_cells, dtype=bool)
    )
    step = np.where(already, 0, -1)
    for position, site in enumerate(order, start=1):
        newly = (dist[:, site] <= threshold) & (step == -1)
        step[newly] = position
    return step
