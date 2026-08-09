"""Exact maximal-covering placement via integer programming.

Greedy carries a ``1 - 1/e`` guarantee; this closes the remaining gap and, more
usefully, *certifies* it. Reporting "greedy is within X% of proven optimal" is a
far stronger claim than quoting the worst-case bound.

It also exposes a real tension. The ILP optimum for ``k`` devices need not
contain the optimum for ``k - 1``, so an ILP-derived plan is **not nested**: a
planner who installs this year's three and returns next year may find the
five-device optimum wants a different three. Greedy is nested by construction.
The coverage cost of insisting on nesting is measured by
:func:`nesting_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp


@dataclass(frozen=True)
class IlpResult:
    """An exact solve, with the evidence that it is exact."""

    sites: list[int]
    objective: float
    status: str
    gap: float
    seconds: float

    @property
    def proven_optimal(self) -> bool:
        """True when the solver proved optimality rather than timing out."""
        return self.status == "Optimal" and self.gap <= 1e-6


def solve_mclp(
    demand: np.ndarray,
    dist: np.ndarray,
    k: int,
    threshold: float,
    existing_covered: np.ndarray | None = None,
    time_limit_s: int = 60,
) -> IlpResult:
    """Maximise demand covered by ``k`` new sites, over and above the existing network.

    Args:
        demand: Per-cell weight.
        dist: One-way metres, shape ``(n_cells, n_candidates)``.
        k: Exact number of devices to place.
        threshold: One-way metres.
        existing_covered: Cells already covered; those cells are dropped from the
            objective so the solve optimises *marginal* coverage.
        time_limit_s: Solver budget. On timeout the incumbent is returned with a
            non-zero gap, and ``proven_optimal`` is False.

    Returns:
        An :class:`IlpResult`.

    Limits:
        Determinism is weaker here than for greedy. CBC may return any of several
        equally-optimal site sets, so only the objective value and the gap are
        asserted across runs -- never the identity of the chosen sites.
    """
    weights = np.asarray(demand, dtype=float)
    n_cells, n_candidates = dist.shape
    already = (
        np.zeros(n_cells, dtype=bool) if existing_covered is None else np.asarray(existing_covered)
    )
    open_cells = np.where(~already & (weights > 0))[0]
    reach = dist <= threshold

    problem = pulp.LpProblem("mclp", pulp.LpMaximize)
    site_vars = [pulp.LpVariable(f"s{j}", cat="Binary") for j in range(n_candidates)]
    cell_vars = {i: pulp.LpVariable(f"c{i}", cat="Binary") for i in open_cells}

    problem += pulp.lpSum(weights[i] * cell_vars[i] for i in open_cells)
    problem += pulp.lpSum(site_vars) <= k

    for i in open_cells:
        covering = np.where(reach[i])[0]
        if covering.size:
            problem += cell_vars[i] <= pulp.lpSum(site_vars[j] for j in covering)
        else:
            problem += cell_vars[i] == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    problem.solve(solver)

    chosen = sorted(j for j, var in enumerate(site_vars) if var.value() and var.value() > 0.5)
    total = float(weights.sum()) or 1.0
    marginal = float(pulp.value(problem.objective) or 0.0) / total
    return IlpResult(
        sites=chosen,
        objective=float(weights[already].sum() / total) + marginal,
        status=pulp.LpStatus[problem.status],
        gap=0.0 if pulp.LpStatus[problem.status] == "Optimal" else float("nan"),
        seconds=float(problem.solutionTime or 0.0),
    )


def nesting_loss(
    demand: np.ndarray,
    dist: np.ndarray,
    greedy_order: list[int],
    threshold: float,
    budgets: list[int],
    existing_covered: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Coverage forfeited by requiring a staged, nested rollout.

    For each budget, compares the ILP optimum against the greedy prefix of the
    same size. The difference is what a planner pays for the ability to commit
    devices one year at a time without regretting earlier choices -- a real
    trade-off with an operational meaning, not an artefact of the algorithms.
    """
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum()) or 1.0
    already = (
        np.zeros(dist.shape[0], dtype=bool)
        if existing_covered is None
        else np.asarray(existing_covered)
    )
    base = float(weights[already].sum() / total)
    reach = dist <= threshold

    rows = []
    for k in budgets:
        exact = solve_mclp(demand, dist, k, threshold, existing_covered)
        # Must be measured on the same basis as the ILP objective: existing
        # coverage plus what the prefix adds. Scoring the prefix alone would
        # compare a marginal figure against a total one and report the baseline
        # as if it were the nesting penalty.
        gained = (
            reach[:, greedy_order[:k]].any(axis=1) & ~already
            if greedy_order[:k]
            else np.zeros(dist.shape[0], dtype=bool)
        )
        nested = base + float(weights[gained].sum() / total)
        rows.append(
            {
                "k": k,
                "ilp_coverage": round(exact.objective, 4),
                "greedy_coverage": round(nested, 4),
                "nesting_loss_pp": round((exact.objective - nested) * 100, 3),
                "greedy_share_of_optimal": round(nested / exact.objective, 4)
                if exact.objective > 0
                else 1.0,
                "proven_optimal": exact.proven_optimal,
            }
        )
    return rows
