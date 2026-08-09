"""Gates on the placement optimisers and the simulation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aed_siting.metrics import coverage, weighted_quantile
from aed_siting.optimize.greedy import (
    existing_coverage_mask,
    greedy_select,
    leave_one_out_criticality,
)
from aed_siting.optimize.objectives import (
    compare_objectives,
    kcenter_gonzalez,
    kcenter_minimax,
    pmedian_greedy,
)
from aed_siting.optimize.oracle import brute_force_best
from aed_siting.simulate import outage_stress, sample_incidents, simulate

THRESHOLD = 126.0


@pytest.fixture(scope="module")
def toy():
    """A small instance where the optimum is checkable by hand.

    Four cells in a line, three candidates. Cell weights make the middle pair
    worth more than the ends, so a coverage objective and a worst-case objective
    genuinely disagree.
    """
    demand = np.array([1.0, 5.0, 5.0, 1.0])
    dist = np.array(
        [
            [10.0, 400.0, 900.0],
            [100.0, 120.0, 800.0],
            [400.0, 100.0, 300.0],
            [900.0, 500.0, 50.0],
        ]
    )
    existing = np.full((4, 1), 1000.0)
    return demand, dist, existing


# --- greedy ----------------------------------------------------------------


@pytest.mark.gate
def test_greedy_matches_bruteforce_k1(toy):
    demand, dist, existing = toy
    already = existing_coverage_mask(existing, THRESHOLD)
    result = greedy_select(demand, dist, 1, THRESHOLD, already)
    best, _ = brute_force_best(demand, dist, 1, THRESHOLD, already)
    assert result.order == best


@pytest.mark.gate
@pytest.mark.slow
def test_greedy_matches_bruteforce_k2(toy):
    demand, dist, existing = toy
    already = existing_coverage_mask(existing, THRESHOLD)
    result = greedy_select(demand, dist, 2, THRESHOLD, already)
    _, best_score = brute_force_best(demand, dist, 2, THRESHOLD, already)
    achieved = coverage(demand, dist, result.order, THRESHOLD)
    assert achieved == pytest.approx(best_score, abs=1e-9)


@pytest.mark.gate
def test_greedy_sequence_is_nested(toy):
    """A staged budget must never invalidate an earlier choice."""
    demand, dist, existing = toy
    already = existing_coverage_mask(existing, THRESHOLD)
    full = greedy_select(demand, dist, 3, THRESHOLD, already).order
    for k in range(1, 4):
        assert greedy_select(demand, dist, k, THRESHOLD, already).order == full[:k]


@pytest.mark.gate
def test_existing_network_is_preseeded(toy):
    """k=0 must equal the existing network's coverage, not zero."""
    demand, dist, _ = toy
    close = np.full((4, 1), 10.0)
    already = existing_coverage_mask(close, THRESHOLD)
    result = greedy_select(demand, dist, 0, THRESHOLD, already)
    assert result.baseline_coverage == pytest.approx(1.0)
    assert result.order == []


@pytest.mark.gate
def test_greedy_is_marginal_not_absolute():
    """Two overlapping sites must not both be chosen when a third adds new demand.

    This is the difference between the method and the demand-ranked baseline; if
    it ever collapses, the headline result is meaningless.
    """
    demand = np.array([10.0, 10.0, 1.0])
    dist = np.array([[1.0, 1.0, 900.0], [1.0, 1.0, 900.0], [900.0, 900.0, 1.0]])
    existing = np.full((3, 1), 5000.0)
    order = greedy_select(
        demand, dist, 2, THRESHOLD, existing_coverage_mask(existing, THRESHOLD)
    ).order
    assert 2 in order, "greedy re-covered covered ground instead of taking new demand"


@pytest.mark.gate
def test_tiebreak_is_lowest_index():
    """Identical candidates must resolve deterministically."""
    demand = np.array([1.0, 1.0])
    dist = np.array([[1.0, 1.0], [1.0, 1.0]])
    existing = np.full((2, 1), 5000.0)
    order = greedy_select(
        demand, dist, 1, THRESHOLD, existing_coverage_mask(existing, THRESHOLD)
    ).order
    assert order == [0]


def test_leave_one_out_criticality_finds_sole_cover():
    """A device that is the only cover for real demand must score above zero."""
    demand = np.array([4.0, 1.0])
    existing = np.array([[1.0, 900.0], [900.0, 1.0]])
    scores = leave_one_out_criticality(demand, existing, THRESHOLD)
    assert scores[0] > scores[1] > 0


# --- alternative objectives -------------------------------------------------


@pytest.mark.gate
def test_kcenter_reduces_max_distance(toy):
    """The equity objective must beat the coverage objective on worst case."""
    demand, dist, existing = toy
    already = existing_coverage_mask(existing, THRESHOLD)
    cov = greedy_select(demand, dist, 2, THRESHOLD, already).order
    table, _ = compare_objectives(demand, dist, existing, 2, THRESHOLD, cov)
    coverage_row = table[table["objective"].str.startswith("max coverage")].iloc[0]
    kcenter_row = table[table["objective"].str.contains("minimax")].iloc[0]
    assert kcenter_row["max_m"] <= coverage_row["max_m"], (
        "k-center did not improve worst-case distance"
    )


@pytest.mark.gate
def test_objectives_disagree_and_the_tradeoff_is_visible(toy):
    """Coverage should win on coverage; p-median should win on mean distance."""
    demand, dist, existing = toy
    cov = greedy_select(
        demand, dist, 2, THRESHOLD, existing_coverage_mask(existing, THRESHOLD)
    ).order
    table, agreed = compare_objectives(demand, dist, existing, 2, THRESHOLD, cov)
    nothing = table[table["objective"] == "do nothing"].iloc[0]

    # Each objective must improve the quantity it optimises. Asserting that each
    # also *wins* its column is too strong: on a small instance one selection can
    # dominate several metrics at once, which is a property of the instance, not
    # a defect in the objectives.
    coverage_row = table[table["objective"].str.startswith("max coverage")].iloc[0]
    kcenter_row = table[table["objective"].str.contains("minimax")].iloc[0]
    pmedian_row = table[table["objective"].str.startswith("min average")].iloc[0]

    assert coverage_row["coverage"] >= nothing["coverage"]
    assert kcenter_row["max_m"] <= nothing["max_m"]
    assert pmedian_row["mean_m"] <= nothing["mean_m"]
    assert isinstance(agreed, list)


def test_every_objective_respects_the_budget(toy):
    demand, dist, existing = toy
    for sites in (
        kcenter_minimax(dist, existing, 2),
        kcenter_gonzalez(dist, existing, 2),
        pmedian_greedy(demand, dist, existing, 2),
    ):
        assert len(sites) <= 2
        assert len(set(sites)) == len(sites), "a site was selected twice"


# --- simulation -------------------------------------------------------------


@pytest.mark.gate
def test_simulation_matches_analytic_quantiles(town_artifacts):
    """The demonstration layer must not drift from the numbers it illustrates.

    Sampling is for legibility only; the reported metric is the exact weighted
    quantile. If these diverge, the map is telling a different story from the
    evaluation report.
    """
    cells = town_artifacts["cells"]
    incidents = town_artifacts["incidents_before"]
    sampled_p90 = incidents["distance_m"].quantile(0.9)

    demand = cells["demand"].to_numpy()
    per_cell = (
        incidents.groupby("cell")["distance_m"].first().reindex(range(len(demand))).to_numpy()
    )
    analytic_p90 = weighted_quantile(per_cell, demand, 0.9)[0]
    assert sampled_p90 == pytest.approx(analytic_p90, rel=0.15), (
        f"sampled p90 {sampled_p90:.0f} m diverges from analytic {analytic_p90:.0f} m"
    )


@pytest.mark.gate
def test_simulation_is_seeded(town_artifacts):
    """The same seed must reproduce the same incidents."""
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    first = sample_incidents(grid, demand, n=200, seed=42)
    second = sample_incidents(grid, demand, n=200, seed=42)
    third = sample_incidents(grid, demand, n=200, seed=43)
    assert (first == second).all()
    assert not (first == third).all(), "different seeds produced identical draws"


@pytest.mark.gate
def test_incidents_are_labelled_synthetic(town_artifacts):
    """The brief requires scenarios to be scripted, historical or simulated."""
    incidents = town_artifacts["incidents_after"]
    assert "scenario" in incidents.columns
    assert set(incidents["scenario"].unique()) <= {"before", "after"}


@pytest.mark.gate
def test_outage_never_increases_coverage(town_artifacts):
    """Losing devices cannot help. Monotonicity sanity on the stress test."""
    stress = town_artifacts["outage_stress"]
    baseline = town_artifacts["summary"]["baseline_coverage"]
    assert (stress["coverage_mean"] <= baseline + 1e-9).all()
    assert stress["coverage_mean"].is_monotonic_decreasing
    assert (stress["coverage_p05"] <= stress["coverage_mean"]).all()


def test_outage_stress_handles_more_losses_than_devices():
    demand = np.array([1.0, 1.0])
    existing = np.full((2, 3), 10.0)
    result = outage_stress(demand, existing, THRESHOLD, outage_sizes=(1, 99), trials=5)
    assert list(result["devices_lost"]) == [1]


@pytest.mark.gate
def test_outage_stress_includes_planned_devices():
    """Planned devices must be able to fail, and must be able to help.

    Regression gate: the outage view called ``live_simulation(..., (), seed)`` and
    ``outage_stress`` accepted only ``dist_existing``, so the funded plan could not
    enter the model at all -- the budget slider moved every other view on the page
    and left this chart frozen. Two things have to hold once it is wired up: a
    planned device that covers new ground raises coverage under outage, and the
    failure draw treats it as fallible rather than permanent.
    """
    demand = np.array([1.0, 1.0])
    # Two existing devices both reach cell 0 only; cell 1 is reachable solely by
    # the planned device. Two existing devices are needed because an outage of a
    # whole one-device network is skipped as degenerate.
    existing = np.array([[10.0, 10.0], [900.0, 900.0]])
    planned = np.array([[900.0], [10.0]])

    alone = outage_stress(demand, existing, THRESHOLD, outage_sizes=(1,), trials=30, seed=1)
    with_plan = outage_stress(
        demand, existing, THRESHOLD, outage_sizes=(1,), trials=30, seed=1, dist_planned=planned
    )

    # The plan covers ground the existing network cannot, so it must help.
    assert with_plan["coverage_mean"].iloc[0] > alone["coverage_mean"].iloc[0]
    # Three devices in the network now, not two -- losing one is a third of it.
    assert with_plan["share_of_network"].iloc[0] == pytest.approx(1 / 3, abs=5e-4)
    # And the planned device is fallible: some draw must remove it, costing cover.
    assert with_plan["coverage_worst"].iloc[0] < 1.0


def test_outage_stress_without_planned_is_unchanged():
    """The new argument must not perturb the existing-only path."""
    demand = np.array([1.0, 1.0, 1.0])
    existing = np.array([[10.0, 900.0], [900.0, 10.0], [10.0, 10.0]])
    kwargs = {"outage_sizes": (1,), "trials": 10, "seed": 7}
    default = outage_stress(demand, existing, THRESHOLD, **kwargs)
    explicit_none = outage_stress(demand, existing, THRESHOLD, dist_planned=None, **kwargs)
    empty = outage_stress(demand, existing, THRESHOLD, dist_planned=np.empty((3, 0)), **kwargs)
    pd.testing.assert_frame_equal(default, explicit_none)
    pd.testing.assert_frame_equal(default, empty)


def test_simulate_marks_coverage_against_the_threshold(town_artifacts):
    grid, cells = town_artifacts["grid"], town_artifacts["cells"]
    demand = cells["demand"].to_numpy()
    nearest = np.where(cells["covered_before"].to_numpy(), 10.0, 900.0)
    result = simulate(grid, demand, nearest, THRESHOLD, n=300, seed=1)
    assert 0.0 <= result.covered_share <= 1.0
    assert result.incidents["covered"].equals(result.incidents["distance_m"] <= THRESHOLD)
