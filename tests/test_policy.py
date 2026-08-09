"""Gates on priority ordering.

The policy is a *list of sort keys*, so these tests are mostly about the ordering
actually mattering — a lexicographic rank over continuous scores that never
reaches its second key is indistinguishable from a single-key sort, and would
silently make every preset identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from aed_siting.optimize.policy import (
    KEYS,
    PRESETS,
    build_queue,
    covered_at_step,
    rank_candidates,
)

THRESHOLD = 126.0


@pytest.fixture(scope="module")
def instance():
    """An instance where coverage and worst-case genuinely disagree.

    Cells 0-2 are a dense cluster the existing network already serves. Cell 4 is
    a busy uncovered block; cell 3 is a remote outlier with almost no demand and
    is what sets the worst-case distance.

    Candidate A serves the busy block — best on ``demand``, useless on ``gap``,
    because the outlier still sets the maximum. Candidate B serves the outlier —
    best on ``gap``, nearly worthless on ``demand``. Any instance where both
    candidates leave the same worst cell unserved cannot discriminate at all,
    which is exactly the mistake this fixture originally made.
    """
    demand = np.array([10.0, 10.0, 10.0, 0.5, 8.0])
    dist = np.array(
        [
            [900.0, 900.0],
            [900.0, 900.0],
            [900.0, 900.0],
            [900.0, 40.0],  # only B reaches the outlier
            [50.0, 900.0],  # only A reaches the busy block
        ]
    )
    existing = np.array([[100.0], [100.0], [100.0], [700.0], [500.0]])
    return demand, dist, existing


@pytest.mark.gate
def test_ordering_changes_the_plan(instance):
    """Reversing the key order must change which site is chosen first.

    If it does not, banding is broken and every preset collapses to one policy.
    """
    demand, dist, existing = instance
    by_demand = build_queue(("demand", "gap"), demand, dist, existing, 1, THRESHOLD)
    by_gap = build_queue(("gap", "demand"), demand, dist, existing, 1, THRESHOLD)
    assert by_demand.order != by_gap.order, "key order had no effect on the plan"
    assert by_demand.order == [0], "demand-first should take the busy uncovered block"
    assert by_gap.order == [1], "gap-first should take the remote outlier"


@pytest.mark.gate
def test_no_policy_degenerates_into_index_order(instance, town_artifacts):
    """A saturated key must hand over to the next one, never to the index.

    Regression gate for a real defect: the old k-center returned
    ``[184, 0, 1, 2, 3, 4, 5, 6, 7, 8]`` because once the worst cell became
    unreachable every candidate scored identically and ``argmin`` walked the
    index, presenting nine arbitrary buildings as an equity plan.
    """
    for preset in PRESETS:
        slug = preset.lower().replace(" ", "_").replace("-", "_")
        queue = town_artifacts.get(f"install_queue_{slug}")
        if queue is None or "deciding_key" not in queue.columns:
            continue
        # Check the failure directly rather than via a proxy. Counting runs of
        # consecutive indices also flags legitimate sequences -- candidates are
        # sorted by footprint, so neighbouring buildings can genuinely be the best
        # successive picks. `deciding_key == "index"` means no criterion separated
        # the candidates at all, which is the actual defect.
        arbitrary = queue[queue["deciding_key"] == "index"]
        assert arbitrary.empty, (
            f"{preset}: {len(arbitrary)} pick(s) decided by candidate index alone"
        )


@pytest.mark.gate
def test_banding_reaches_the_second_key():
    """Near-ties on key 1 must be separated by key 2, not by index."""
    available = np.ones(3, dtype=bool)
    scores = {
        # 0 and 1 are within the 2% band; 2 is clearly worse.
        "gap": np.array([100.0, 99.5, 10.0]),
        "demand": np.array([0.1, 0.9, 0.9]),
    }
    choice, deciding = rank_candidates(("gap", "demand"), scores, available)
    assert choice == 1, "the second key did not break a near-tie on the first"
    assert deciding == "demand"


@pytest.mark.gate
def test_saturated_first_key_falls_through():
    """When every candidate ties on key 1, key 2 decides — not the index."""
    available = np.ones(4, dtype=bool)
    scores = {
        "gap": np.zeros(4),  # fully saturated, as in the real failure
        "demand": np.array([0.1, 0.4, 0.9, 0.2]),
    }
    choice, deciding = rank_candidates(("gap", "demand"), scores, available)
    assert choice == 2 and deciding == "demand"


@pytest.mark.gate
def test_queue_is_nested(instance):
    """A staged budget must never invalidate an earlier pick, under any ordering."""
    demand, dist, existing = instance
    for keys in (("demand", "gap"), ("gap", "demand"), ("mean", "demand")):
        full = build_queue(keys, demand, dist, existing, 2, THRESHOLD).order
        assert build_queue(keys, demand, dist, existing, 1, THRESHOLD).order == full[:1]


@pytest.mark.gate
def test_policies_are_deterministic(instance):
    """Same inputs, same plan — byte-reproducibility depends on it."""
    demand, dist, existing = instance
    for keys in (("demand", "gap"), ("gap", "demand")):
        first = build_queue(keys, demand, dist, existing, 2, THRESHOLD)
        second = build_queue(keys, demand, dist, existing, 2, THRESHOLD)
        assert first.order == second.order
        assert first.deciding_key == second.deciding_key


@pytest.mark.gate
def test_every_preset_produces_a_queue(town_artifacts):
    """Each preset offered in the UI must have a shipped artifact."""
    for preset in PRESETS:
        slug = preset.lower().replace(" ", "_").replace("-", "_")
        queue = town_artifacts.get(f"install_queue_{slug}")
        assert queue is not None, f"missing install_queue_{slug}.csv"
        assert len(queue) > 0
        assert queue["cumulative_coverage"].is_monotonic_increasing


@pytest.mark.gate
def test_covered_at_step_is_monotone(town_artifacts):
    """A cell covered at step k must stay covered at every later step."""
    cells = town_artifacts["cells"]
    for column in [c for c in cells.columns if c.startswith("step_")]:
        steps = cells[column].to_numpy()
        assert steps.min() >= -1
        for k in range(0, int(steps.max()) + 1):
            now = (steps >= 0) & (steps <= k)
            later = (steps >= 0) & (steps <= k + 1)
            assert (now <= later).all(), f"{column}: coverage went backwards at step {k}"


@pytest.mark.gate
def test_covered_at_step_agrees_with_the_queue(town_artifacts):
    """Step 0 must equal the existing network's coverage."""
    cells = town_artifacts["cells"]
    for column in [c for c in cells.columns if c.startswith("step_")]:
        assert ((cells[column] == 0) == cells["covered_before"]).all(), (
            f"{column}: step 0 disagrees with covered_before"
        )


@pytest.mark.gate
def test_new_ground_share_bounds(town_artifacts):
    """Every proposal's new-ground share must be a valid fraction.

    Low values would mean the tool is recommending devices that mostly duplicate
    the existing network — a real warning worth surfacing, not hiding.
    """
    queue = town_artifacts["install_queue"]
    if "new_ground_share" not in queue.columns:
        pytest.skip("queue predates new_ground_share")
    share = queue["new_ground_share"]
    assert share.between(0.0, 1.0).all()
    assert share.min() > 0.2, f"a proposal is only {share.min():.0%} new ground"


@pytest.mark.gate
def test_live_solve_matches_the_precomputed_preset():
    """The UI's live solver and the pipeline must not drift apart.

    The UI solves arbitrary orderings from shipped matrices while the pipeline
    precomputes the presets; if the two disagree, switching to "Custom" with a
    preset's own ordering would silently produce a different plan.

    This gate is why the matrices ship as float32. As uint16 metres they were
    13.5 MB instead of 27, but rounding flipped near-ties inside the 2% band and
    14 of 20 town/preset pairs diverged.
    """
    import numpy as np
    import pandas as pd

    from aed_siting.config import PROCESSED_DIR, TOWNS

    checked = 0
    for slug in TOWNS:
        directory = PROCESSED_DIR / slug
        if not (directory / "solver.npz").exists():
            continue
        with np.load(directory / "solver.npz") as payload:
            dist = payload["dist"].astype(float)
            dist_existing = payload["dist_existing"].astype(float)
            weights = payload["demand"].astype(float)
            threshold = float(payload["threshold"])
            subzones = payload["subzone"].astype(int)
        for name, keys in PRESETS.items():
            slug_name = name.lower().replace(" ", "_").replace("-", "_")
            path = directory / f"install_queue_{slug_name}.csv"
            if not path.exists():
                continue
            expected = pd.read_csv(path)["site_index"].tolist()
            live = build_queue(
                keys, weights, dist, dist_existing, len(expected), threshold, subzones
            ).order
            assert live == expected, f"{slug}/{name}: live solve diverged from precomputed"
            checked += 1
    if checked == 0:
        pytest.skip("no solver artifacts present")


@pytest.mark.gate
def test_stability_joins_to_the_queue(town_artifacts):
    """Every stability row must resolve to a labelled site.

    Regression gate: stability was keyed on the integer candidate index while the
    queue showed `TP-084`, so a reader could not connect a robustness score to a
    place on the ground.
    """
    stability = town_artifacts.get("site_stability")
    candidates = town_artifacts.get("candidates")
    if stability is None or candidates is None:
        pytest.skip("artifacts absent")
    for column in ("site_id", "label", "site_index"):
        assert column in stability.columns, f"stability lacks {column}"
    assert stability["site_id"].astype(str).str.match(r"^[A-Z]{2,3}-\d+$").all()
    assert set(stability["site_id"]) <= set(candidates["site_id"])


@pytest.mark.gate
def test_subzone_artifact_has_before_and_after(town_artifacts):
    """Equity is drawn as a map, so the geometry must carry both states."""
    subzones = town_artifacts.get("subzones")
    if subzones is None:
        pytest.skip("subzone artifact absent")
    for column in ("subzone", "coverage_before", "coverage_after", "delta_pp", "label"):
        assert column in subzones.columns
    assert subzones["coverage_before"].between(0, 1).all()
    assert subzones["coverage_after"].between(0, 1).all()
    assert (subzones["coverage_after"] >= subzones["coverage_before"] - 1e-9).all(), (
        "a subzone lost coverage after adding devices"
    )


@pytest.mark.gate
def test_pinned_site_is_taken_first(instance):
    """A pinned site must appear, and appear at the front.

    Overrides are constraints on the solve, not edits to a finished plan, so the
    rest of the queue is re-optimised given the pin rather than left optimal for
    a plan that no longer exists.
    """
    demand, dist, existing = instance
    free = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD)
    pinned = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD, pinned=(1,))
    assert pinned.order[0] == 1
    assert pinned.deciding_key[0] == "pinned"
    assert len(set(pinned.order)) == len(pinned.order), "a pin was also chosen greedily"
    assert free.order != pinned.order or free.order[0] == 1


@pytest.mark.gate
def test_vetoed_site_never_appears(instance):
    demand, dist, existing = instance
    free = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD)
    banned = free.order[0]
    vetoed = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD, vetoed=(banned,))
    assert banned not in vetoed.order


@pytest.mark.gate
def test_override_cost_is_reported_not_assumed(instance):
    """An override's effect must be measurable, in either direction.

    It is tempting to assert a constraint can only hurt, but greedy is a
    heuristic, not an exact method: on real towns a forced first pick sometimes
    leads it down a *better* path (measured: -0.07 pp in Ang Mo Kio, -0.05 pp in
    Bedok). The UI therefore reports the delta rather than claiming a bound the
    algorithm does not provide.
    """
    demand, dist, existing = instance
    free = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD)
    pinned = build_queue(("demand", "gap"), demand, dist, existing, 2, THRESHOLD, pinned=(1,))
    assert free.cumulative_coverage and pinned.cumulative_coverage
    delta = free.cumulative_coverage[-1] - pinned.cumulative_coverage[-1]
    assert np.isfinite(delta)


@pytest.mark.gate
def test_incidents_follow_the_budget(town_artifacts):
    """Simulated incidents must recompute with the device count.

    Regression gate: the shipped incident CSVs are baked at the pipeline's own
    budget and carry no budget dimension, so nothing in the UI could make them
    react. They are now derived live.
    """
    import geopandas as gpd
    import pandas as pd

    from aed_siting.config import PROCESSED_DIR
    from aed_siting.simulate import simulate

    directory = PROCESSED_DIR / town_artifacts["summary"]["town"]
    if not (directory / "solver.npz").exists():
        pytest.skip("solver artifacts absent")
    with np.load(directory / "solver.npz") as payload:
        dist = payload["dist"].astype(float)
        dist_existing = payload["dist_existing"].astype(float)
        weights = payload["demand"].astype(float)
        threshold = float(payload["threshold"])
    grid = gpd.read_file(directory / "grid.geojson")
    queue = pd.read_csv(directory / "install_queue_most_people_reached.csv")

    before = dist_existing.min(axis=1)
    shares = []
    for k in (0, 10, 40):
        sites = queue.head(k)["site_index"].tolist()
        nearest = np.minimum(before, dist[:, sites].min(axis=1)) if sites else before
        shares.append(simulate(grid, weights, nearest, threshold, n=400, seed=42).covered_share)
    assert shares[0] < shares[1] < shares[2], f"incidents did not track the budget: {shares}"


@pytest.mark.gate
def test_same_seed_reproduces_the_same_incidents(town_artifacts):
    import geopandas as gpd

    from aed_siting.config import PROCESSED_DIR
    from aed_siting.simulate import simulate

    directory = PROCESSED_DIR / town_artifacts["summary"]["town"]
    if not (directory / "solver.npz").exists():
        pytest.skip("solver artifacts absent")
    with np.load(directory / "solver.npz") as payload:
        nearest = payload["dist_existing"].astype(float).min(axis=1)
        weights = payload["demand"].astype(float)
        threshold = float(payload["threshold"])
    grid = gpd.read_file(directory / "grid.geojson")
    first = simulate(grid, weights, nearest, threshold, n=300, seed=7).incidents
    again = simulate(grid, weights, nearest, threshold, n=300, seed=7).incidents
    other = simulate(grid, weights, nearest, threshold, n=300, seed=8).incidents
    assert first.equals(again)
    assert not first.equals(other)


def test_unknown_key_is_rejected(instance):
    demand, dist, existing = instance
    with pytest.raises(ValueError, match="unknown policy key"):
        build_queue(("nonsense",), demand, dist, existing, 1, THRESHOLD)


def test_all_declared_keys_are_scorable(instance):
    """Every key in KEYS must actually compute, including equity."""
    demand, dist, existing = instance
    subzones = np.array([0, 0, 1, 1, 1])
    for key in KEYS:
        result = build_queue((key, "demand"), demand, dist, existing, 1, THRESHOLD, subzones)
        assert len(result.order) <= 1


def test_covered_at_step_marks_unreachable_cells(instance):
    demand, dist, existing = instance
    order = build_queue(("demand",), demand, dist, existing, 1, THRESHOLD).order
    steps = covered_at_step(order, dist, existing, THRESHOLD)
    assert (steps == -1).any(), "a cell no device reaches should stay at -1"
