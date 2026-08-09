"""Synthetic incident simulation.

The *reported* metrics stay the exact analytic weighted quantiles in
:mod:`aed_siting.metrics` -- sampling would only add noise to a quantity
available in closed form. This module exists for two other reasons:

1. **Legibility.** "Here are 1,000 simulated emergencies, and this is how far each
   one was from a device" is understandable without a statistics background;
   "the demand-weighted p90 is 396 m" is not.
2. **Stress testing.** Some questions have no closed form -- what happens if
   several devices are out of service at once, and how bad is a bad draw?

Every incident is synthetic and labelled as such, satisfying the brief's
requirement that scenarios be scripted, historical, or explicitly simulated.
A gate asserts the sampled distribution converges on the analytic one, so the
demonstration layer can never drift away from the numbers it illustrates.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from aed_siting.config import CRS_WGS84, SEED, WALK_SPEED_MS

DEFAULT_INCIDENTS = 1000


@dataclass(frozen=True)
class SimulationResult:
    """Sampled incidents and their retrieval distances."""

    incidents: pd.DataFrame
    covered_share: float
    median_m: float
    p90_m: float

    def summary(self) -> str:
        """One-line summary for the UI and the tracker."""
        return (
            f"{len(self.incidents)} synthetic incidents: {self.covered_share:.1%} within "
            f"the threshold, median {self.median_m:.0f} m, p90 {self.p90_m:.0f} m."
        )


def sample_incidents(
    grid: gpd.GeoDataFrame,
    demand: np.ndarray,
    n: int = DEFAULT_INCIDENTS,
    seed: int = SEED,
) -> np.ndarray:
    """Draw cell indices for ``n`` synthetic incidents, proportional to demand.

    Assumptions:
        Incident risk is proportional to the demand proxy. This is a *modelling
        choice for illustration*, not an epidemiological claim -- the brief
        forbids inferring individual medical risk, and nothing here does.

    Limits:
        Incidents land at cell centroids, not real addresses. A cell is ~0.017 km²,
        so the placement error is on the order of the cell itself.
    """
    weights = np.asarray(demand, dtype=float)
    total = weights.sum()
    probabilities = (
        weights / total if total > 0 else np.full(len(weights), 1.0 / max(len(weights), 1))
    )
    rng = np.random.default_rng(seed)
    return rng.choice(len(grid), size=n, replace=True, p=probabilities)


def simulate(
    grid: gpd.GeoDataFrame,
    demand: np.ndarray,
    nearest_m: np.ndarray,
    threshold: float,
    n: int = DEFAULT_INCIDENTS,
    seed: int = SEED,
    label: str = "after",
) -> SimulationResult:
    """Sample incidents and record the retrieval distance each one faces.

    Args:
        grid: H3 grid, EPSG:3414.
        demand: Per-cell weight used as the sampling distribution.
        nearest_m: One-way distance from each cell to its nearest device.
        threshold: One-way metres.
        n: Number of incidents.
        seed: Sampling seed; the same seed always yields the same draw.
        label: Scenario tag carried into the output frame.

    Returns:
        A :class:`SimulationResult` whose ``incidents`` frame carries ``lat``,
        ``lon``, ``distance_m``, ``seconds`` (round-trip) and ``covered``.
    """
    picks = sample_incidents(grid, demand, n=n, seed=seed)
    wgs84 = grid.to_crs(CRS_WGS84)
    distances = np.asarray(nearest_m, dtype=float)[picks]
    finite = np.where(np.isfinite(distances), distances, np.nan)

    incidents = pd.DataFrame(
        {
            "cell": picks,
            "lat": wgs84.geometry.y.to_numpy()[picks].round(6),
            "lon": wgs84.geometry.x.to_numpy()[picks].round(6),
            "distance_m": np.round(finite, 1),
            "seconds": np.round(2.0 * finite / WALK_SPEED_MS, 1),
            "covered": distances <= threshold,
            "scenario": label,
        }
    )
    usable = incidents["distance_m"].dropna()
    return SimulationResult(
        incidents=incidents,
        covered_share=float(incidents["covered"].mean()),
        median_m=float(usable.median()) if len(usable) else float("inf"),
        p90_m=float(usable.quantile(0.9)) if len(usable) else float("inf"),
    )


def outage_stress(
    demand: np.ndarray,
    dist_existing: np.ndarray,
    threshold: float,
    outage_sizes: tuple[int, ...] = (1, 5, 10, 25),
    trials: int = 100,
    seed: int = SEED,
    dist_planned: np.ndarray | None = None,
) -> pd.DataFrame:
    """Coverage when randomly chosen devices in the network are unavailable.

    There is no closed form for "how bad is an unlucky combination of outages",
    so this is one of the few places sampling genuinely earns its place.

    Args:
        demand: Per-cell demand weights.
        dist_existing: One-way metres from each cell to each existing device.
        threshold: One-way reach in metres.
        outage_sizes: Numbers of simultaneously failed devices to test.
        trials: Random draws per outage size.
        seed: Seeds the failure draw; same seed reproduces the same trials.
        dist_planned: Distances to the devices this plan would fund. When given,
            failures are drawn from the **existing plus planned** network, so a
            device we recommend is as liable to be out as one already installed.
            Omitting it stresses the existing network alone.

    Returns:
        Mean and 5th-percentile coverage per outage size. The p05 column is the
        bad-draw case, which is the one worth planning against.

    Limits:
        Outages are drawn uniformly. Real failures correlate -- a power cut takes
        out a whole block at once -- so this understates clustered risk.

        ``share_of_network`` is a share of whichever network was passed in, so a
        run including planned devices is not directly comparable with one that
        does not: the same absolute loss is a smaller fraction of a bigger fleet.
    """
    weights = np.asarray(demand, dtype=float)
    total = float(weights.sum()) or 1.0
    # Planned devices are appended as extra columns, so a failure draw treats
    # them exactly like existing ones. Assuming a device we recommend can never
    # go out would flatter the plan precisely where the chart is meant to test it.
    network = (
        dist_existing
        if dist_planned is None or not dist_planned.size
        else np.hstack([dist_existing, dist_planned])
    )
    n_devices = network.shape[1]
    rng = np.random.default_rng(seed)

    rows = []
    for size in outage_sizes:
        if size >= n_devices:
            continue
        coverages = []
        for _ in range(trials):
            keep = rng.permutation(n_devices)[: n_devices - size]
            covered = (network[:, keep] <= threshold).any(axis=1)
            coverages.append(float(weights[covered].sum() / total))
        values = np.asarray(coverages)
        rows.append(
            {
                "devices_lost": size,
                "share_of_network": round(size / n_devices, 3),
                "coverage_mean": round(float(values.mean()), 4),
                "coverage_p05": round(float(np.percentile(values, 5)), 4),
                "coverage_worst": round(float(values.min()), 4),
            }
        )
    return pd.DataFrame(rows)
