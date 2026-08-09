"""End-to-end pipeline.

Writes every artifact the UI and the evaluation report consume into
``data/processed/<town>/``. The UI reads only those files, so a deployed demo
never touches a network -- which is what makes the reproducibility claim true
rather than aspirational.

Run twice with the same seed and the outputs are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from aed_siting.baselines import coverage_band, demand_ranked_topk, do_nothing, random_k
from aed_siting.candidates import abstained_sites, build_candidates
from aed_siting.config import (
    DEFAULT_TOWN,
    RETRIEVAL_BUDGET_MIN,
    SEED,
    TOWNS,
    osm_path,
    processed_dir,
    threshold_m,
    town_area,
    town_label,
)
from aed_siting.demand import build_demand, proxy_agreement
from aed_siting.distance import DetourEuclidean
from aed_siting.eligibility import validate_against_existing
from aed_siting.grid import build_grid, grid_summary
from aed_siting.hours import always_open_mask, hours_report
from aed_siting.io.aed import load_aed
from aed_siting.io.boundary import load_subzones, load_town_boundary
from aed_siting.io.osm import buildings, load_osm
from aed_siting.metrics import catchment_load, evaluate, nearest, subzone_coverage
from aed_siting.naming import label_candidates, label_existing, label_quality, sites_reference
from aed_siting.optimize.greedy import (
    existing_coverage_mask,
    greedy_select,
    leave_one_out_criticality,
)
from aed_siting.optimize.ilp import nesting_loss
from aed_siting.optimize.objectives import compare_objectives
from aed_siting.optimize.policy import PRESETS, covered_at_step
from aed_siting.optimize.policy import build_queue as build_policy_queue
from aed_siting.quality import find_colocated_sites, validate_postal_codes
from aed_siting.queue import build_queue
from aed_siting.scenarios import (
    consensus_sites,
    cross_proxy_sensitivity,
    hours_scenario,
    stability_table,
    threshold_sweep,
)
from aed_siting.simulate import outage_stress, simulate
from aed_siting.voronoi import measure

PRIMARY_PROXY = "area"


def run(
    town: str = DEFAULT_TOWN,
    k: int = 30,
    seed: int = SEED,
    budget_min: float = RETRIEVAL_BUDGET_MIN,
) -> dict:
    """Execute the full pipeline for one town and write its artifacts.

    Args:
        town: Town slug from :data:`aed_siting.config.TOWNS`.
        k: Devices to place in the headline plan.
        seed: Random seed. Consumed only by the random-k baseline and the
            simulation layer; the reported metrics are all deterministic.
        budget_min: Round-trip retrieval budget for the headline threshold.

    Returns:
        A summary dict, also written to ``run_summary.json`` in the town's
        artifact directory.
    """
    out_dir = processed_dir(town)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_extra: dict[str, pd.DataFrame] = {}
    limit = threshold_m(budget_min)
    area = town_area(town)

    boundary = load_town_boundary(town=area)
    subzones = load_subzones(town=area)
    grid = build_grid(boundary)
    existing, load_report = load_aed(boundary=boundary)
    existing = label_existing(existing)
    osm = load_osm(osm_path(town), boundary=boundary)
    candidates, candidate_report = build_candidates(osm)
    candidates = label_candidates(candidates, town)
    demand, demand_report = build_demand(grid, osm)

    model = DetourEuclidean()
    dist = model.matrix(grid, candidates)
    dist_existing = model.matrix(grid, existing) + existing["floor_penalty_m"].to_numpy()[None, :]
    open_mask = always_open_mask(existing).to_numpy()

    weights = demand[PRIMARY_PROXY].to_numpy()
    already = existing_coverage_mask(dist_existing, limit)

    # --- policies -----------------------------------------------------------
    # Every preset is an ordering of the same criteria; the default reproduces
    # marginal-coverage greedy exactly (`demand` first).
    cell_subzone = _subzone_of_cell(grid, subzones)
    subzone_ids = _subzone_codes(grid, subzones)
    # Build every preset to *saturation* rather than to a fixed budget. The greedy
    # loop stops when no candidate adds coverage, so the queue length is a real
    # finding -- Jurong East exhausts at 49 devices because it runs out of
    # eligible buildings, while Bedok keeps improving to 234. Because the sequence
    # is nested, any smaller budget is just a prefix, so the UI can answer any k
    # up to saturation by slicing instead of re-solving.
    saturation_cap = dist.shape[1]
    policies = {
        name: build_policy_queue(
            keys, weights, dist, dist_existing, saturation_cap, limit, subzone_ids
        )
        for name, keys in PRESETS.items()
    }
    default_name = next(iter(PRESETS))
    greedy = policies[default_name]
    # `k` now only decides which prefix the headline numbers describe.
    k = min(k, len(greedy.order)) or 1
    combined = np.concatenate([dist, dist_existing], axis=1)
    existing_cols = list(range(dist.shape[1], combined.shape[1]))

    def score(sites: list[int]) -> dict:
        return evaluate(weights, combined, list(sites) + existing_cols, limit).as_row()

    comparison = pd.DataFrame(
        [
            {"method": "do nothing (existing network only)", "k": 0, **score(do_nothing())},
            {
                "method": "demand-ranked top-k (baseline)",
                "k": k,
                **score(demand_ranked_topk(weights, dist, k, limit)),
            },
            {"method": "marginal-coverage greedy (method)", "k": k, **score(greedy.top(k))},
        ]
    )
    random_band = coverage_band(
        [score(t)["coverage"] for t in random_k(len(candidates), k, seed=seed)]
    )

    # --- alternative objectives -------------------------------------------
    objectives, agreed_sites = compare_objectives(
        weights, dist, dist_existing, k, limit, greedy.top(k)
    )

    # --- scenarios ---------------------------------------------------------
    cross_proxy = cross_proxy_sensitivity(demand, dist, dist_existing, k, limit)
    sweep = threshold_sweep(weights, dist, dist_existing, k)
    hours = hours_scenario(weights, dist, dist_existing, dist_existing[:, open_mask], k, limit)
    selections = {
        f"proxy:{name}": _solve(demand[name].to_numpy(), dist, dist_existing, k, limit)
        for name in demand.columns
    }
    selections.update({f"threshold:{r.budget_min}min": list(r.sites) for r in sweep.itertuples()})
    selections.update({f"hours:{r.scenario}": list(r.sites) for r in hours.itertuples()})
    stability = stability_table(selections)
    # Carry the TP-xxx label so stability joins to the queue and the map. Keyed on
    # the integer index alone it could not be resolved to a place on the ground.
    label_by_index = dict(zip(candidates["site_index"], candidates["site_id"], strict=True))
    name_by_index = dict(zip(candidates["site_index"], candidates["label"], strict=True))
    stability = stability.rename(columns={"site_id": "site_index"})
    stability["site_id"] = stability["site_index"].map(label_by_index)
    stability["label"] = stability["site_index"].map(name_by_index)
    stability = stability[["site_id", "label", "site_index", "times_chosen", "share"]]

    # --- simulation ---------------------------------------------------------
    before_nearest = nearest(dist_existing, range(dist_existing.shape[1]))
    after_nearest = nearest(combined, list(greedy.top(k)) + existing_cols)
    sim_before = simulate(grid, weights, before_nearest, limit, seed=seed, label="before")
    sim_after = simulate(grid, weights, after_nearest, limit, seed=seed, label="after")
    outages = outage_stress(weights, dist_existing, limit, seed=seed)

    # --- voronoi diagnostics ------------------------------------------------
    # Service areas answer questions coverage % cannot: who is over-subscribed,
    # and where the single worst-served point is. Loads come from the existing
    # catchment computation rather than being recomputed here.
    load_before = catchment_load(weights, dist_existing, range(dist_existing.shape[1])).to_numpy()
    cells_before, vm_before = measure(existing, boundary, load_before)

    proposed = candidates.iloc[greedy.top(k)]
    combined_devices = gpd.GeoDataFrame(
        pd.concat(
            [
                existing[["label", "geometry"]],
                proposed[["label", "geometry"]],
            ],
            ignore_index=True,
        ),
        crs=existing.crs,
    )
    dist_combined = combined[:, existing_cols + list(greedy.top(k))]
    load_after = catchment_load(weights, dist_combined, range(dist_combined.shape[1])).to_numpy()
    cells_after, vm_after = measure(combined_devices, boundary, load_after)

    # --- equity, criticality, queue ---------------------------------------
    after_idx = list(greedy.top(k)) + existing_cols
    criticality = (
        pd.DataFrame(
            {
                "AED_ID": existing["AED_ID"].to_numpy(),
                "label": existing["label"].to_numpy(),
                "sole_cover_demand": leave_one_out_criticality(weights, dist_existing, limit).round(
                    6
                ),
            }
        )
        .sort_values("sole_cover_demand", ascending=False)
        .reset_index(drop=True)
    )
    queue = build_queue(
        candidates,
        greedy.order,
        greedy.marginal_gain,
        greedy.cumulative_coverage,
        weights,
        dist,
        dist_existing,
        limit,
    )
    queue["stable"] = (
        queue["site_index"].map(stability.set_index("site_index")["share"].to_dict()).fillna(0.0)
    )

    # One queue and one coverage timeline per preset, so the UI switches policy
    # without re-solving. `covered_at_step` is what makes the animation a replay
    # of the real result rather than a decoration.
    cell_steps = {}
    for name, result in policies.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        frame = build_queue(
            candidates,
            result.order,
            result.marginal_gain,
            result.cumulative_coverage,
            weights,
            dist,
            dist_existing,
            limit,
        )
        frame["deciding_key"] = result.deciding_key[: len(frame)]
        frame["stable"] = (
            frame["site_index"]
            .map(stability.set_index("site_index")["share"].to_dict())
            .fillna(0.0)
        )
        artifacts_extra[f"install_queue_{slug}.csv"] = frame
        cell_steps[f"step_{slug}"] = covered_at_step(result.order, dist, dist_existing, limit)

    # --- write -------------------------------------------------------------
    artifacts = {
        **artifacts_extra,
        "install_queue.csv": queue,
        "sites.csv": sites_reference(candidates),
        "label_quality.csv": label_quality(candidates).reset_index(),
        "method_vs_baseline.csv": comparison,
        "objectives.csv": objectives,
        "cross_proxy_sensitivity.csv": cross_proxy.reset_index(),
        "threshold_sweep.csv": sweep.drop(columns="sites"),
        "hours_scenario.csv": hours.drop(columns="sites"),
        "site_stability.csv": stability,
        "equity_before.csv": subzone_coverage(
            weights, dist_existing, range(dist_existing.shape[1]), cell_subzone, limit
        ).reset_index(),
        "equity_after.csv": subzone_coverage(
            weights, combined, after_idx, cell_subzone, limit
        ).reset_index(),
        "criticality.csv": criticality,
        "nesting_loss.csv": pd.DataFrame(
            nesting_loss(weights, dist, greedy.order, limit, [1, 3, 5, k], already)
        ),
        "proxy_agreement.csv": proxy_agreement(demand).reset_index(),
        "eligibility_validation.csv": validate_against_existing(
            buildings(osm), existing
        ).reset_index(),
        "demand.csv": demand.reset_index(),
        "incidents_before.csv": sim_before.incidents,
        "incidents_after.csv": sim_after.incidents,
        "outage_stress.csv": outages,
        "voronoi_metrics.csv": pd.DataFrame(
            [{"stage": "before", **vm_before.as_row()}, {"stage": "after", **vm_after.as_row()}]
        ),
    }
    for name, frame in artifacts.items():
        frame.to_csv(out_dir / name, index=False)

    _write_geo(candidates, out_dir / "candidates.geojson")
    _write_geo(existing, out_dir / "existing_aeds.geojson")
    _write_geo(grid.drop(columns="hex_geometry"), out_dir / "grid.geojson")

    # Subzone geometry with its own coverage, so equity can be drawn as a map
    # rather than read off a twelve-line slope chart.
    equity_before_frame = subzone_coverage(
        weights, dist_existing, range(dist_existing.shape[1]), cell_subzone, limit
    )
    equity_after_frame = subzone_coverage(weights, combined, after_idx, cell_subzone, limit)
    subzone_map = subzones[["subzone", "geometry"]].copy()
    subzone_map["coverage_before"] = (
        subzone_map["subzone"].map(equity_before_frame["coverage"]).fillna(0.0).round(4)
    )
    subzone_map["coverage_after"] = (
        subzone_map["subzone"].map(equity_after_frame["coverage"]).fillna(0.0).round(4)
    )
    subzone_map["delta_pp"] = (
        (subzone_map["coverage_after"] - subzone_map["coverage_before"]) * 100
    ).round(2)
    subzone_map["label"] = [
        f"{name} — {before:.0%} → {after:.0%}"
        for name, before, after in zip(
            subzone_map["subzone"],
            subzone_map["coverage_before"],
            subzone_map["coverage_after"],
            strict=True,
        )
    ]
    _write_geo(subzone_map, out_dir / "subzones.geojson")
    # Ship the distance matrices so the UI can solve an arbitrary priority
    # ordering live rather than the pipeline precomputing every permutation.
    # float32, not uint16: rounding to whole metres is far below the model's own
    # error, but it flips near-ties inside the policy's 2% tolerance band, so a
    # live solve diverged from the precomputed preset on 14 of 20 town/preset
    # pairs. Exactness between the two paths is worth the extra megabytes.
    np.savez_compressed(
        out_dir / "solver.npz",
        dist=dist.astype(np.float32),
        dist_existing=dist_existing.astype(np.float32),
        demand=weights.astype(np.float32),
        subzone=subzone_ids.astype(np.int16),
        threshold=np.float32(limit),
    )

    _write_geo(cells_before[~cells_before.geometry.is_empty], out_dir / "voronoi_before.geojson")
    _write_geo(cells_after[~cells_after.geometry.is_empty], out_dir / "voronoi_after.geojson")

    covered_after = (combined[:, after_idx] <= limit).any(axis=1)
    pd.DataFrame(
        {
            "h3": grid["h3"],
            "demand": weights.round(6),
            "covered_before": already,
            "covered_after": covered_after,
            **cell_steps,
        }
    ).to_csv(out_dir / "cells.csv", index=False)

    summary = {
        "town": town,
        "town_label": town_label(town),
        "planning_area": area,
        "seed": seed,
        "k": k,
        "budget_min": budget_min,
        "threshold_m": round(limit, 1),
        "primary_proxy": PRIMARY_PROXY,
        "grid": grid_summary(grid, boundary).to_dict(),
        "existing_aeds": len(existing),
        "candidates": len(candidates),
        "abstained_sites": len(abstained_sites(osm)),
        "candidate_report": asdict(candidate_report),
        "demand_report": {**asdict(demand_report), "notes": demand_report.notes},
        "load_report": asdict(load_report),
        "hours": hours_report(existing).summary(),
        "postal_validation": validate_postal_codes(existing).summary(),
        "colocation": find_colocated_sites(existing).summary(),
        "baseline_coverage": round(greedy.baseline_coverage, 4),
        "coverage_after": round(greedy.cumulative_coverage[-1], 4)
        if greedy.cumulative_coverage
        else None,
        "random_k_band": {key: round(value, 4) for key, value in random_band.items()},
        "consensus_sites": consensus_sites(selections),
        "objective_agreed_sites": agreed_sites,
        "saturation": {
            name: {
                "devices": len(result.order),
                # Distance-optimising orderings keep going after coverage stops
                # improving -- a device that adds no coverage can still shorten
                # the average trip. Both numbers are reported so a planner can
                # see where the coverage argument ends.
                "coverage_saturates_at": int(
                    sum(1 for g in result.marginal_gain if g > 0) or len(result.order)
                ),
                "coverage": round(
                    result.cumulative_coverage[-1]
                    if result.cumulative_coverage
                    else result.baseline_coverage,
                    4,
                ),
            }
            for name, result in policies.items()
        },
        "policies": {
            name: {
                "keys": list(result.keys),
                "coverage": round(
                    result.cumulative_coverage[-1]
                    if result.cumulative_coverage
                    else result.baseline_coverage,
                    4,
                ),
                "sites": result.order,
            }
            for name, result in policies.items()
        },
        "sites_placed": len(greedy.order),
        "simulation_before": sim_before.summary(),
        "simulation_after": sim_after.summary(),
        "voronoi_before": vm_before.summary(),
        "voronoi_after": vm_after.summary(),
        "worst_point_before": vm_before.largest_empty_circle_point,
        "worst_point_after": vm_after.largest_empty_circle_point,
        "worst_radius_before": round(vm_before.largest_empty_circle_m, 1),
        "worst_radius_after": round(vm_after.largest_empty_circle_m, 1),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _solve(weights: np.ndarray, dist, dist_existing, k: int, limit: float) -> list[int]:
    """Greedy selection under one demand surface."""
    return greedy_select(
        weights, dist, k, limit, existing_coverage_mask(dist_existing, limit)
    ).order


def _subzone_codes(grid: gpd.GeoDataFrame, subzones: gpd.GeoDataFrame) -> np.ndarray:
    """Integer subzone label per grid cell, for the ``equity`` policy key."""
    return _subzone_of_cell(grid, subzones).astype("category").cat.codes.to_numpy()


def _subzone_of_cell(grid: gpd.GeoDataFrame, subzones: gpd.GeoDataFrame) -> pd.Series:
    """Label each grid cell with the subzone containing its centroid."""
    joined = gpd.sjoin(
        grid[["geometry"]], subzones[["subzone", "geometry"]], predicate="within", how="left"
    )
    return joined["subzone"].fillna("(outside subzones)").reset_index(drop=True)


def _write_geo(frame: gpd.GeoDataFrame, path: Path) -> None:
    """Write a GeoDataFrame, dropping extra geometry columns GeoJSON cannot hold."""
    out = frame.copy()
    for column in list(out.columns):
        if column != out.geometry.name and out[column].dtype == "geometry":
            out = out.drop(columns=column)
    out.to_file(path, driver="GeoJSON")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AED siting pipeline (planning only).")
    parser.add_argument("--town", default=DEFAULT_TOWN, choices=sorted(TOWNS))
    parser.add_argument("--all-towns", action="store_true", help="build every town in one pass")
    parser.add_argument("--k", type=int, default=30, help="devices to place")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budget-min", type=float, default=RETRIEVAL_BUDGET_MIN)
    args = parser.parse_args(argv)

    towns = sorted(TOWNS) if args.all_towns else [args.town]
    summaries = {}
    for slug in towns:
        summary = run(town=slug, k=args.k, seed=args.seed, budget_min=args.budget_min)
        summaries[slug] = summary
        print(
            f"[{slug:12s}] {summary['existing_aeds']:4d} devices · "
            f"{summary['candidates']:4d} candidates · "
            f"{summary['baseline_coverage']:.1%} -> {summary['coverage_after']:.1%}"
        )
    if args.all_towns:
        (processed_dir(DEFAULT_TOWN).parent / "towns.json").write_text(
            json.dumps({slug: {"label": town_label(slug)} for slug in towns}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
