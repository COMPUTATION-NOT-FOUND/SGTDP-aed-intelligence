"""AED siting planner.

Reads only pre-computed artifacts from ``data/processed/<town>/`` — there is no
runtime network path, which keeps a deployed demo reproducible and inside the
challenge brief's supplemental-data rules.

Five screens: **Start here** (what an AED is, why not one everywhere),
**Existing AEDs** (devices grouped by building), **Plan new AEDs** (what to buy),
**Stress-test** (does it hold up) and **Evidence** (why believe it). The safety
banner renders on every one.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# `aed_siting` lives under src/, so it is importable only via an install --
# `requirements.txt` does that with `-e .`. Put this checkout's src/ at the FRONT
# of the path regardless, so the package always matches the app.py beside it.
#
# Preferring src/ unconditionally is the point, not belt-and-braces. An earlier
# version only fell back when the import *failed*, which left a hosted runner
# holding a stale site-packages copy from a previous build: it imported fine, so
# the guard did nothing, and a freshly-pulled app.py called an old signature --
# `demand_hexes()` got four arguments and the deployed app died with a TypeError
# that never reproduced locally. Version skew, not a missing package.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aed_siting import viz
from aed_siting.baselines import demand_ranked_topk
from aed_siting.config import DEFAULT_TOWN, PROCESSED_DIR, SEED, TOWNS, town_label
from aed_siting.hours import classify_frame
from aed_siting.optimize.policy import KEY_LABELS, KEYS, PRESETS, covered_at_step
from aed_siting.optimize.policy import build_queue as solve_policy
from aed_siting.queue import build_queue as build_queue_frame
from aed_siting.safety import BANNER, DISCLOSURES, OFFICIAL_CHANNELS, safe_failure
from aed_siting.simulate import outage_stress, simulate

st.set_page_config(page_title="AED Siting — Singapore", layout="wide")


def banner() -> None:
    """Render the mandated safety banner. Every screen calls this."""
    st.error(BANNER)


def slug_of(preset: str) -> str:
    """Artifact suffix for a preset name."""
    return preset.lower().replace(" ", "_").replace("-", "_")


@st.cache_data(show_spinner=False)
def load_town(slug: str) -> dict:
    """Load one town's artifacts. Empty dict if the pipeline has not been run."""
    directory = PROCESSED_DIR / slug
    if not (directory / "run_summary.json").exists():
        return {}
    out: dict = {"summary": json.loads((directory / "run_summary.json").read_text())}
    for path in directory.glob("*.csv"):
        out[path.stem] = pd.read_csv(path)
    for path in directory.glob("*.geojson"):
        out[path.stem] = gpd.read_file(path)
    return out


@st.cache_data(show_spinner=False)
def load_solver(slug: str) -> dict | None:
    """Distance matrices for solving an arbitrary priority ordering in the UI.

    Shipped as uint16 metres so any ordering can be solved live in milliseconds,
    instead of the pipeline precomputing every permutation of five keys.
    """
    path = PROCESSED_DIR / slug / "solver.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def unreachable_cells(slug: str):
    """Cells no candidate site can reach, whatever the budget or the ordering.

    These are the gaps that are not a funding decision at all: no eligible
    building lies within one device's reach, so no plan at any budget closes
    them. Drawing them the same amber as fundable gaps is what made the map read
    as broken -- a third of Toa Payoh's remaining gap is this, not backlog.
    """
    payload = load_solver(slug)
    if payload is None:
        return None
    dist = payload["dist"].astype(float)
    return ~(dist <= float(payload["threshold"])).any(axis=1)


#: Methods that produce a genuinely *ordered* queue. The ILP yields an optimal
#: set per budget with no internal order, random-k is a distribution and
#: do-nothing is empty, so those three are reported as comparisons instead of
#: being given a meaningless `rank` column.
ORDERABLE = {
    "Marginal-coverage greedy": "policy",
    "Demand-ranked top-k (baseline)": "baseline",
    "Greedy k-center (equity)": "kcenter",
}
COMPARISON_ONLY = ("MCLP via ILP", "Random-k over 50 seeds", "Do nothing")


@st.cache_data(show_spinner="Solving…")
def solve_method(
    slug: str,
    method: str,
    keys: tuple[str, ...],
    k: int,
    pinned: tuple[str, ...] = (),
    vetoed: tuple[str, ...] = (),
):
    """Build a queue under one method, live from the shipped matrices."""
    payload = load_solver(slug)
    art = load_town(slug)
    candidates = art.get("candidates")
    if payload is None or candidates is None:
        return None
    dist = payload["dist"].astype(float)
    dist_existing = payload["dist_existing"].astype(float)
    demand = payload["demand"].astype(float)
    threshold = float(payload["threshold"])

    index_of = dict(zip(candidates["site_id"], candidates["site_index"], strict=True))
    pins = tuple(index_of[s] for s in pinned if s in index_of)
    vetos = tuple(index_of[s] for s in vetoed if s in index_of)

    if ORDERABLE[method] == "baseline":
        order = [
            s for s in demand_ranked_topk(demand, dist, k + len(vetos), threshold) if s not in vetos
        ]
        order = list(pins) + [s for s in order if s not in pins]
        order = order[:k]
        result = solve_policy(("demand",), demand, dist, dist_existing, 0, threshold)
        result.order = order
        covered = (dist_existing <= threshold).any(axis=1)
        total = demand.sum() or 1.0
        gains, cumulative = [], []
        for site in order:
            new = (dist[:, site] <= threshold) & ~covered
            gains.append(float(demand[new].sum() / total))
            covered = covered | (dist[:, site] <= threshold)
            cumulative.append(float(demand[covered].sum() / total))
        # The baseline has no stopping rule of its own -- it simply ranks every
        # candidate -- so trim it where coverage stops improving, matching how the
        # other methods saturate.
        useful = max((i + 1 for i, g in enumerate(gains) if g > 0), default=1)
        result.order = order[:useful]
        result.marginal_gain, result.cumulative_coverage = gains[:useful], cumulative[:useful]
        result.deciding_key = ["standalone demand"] * useful
    else:
        ordering = ("gap", "demand") if ORDERABLE[method] == "kcenter" else keys
        result = solve_policy(
            ordering,
            demand,
            dist,
            dist_existing,
            k,
            threshold,
            payload["subzone"].astype(int),
            pinned=pins,
            vetoed=vetos,
        )

    frame = build_queue_frame(
        candidates,
        result.order,
        result.marginal_gain,
        result.cumulative_coverage,
        demand,
        dist,
        dist_existing,
        threshold,
    )
    frame["deciding_key"] = result.deciding_key[: len(frame)]
    frame["origin"] = [
        "pinned" if sid in pinned else ("algorithm" if sid not in vetoed else "vetoed")
        for sid in frame["site_id"]
    ]
    stability = art.get("site_stability")
    if stability is not None and "site_index" in stability.columns:
        frame["stable"] = (
            frame["site_index"]
            .map(stability.set_index("site_index")["share"].to_dict())
            .fillna(0.0)
        )
    return frame, covered_at_step(result.order, dist, dist_existing, threshold)


@st.cache_data(show_spinner=False)
def live_simulation(slug: str, sites: tuple[int, ...], seed: int, n: int = 1000):
    """Incidents and outage stress for the *current* budget, computed on demand.

    The shipped `incidents_*.csv` are baked at the pipeline's own budget and have
    no budget dimension, so nothing in the UI could make them react to the device
    count. Deriving them here from `solver.npz` is what makes every dependent view
    — histogram, incident map, counters — follow the slider.
    """
    payload = load_solver(slug)
    art = load_town(slug)
    grid = art.get("grid")
    if payload is None or grid is None:
        return None
    dist = payload["dist"].astype(float)
    dist_existing = payload["dist_existing"].astype(float)
    demand = payload["demand"].astype(float)
    threshold = float(payload["threshold"])

    before_nearest = dist_existing.min(axis=1)
    after_nearest = (
        np.minimum(before_nearest, dist[:, list(sites)].min(axis=1)) if sites else before_nearest
    )
    # Outages are drawn from the network the planner would actually own, so the
    # devices this plan funds can fail too. The reference coverage has to move
    # with it: scoring an existing+planned network against the existing-only
    # baseline would report the plan's own gain as resilience.
    planned = dist[:, list(sites)] if sites else None
    total = float(demand.sum()) or 1.0
    return {
        "before": simulate(grid, demand, before_nearest, threshold, n=n, seed=seed, label="before"),
        "after": simulate(grid, demand, after_nearest, threshold, n=n, seed=seed, label="after"),
        "outages": outage_stress(demand, dist_existing, threshold, seed=seed, dist_planned=planned),
        "outage_baseline": float(demand[after_nearest <= threshold].sum() / total),
        "n_network": int(dist_existing.shape[1] + (len(sites) if sites else 0)),
        "n_existing": int(dist_existing.shape[1]),
    }


def available_towns() -> list[str]:
    """Towns with artifacts on disk."""
    return [slug for slug in TOWNS if (PROCESSED_DIR / slug / "run_summary.json").exists()]


def queue_for(art: dict, preset: str) -> pd.DataFrame | None:
    """The install queue produced by one priority ordering."""
    if art.get("_custom_queue") is not None:
        return art["_custom_queue"]
    return art.get(f"install_queue_{slug_of(preset)}", art.get("install_queue"))


def steps_for(art: dict, preset: str) -> np.ndarray | None:
    """Per-cell step at which each cell first becomes covered under a preset."""
    if art.get("_custom_steps") is not None:
        return art["_custom_steps"]
    cells = art.get("cells")
    column = f"step_{slug_of(preset)}"
    if cells is None or column not in cells.columns:
        return None
    return cells[column].to_numpy()


def _picked(art: dict, preset: str, budget: int) -> gpd.GeoDataFrame | None:
    """Candidate rows for the first ``budget`` recommendations."""
    queue, candidates = queue_for(art, preset), art.get("candidates")
    if queue is None or candidates is None or queue.empty:
        return None
    return candidates[candidates["site_id"].isin(queue.head(budget)["site_id"])]


def plan_map(
    art: dict,
    preset: str,
    budget: int,
    layers_on: dict,
    flash: int | None = None,
    pick_mode: str = "Off",
):
    """Assemble the Plan map: covered ground, remaining gaps, and what the plan adds.

    ``pick_mode`` adds the clickable candidate layer. It is off by default so the
    map stays a picture of the plan until a planner explicitly starts overriding.
    """
    s = art["summary"]
    radius = s["threshold_m"]
    existing, grid, cells = art.get("existing_aeds"), art.get("grid"), art.get("cells")
    steps = steps_for(art, preset)
    layers = []

    if layers_on["demand"] and grid is not None and cells is not None:
        # Uncovered *at this step*, so gaps genuinely close as the budget grows.
        if steps is not None:
            uncovered = ~((steps >= 0) & (steps <= budget))
        else:
            uncovered = ~cells["covered_before"].to_numpy()
        layers.append(
            viz.demand_hexes(
                grid,
                cells["demand"].to_numpy(),
                uncovered,
                unreachable_cells(s["town"]),
            )
        )
        if flash is not None and steps is not None:
            layers.append(viz.flash_hexes(grid, steps == flash))
    if layers_on["existing"] and existing is not None:
        layers += [viz.coverage_area(existing, radius), viz.device_points(existing, viz.EXISTING)]
    proposed = _picked(art, preset, budget)
    if layers_on["proposed"] and proposed is not None and len(proposed):
        layers += [
            viz.new_ground(proposed, existing, radius),
            viz.device_points(proposed, viz.PROPOSED, radius_m=45),
        ]
    if layers_on["worst"]:
        layers.append(
            viz.worst_point_layer(s.get("worst_point_before"), s.get("worst_radius_before", 0))
        )
    if pick_mode != "Off":
        slug = s["town"]
        layers.append(
            viz.candidate_markers(
                art.get("candidates"),
                proposed_ids=tuple(proposed["site_id"]) if proposed is not None else (),
                pinned=tuple(st.session_state.get(f"pins_{slug}", [])),
                vetoed=tuple(st.session_state.get(f"vetos_{slug}", [])),
            )
        )
    return viz.deck(layers, existing if existing is not None else grid)


def cost_of_choice(art: dict, preset: str) -> str:
    """One line stating what this priority ordering costs against the default."""
    policies = art["summary"].get("policies", {})
    default = next(iter(PRESETS))
    if preset not in policies or default not in policies:
        return ""
    here, base = policies[preset]["coverage"], policies[default]["coverage"]
    keys = " → ".join(KEY_LABELS.get(k, k).lower() for k in policies[preset]["keys"])
    delta = (here - base) * 100
    if preset == default:
        return f"Ranking by {keys}. Coverage {here:.1%} — the default."
    return f"Ranking by {keys}. Coverage {here:.1%} ({delta:+.1f} points against the default ordering)."


def _gap_breakdown(art: dict, preset: str, budget: int) -> None:
    """Say, in words, what the amber and the inert shading each add up to.

    The two are entirely different problems -- one is a funding backlog, the
    other is the candidate pool running out -- and reading them off a map by eye
    is exactly what a viewer cannot do. Derived from the same masks the layer
    uses, so the caption cannot drift away from the picture.
    """
    cells, steps = art.get("cells"), steps_for(art, preset)
    dead = unreachable_cells(art["summary"]["town"])
    if cells is None or dead is None or len(dead) != len(cells):
        return
    demand = cells["demand"].to_numpy(dtype=float)
    total = demand.sum() or 1.0
    uncovered = (
        ~((steps >= 0) & (steps <= budget))
        if steps is not None
        else ~cells["covered_before"].to_numpy()
    )
    gap = demand[uncovered].sum() / total
    inert = demand[uncovered & dead].sum() / total
    fundable = gap - inert
    if gap <= 0:
        return
    st.caption(
        f"**{gap:.1%} of demand is still uncovered at {budget} device(s).** "
        f"Of that, **{fundable * 100:.1f} points** closes as you fund more of this same "
        f"queue — those are the amber cells. The remaining **{inert * 100:.1f} points** has "
        "no eligible building within reach of it, so no plan at any budget covers "
        "it; those cells are drawn muted. Widening the candidate pool, not a better "
        "optimiser, is what would move them."
    )


def _handle_map_click(event, slug: str, target: str) -> None:
    """Turn a map selection into a pending override, then rerun.

    The override multiselects live in the sidebar, which Streamlit renders
    *before* the tabs; writing to a widget's key after that widget exists raises.
    So the click only records an intent here, and :func:`apply_pending_overrides`
    folds it into the widget state at the top of the next run, before the
    multiselect is built.

    Clicks are additive on purpose. A click that toggled would fight the
    selection state, which survives the rerun and would immediately re-fire;
    removal therefore stays with the sidebar lists, which is also the path that
    works without a mouse.
    """
    if event is None:
        return
    selection = getattr(event, "selection", None) or {}
    objects = selection.get("objects") or {}
    picked = objects.get(viz.CANDIDATE_LAYER_ID) or []
    if not picked:
        return
    site_id = picked[0].get("site_id")
    if not site_id or site_id in st.session_state.get(f"{target}_{slug}", []):
        return
    st.session_state[f"_pending_{target}_{slug}"] = site_id
    st.rerun()


def apply_pending_overrides(slug: str) -> None:
    """Fold map-click intents into the override widgets before they are built."""
    for target in ("pins", "vetos"):
        pending = st.session_state.pop(f"_pending_{target}_{slug}", None)
        if not pending:
            continue
        current = list(st.session_state.get(f"{target}_{slug}", []))
        if pending not in current:
            st.session_state[f"{target}_{slug}"] = [*current, pending]


#: Plain-language names for the ranking methods. The keys stay the technical
#: names because tests and docs refer to them.
METHOD_LABELS = {
    "Marginal-coverage greedy": "Most new people reached per device (recommended)",
    "Demand-ranked top-k (baseline)": "Busiest spots first (simple baseline)",
    "Greedy k-center (equity)": "Worst-served places first (fairness)",
}

#: What each stated-hours bucket means to a reader. These are the hours the
#: registry lists, not checked access.
HOURS_LABELS = {
    "always_open": "24 hours, every day",
    "extended": "Long hours (14 h+ a day)",
    "daytime": "Daytime only",
    "closed": "Listed as closed",
    "unparseable": "Hours unclear",
}
_HOURS_RANK = list(HOURS_LABELS)
_HOURS_COLOURS = {
    "always_open": [33, 128, 84],
    "extended": [52, 120, 190],
    "daytime": [214, 140, 40],
    "closed": [120, 120, 120],
    "unparseable": [120, 120, 120],
}


@st.cache_data(show_spinner=False)
def buildings_with_aeds(slug: str) -> pd.DataFrame | None:
    """Existing devices grouped by building, with floors, locations and listed hours.

    One building often holds several devices (a mall, a hospital, a sports hall),
    and a map of single dots hides that. Grouping uses the postal code, which in
    Singapore identifies one building.
    """
    existing = load_town(slug).get("existing_aeds")
    if existing is None or len(existing) == 0:
        return None
    frame = pd.DataFrame(existing.drop(columns="geometry"))
    frame["hours_bucket"] = classify_frame(frame)
    frame["building_key"] = frame["POSTAL_CODE"].fillna(frame["AED_ID"])

    def _join(values) -> str:
        seen = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
        return "; ".join(dict.fromkeys(seen)) or "—"

    rows = []
    for _, group in frame.groupby("building_key", sort=False):
        first = group.iloc[0]
        buckets = sorted(set(group["hours_bucket"]), key=_HOURS_RANK.index)
        best = buckets[0]
        hours = HOURS_LABELS[best] + (" (varies by device)" if len(buckets) > 1 else "")
        street = " ".join(
            str(v) for v in (first.get("HOUSE_NUMBER"), first.get("ROAD_NAME")) if pd.notna(v)
        )
        rows.append(
            {
                "building": first.get("label") or first.get("BUILDING_NAME") or street,
                "street": street,
                "aeds": len(group),
                "floors": _join(
                    sorted(
                        group["AED_LOCATION_FLOOR_LEVEL"].dropna().astype(str),
                        key=lambda f: (not f.lstrip("B-").isdigit(), f.zfill(4)),
                    )
                ),
                "where": _join(group["AED_LOCATION_DESCRIPTION"]),
                "hours": hours,
                "hours_bucket": best,
                "lat": float(group["LATITUDE"].mean()),
                "lon": float(group["LONGITUDE"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["aeds", "building"], ascending=[False, True])


def _buildings_map(buildings: pd.DataFrame, centre) -> pdk.Deck:
    """Buildings with devices, sized by count and coloured by listed hours."""
    data = buildings.assign(
        label=lambda d: d["building"]
        + " · "
        + d["aeds"].astype(str)
        + " AED(s) · "
        + d["hours"],
        colour=lambda d: d["hours_bucket"].map(_HOURS_COLOURS),
        radius=lambda d: 18 + 10 * np.sqrt(d["aeds"]),
    )[["lon", "lat", "label", "colour", "radius"]]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data.to_dict("records"),
        get_position="[lon, lat]",
        get_fill_color="colour",
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=22,
        opacity=0.85,
        pickable=True,
    )
    return viz.deck([layer], centre)


def _diminishing_returns(art: dict, preset: str) -> dict | None:
    """Numbers behind "why not put one everywhere", from the current queue."""
    queue = queue_for(art, preset)
    if queue is None or queue.empty:
        return None
    gains = queue["coverage_added_pp"].to_numpy()
    out = {
        "ceiling": float(queue["cumulative_coverage"].iloc[-1]),
        "useful": int((gains > 0).sum()),
        "ninety": int((gains.cumsum() >= 0.9 * gains.sum()).argmax()) + 1,
        "unreachable": None,
    }
    cells, dead = art.get("cells"), unreachable_cells(art["summary"]["town"])
    if cells is not None and dead is not None and len(dead) == len(cells):
        demand = cells["demand"].to_numpy(dtype=float)
        uncovered = ~cells["covered_before"].to_numpy(dtype=bool)
        out["unreachable"] = float(demand[uncovered & dead].sum() / (demand.sum() or 1.0))
    return out


# --- screens ---------------------------------------------------------------


def screen_intro(art: dict, preset: str, budget: int) -> None:
    """Start here: what an AED is, what this tool does, and why not one everywhere."""
    s = art["summary"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("What is an AED?")
        st.markdown(
            "An **AED (automated external defibrillator)** is a portable device, about "
            "the size of a lunchbox, kept in a wall cabinet in public places. When "
            "someone's heart suddenly stops pumping (**sudden cardiac arrest**), the AED "
            "reads the heart rhythm through two sticky pads and, only if a shock can "
            "help, delivers one. It speaks each step out loud, so anyone can use it "
            "without training."
        )
        st.subheader("Why the distance matters")
        st.markdown(
            "In a cardiac arrest, every minute before CPR and a shock counts, and an "
            "ambulance takes time to arrive. The best chance is a bystander who starts "
            "CPR while someone else fetches a nearby AED. This tool assumes the fetch "
            f"must be done within a **{s['budget_min']:.0f}-minute round trip** on foot, "
            f"which works out to about **{s['threshold_m']:.0f} m** of walking each way. "
            "A place counts as **covered** if an AED is within that reach."
        )
        st.subheader("What this tool does")
        st.markdown(
            f"1. **Existing AEDs**: shows the {s['existing_aeds']} public AEDs registered in "
            f"{s['town_label']}, grouped by building.\n"
            "2. **Plan new AEDs**: finds where people are *not* covered and ranks "
            f"{s['candidates']} possible host buildings by how much each new device would "
            "help. You choose how many you can fund in the sidebar.\n"
            "3. **Stress-test**: checks the plan against simulated emergencies and broken "
            "devices.\n"
            "4. **Evidence**: compares the method with simpler ones and lists its limits."
        )
        st.info(OFFICIAL_CHANNELS)
    with right:
        st.subheader("Why not put one everywhere?")
        numbers = _diminishing_returns(art, preset)
        st.markdown(
            "- **Each one costs money to keep ready.** Pads and batteries expire, "
            "cabinets need checking, and someone has to own that job for years.\n"
            "- **Each one needs a host.** A building has to agree, keep it reachable, "
            "and be open when it is needed.\n"
            "- **The benefit shrinks fast.** The first few devices fill the biggest gaps. "
            "Later ones mostly overlap with AEDs that are already there."
        )
        if numbers is None:
            st.warning(safe_failure("the install queue for this area is empty."))
        else:
            unreachable = (
                f" **{numbers['unreachable']:.0%}** of demand has no suitable building "
                "within reach at all, so more devices cannot fix it."
                if numbers["unreachable"]
                else ""
            )
            st.markdown(
                f"In **{s['town_label']}**, coverage today is "
                f"**{s['baseline_coverage']:.0%}**. The first **{numbers['ninety']}** new "
                f"devices deliver 90% of everything new devices can add. After "
                f"**{numbers['useful']}**, extra devices add no new coverage, only "
                f"slightly shorter trips. The ceiling is **{numbers['ceiling']:.0%}**."
                f"{unreachable}"
            )
            st.altair_chart(
                viz.coverage_curve(queue_for(art, preset), s["baseline_coverage"], budget),
                width="stretch",
            )
            st.caption(
                "Coverage (up) against number of new devices (across). The curve "
                "flattens: that flat part is why the question is *which* sites first, "
                "not *how many*."
            )
    with st.expander("Words used in this app"):
        st.markdown(
            f"- **Coverage**: the share of people (modelled) within {s['threshold_m']:.0f} m "
            "walking of an AED.\n"
            "- **Demand**: an estimate of where people are, built from housing and "
            "building data. It is not a count of past cardiac arrests.\n"
            "- **Candidate site**: a building that passed the checks to host a device "
            "(public, reachable, the right type).\n"
            "- **Install queue**: candidate sites in the order the tool recommends "
            "funding them.\n"
            "- **Percentage points**: the difference between two percentages, e.g. "
            "60% → 62% is +2 points."
        )


def screen_existing(art: dict) -> None:
    """Existing devices by building: how many, which floor, where inside, listed hours."""
    s = art["summary"]
    buildings = buildings_with_aeds(s["town"])
    if buildings is None or buildings.empty:
        st.warning(safe_failure("no existing device records were found for this area."))
        return

    multi = buildings[buildings["aeds"] > 1]
    all_day = int(
        (classify_frame(pd.DataFrame(art["existing_aeds"].drop(columns="geometry")))
         == "always_open").sum()
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered AEDs", s["existing_aeds"])
    c2.metric("Buildings with an AED", len(buildings))
    c3.metric(
        "Buildings with 2 or more",
        len(multi),
        help=f"Together they hold {int(multi['aeds'].sum())} devices.",
    )
    c4.metric(
        "Listed as 24 hours",
        f"{all_day} of {s['existing_aeds']}",
        help="As written in the registry's hours field, not checked.",
    )

    st.warning(
        "**What this public data cannot tell you:** whether a device works today, when "
        "it was last inspected, when its pads or battery expire, or whether its cabinet "
        "is locked. The public dataset has no such fields, and this tool does not guess "
        "them. Records come from a snapshot dated February 2020."
    )

    shown_hours = st.multiselect(
        "Show buildings by listed hours",
        list(HOURS_LABELS),
        default=[b for b in HOURS_LABELS if b in set(buildings["hours_bucket"])],
        format_func=HOURS_LABELS.get,
    )
    view = buildings[buildings["hours_bucket"].isin(shown_hours)]
    if view.empty:
        st.info("No buildings match these filters.")
        return

    st.pydeck_chart(_buildings_map(view, art["existing_aeds"]), key=f"map_existing_{s['town']}")
    st.markdown(
        viz.legend(
            [(HOURS_LABELS[b], [*_HOURS_COLOURS[b], 230]) for b in ("always_open", "extended",
                                                                   "daytime", "closed")]
        )
        + " Bigger dot = more devices in the building.",
        unsafe_allow_html=True,
    )
    st.dataframe(
        view[["building", "street", "aeds", "floors", "where", "hours"]],
        width="stretch",
        hide_index=True,
        column_config={
            "building": "Building",
            "street": "Street",
            "aeds": st.column_config.NumberColumn("AEDs", width="small"),
            "floors": "Floor(s)",
            "where": "Where inside",
            "hours": "Listed hours",
        },
    )
    st.caption(
        "Hours are the free-text hours in the registry, sorted into groups. Where a "
        "building's devices list different hours, the longest is shown and marked "
        "'varies by device'."
    )


def screen_plan(art: dict, preset: str, budget: int, layers_on: dict) -> None:
    """What to buy: map, queue, and the curve that says when to stop."""
    s = art["summary"]
    queue = queue_for(art, preset)
    if queue is None or queue.empty:
        st.warning(safe_failure("no candidate site produced a positive coverage gain."))
        return

    shown = queue.head(budget)
    before, after = s["baseline_coverage"], float(shown["cumulative_coverage"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage now", f"{before:.1%}")
    c2.metric(f"With {budget} more", f"{after:.1%}", f"{(after - before) * 100:.1f} points")
    c3.metric("Reach per device", f"{s['threshold_m']:.0f} m")
    c4.metric(
        "Worst-served point",
        f"{s.get('worst_radius_before', 0):.0f} m",
        help=(
            "This does not move with the budget or the priority ordering, and that is not a "
            "fault. The nearest eligible candidate building is further from that point than a "
            "device reaches, so no placement can cover it. Only widening the candidate pool "
            "or placing a device manually would change it."
        ),
    )

    pins, vetos = art.get("_override_counts", (0, 0))
    free_queue = art.get("_unconstrained")
    if (pins or vetos) and free_queue is not None and not free_queue.empty:
        at = min(budget, len(free_queue), len(queue))
        unconstrained = float(free_queue["cumulative_coverage"].iloc[at - 1])
        constrained = float(queue["cumulative_coverage"].iloc[at - 1])
        cost = (unconstrained - constrained) * 100
        if abs(cost) < 0.005:
            message = f"**{pins} pinned, {vetos} vetoed** — no measurable cost at this budget."
        elif cost > 0:
            message = (
                f"**{pins} pinned, {vetos} vetoed.** Your overrides cost {cost:.2f} percentage "
                f"points at this budget ({constrained:.1%} against {unconstrained:.1%} "
                "unconstrained). Everything else was re-optimised around them."
            )
        else:
            # Greedy is a heuristic, not an exact method, so a forced first pick can
            # occasionally lead it down a better path. Saying so is more honest than
            # implying a constraint can never help.
            message = (
                f"**{pins} pinned, {vetos} vetoed** — and the plan improved by "
                f"{-cost:.2f} points. The placement heuristic is not exact, so a constraint can "
                "occasionally steer it somewhere better."
            )
        st.warning(message)

    slug = s["town"]
    mode = st.radio(
        "Click the map to override",
        ("Off", "Pin", "Veto"),
        horizontal=True,
        key=f"pickmode_{slug}",
        help=(
            "Pin forces a site into the plan; veto excludes it. Either way the model "
            "re-solves around your choice and shows what it cost."
        ),
    )
    if mode != "Off":
        st.caption(
            f"**{mode} mode.** Click a site marker to {mode.lower()} it. Only vetted "
            "candidate buildings are clickable — clicking open ground does nothing, "
            "because a device can only be placed in a building that passed the "
            "eligibility filter. Use the sidebar lists to undo."
        )

    # A stable key keeps one deck.gl instance alive across reruns. Without it
    # Streamlit remounts the component whenever the layers change, deck.gl
    # re-applies `initial_view_state`, and the camera snaps back to the fitted
    # centre -- which reads as "the map didn't update" when in fact it redrew
    # and jumped home. Keyed by town, so switching town still refits.
    event = st.pydeck_chart(
        plan_map(art, preset, budget, layers_on, pick_mode=mode),
        key=f"map_plan_{slug}",
        on_select="rerun" if mode != "Off" else "ignore",
        selection_mode="single-object",
    )
    if mode != "Off":
        _handle_map_click(event, slug, "pins" if mode == "Pin" else "vetos")
    st.markdown(
        viz.legend(viz.PLAN_LEGEND + (viz.CANDIDATE_LEGEND if mode != "Off" else [])),
        unsafe_allow_html=True,
    )
    _gap_breakdown(art, preset, budget)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Install queue**")
        columns = ["rank", "site_id", "label", "coverage_added_pp", "cumulative_coverage"]
        for optional in ("new_ground_share", "deciding_key", "stable"):
            if optional in shown.columns:
                columns.append(optional)
        st.dataframe(
            shown[columns],
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "site_id": st.column_config.TextColumn("ID", width="small"),
                "label": "Site",
                "coverage_added_pp": st.column_config.NumberColumn("Adds (pp)", format="%.2f"),
                "cumulative_coverage": st.column_config.NumberColumn("Running", format="percent"),
                "new_ground_share": st.column_config.NumberColumn("New ground", format="percent"),
                "deciding_key": st.column_config.TextColumn("Chosen by"),
                # `format="percent"` scales a 0-1 float; "%.0f%%" would render
                # 0.78 as "1%", which is what made this column look meaningless.
                "stable": st.column_config.ProgressColumn(
                    "Robust", min_value=0.0, max_value=1.0, format="percent"
                ),
            },
        )
        st.caption(
            "*New ground* = share of this device's reach the existing network does not cover. "
            "*Chosen by* = which priority actually decided the pick."
        )
    with right:
        st.markdown("**When to stop buying**")
        st.altair_chart(viz.coverage_curve(queue, before, budget), width="stretch")
        st.altair_chart(viz.marginal_gain_chart(queue), width="stretch")


def screen_test(art: dict, preset: str, layers_on: dict, budget: int, seed: int) -> None:
    """Does it hold up: animation, service areas, simulation, stress, sensitivity."""
    st.caption(
        "Does the plan hold up? Watch it fill the gaps, then test it against "
        "simulated emergencies, broken devices and changed assumptions."
    )
    s = art["summary"]
    demo, service, sim_tab, stress_tab, scenario_tab = st.tabs(
        ["▶ Watch the plan", "Service areas", "Simulated incidents", "Outages", "Robustness"]
    )

    with demo:
        queue, steps = queue_for(art, preset), steps_for(art, preset)
        st.markdown("**Watch the gaps close, one funded device at a time.**")
        if queue is None or queue.empty:
            st.warning(safe_failure("nothing to animate."))
        elif st.button("▶ Play", type="primary"):
            slot = st.empty()
            for step in range(0, len(queue) + 1):
                row = queue.iloc[step - 1] if step else None
                coverage = (
                    s["baseline_coverage"] if step == 0 else float(row["cumulative_coverage"])
                )
                gained = int((steps == step).sum()) if steps is not None and step else 0
                remaining = (
                    int((steps < 0).sum() + (steps > step).sum()) if steps is not None else 0
                )
                with slot.container():
                    a, b, c, d = st.columns(4)
                    a.metric("Devices funded", step)
                    b.metric("Coverage", f"{coverage:.1%}")
                    c.metric("Cells gained", gained)
                    d.metric("Still uncovered", remaining)
                    st.markdown(
                        f"**#{step} {row['label']}** — adds {row['coverage_added_pp']:.2f} pp, "
                        f"{gained} cell(s) newly covered"
                        if step
                        else "**Existing network only** — press through to fund devices."
                    )
                    st.pydeck_chart(
                        plan_map(art, preset, step, layers_on, flash=step if step else None),
                        key=f"map_demo_{art['summary']['town']}",
                    )
                time.sleep(0.85)
            st.caption("Replaying precomputed results — nothing is re-solved.")
        else:
            st.pydeck_chart(
                plan_map(art, preset, 0, layers_on),
                key=f"map_demo_{art['summary']['town']}",
            )
            st.caption("Press Play to fund devices one at a time and watch the gaps close.")

    with service:
        st.markdown("**Who serves whom, and how unevenly.**")
        metrics = art.get("voronoi_metrics")
        cells_before, cells_after = art.get("voronoi_before"), art.get("voronoi_after")
        if metrics is None or cells_before is None:
            st.warning(safe_failure("service-area artifacts are missing."))
        else:
            stage = st.radio(
                "Stage", ["before", "after"], horizontal=True, label_visibility="collapsed"
            )
            row = metrics[metrics["stage"] == stage].iloc[0]
            base = metrics[metrics["stage"] == "before"].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Load imbalance (Gini)",
                f"{row['load_gini']:.2f}",
                f"{row['load_gini'] - base['load_gini']:+.2f}" if stage == "after" else None,
            )
            c2.metric("Busiest device serves", f"{row['max_load']:.1%}")
            c3.metric("Worst-served point", f"{row['largest_empty_circle_m']:.0f} m")
            cells = cells_after if stage == "after" and cells_after is not None else cells_before
            st.pydeck_chart(
                viz.deck(
                    [
                        viz.voronoi_layer(cells),
                        viz.worst_point_layer(
                            s.get(f"worst_point_{stage}"), s.get(f"worst_radius_{stage}", 0)
                        ),
                    ],
                    cells,
                ),
                key=f"map_service_{s['town']}",
            )
            st.markdown(viz.legend(viz.SERVICE_LEGEND), unsafe_allow_html=True)
            st.caption("Each area is the ground closest to one device. Hover to name it.")
            with st.expander("Layout quality metrics"):
                st.dataframe(metrics, width="stretch", hide_index=True)

    with sim_tab:
        queue = queue_for(art, preset)
        sites = tuple(queue.head(budget)["site_index"].tolist()) if queue is not None else ()
        live = live_simulation(art["summary"]["town"], sites, seed)
        if live is None:
            st.warning(safe_failure("simulation needs solver.npz — re-run the pipeline."))
        else:
            before, after = live["before"].incidents, live["after"].incidents
            gain = after["covered"].mean() - before["covered"].mean()
            st.markdown(
                f"**Of 1,000 simulated emergencies, {before['covered'].mean():.0%} have a device "
                f"in reach today; funding {budget} takes that to {after['covered'].mean():.0%}.**"
            )
            c1, c2 = st.columns(2)
            c1.metric("Reached in time — now", f"{before['covered'].mean():.1%}")
            c2.metric(
                f"With {budget} more", f"{after['covered'].mean():.1%}", f"{gain * 100:.1f} points"
            )
            st.altair_chart(viz.retrieval_histogram(before, after), width="stretch")
            st.caption(
                f"Synthetic, seed {seed}, recomputed for the current budget. Illustrative — "
                "reported metrics use the exact analytic calculation, not this sample."
            )
            grid = art.get("grid")
            if grid is not None:
                which = st.radio(
                    "Scenario",
                    ["after", "before"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="incident_scenario",
                )
                st.pydeck_chart(
                    viz.deck([viz.incident_points(after if which == "after" else before)], grid),
                    key=f"map_incidents_{s['town']}",
                )
                st.markdown(viz.legend(viz.INCIDENT_LEGEND), unsafe_allow_html=True)
                st.caption("Dots and bars share one colour language, so they read together.")

    with stress_tab:
        crit, existing = art.get("criticality"), art.get("existing_aeds")
        st.markdown("**Which devices in particular?**")
        if crit is None or crit.empty:
            st.warning(safe_failure("criticality artifacts are missing."))
        else:
            sole = crit[crit["sole_cover_demand"] > 0]
            st.markdown(
                f"**{len(sole)} of {len(crit)} devices are the *only* cover for some demand.** "
                "Losing one of these strands people nothing else reaches; losing any of the "
                f"other {len(crit) - len(sole)} costs nothing, because a neighbour already covers "
                "the same ground."
            )
            st.pydeck_chart(
                viz.deck(
                    [
                        viz.coverage_area(existing, s["threshold_m"]),
                        viz.criticality_points(existing, crit),
                    ],
                    existing,
                ),
                key=f"map_criticality_{s['town']}",
            )
            st.markdown(
                viz.legend(
                    [
                        (
                            "Single point of failure — bigger = more demand at risk",
                            [*viz.UNCOVERED, 220],
                        ),
                        ("Covered today", [*viz.EXISTING, 120]),
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(
                sole.head(15)[["AED_ID", "label", "sole_cover_demand"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "AED_ID": st.column_config.TextColumn("Device", width="small"),
                    "label": "Location",
                    "sole_cover_demand": st.column_config.NumberColumn(
                        "Demand it alone protects", format="percent"
                    ),
                },
            )
            st.caption(
                "This flags devices whose loss would hurt — it is never a recommendation to "
                "remove one. It informs maintenance priority and the replacement cycle."
            )

        stress_queue = queue_for(art, preset)
        stress_sites = (
            tuple(stress_queue.head(budget)["site_index"].tolist())
            if stress_queue is not None
            else ()
        )
        live = live_simulation(art["summary"]["town"], stress_sites, seed)
        outages = live["outages"] if live else art.get("outage_stress")
        if outages is not None and not outages.empty:
            st.divider()
            st.markdown(f"**If devices fail at random** — seed {seed}")
            baseline = live["outage_baseline"] if live else s["baseline_coverage"]
            st.altair_chart(viz.outage_chart(outages, baseline), width="stretch")
            if live:
                funded = live["n_network"] - live["n_existing"]
                st.caption(
                    f"Failing devices at random from the **{live['n_existing']} existing "
                    f"+ {funded} funded** network — a device this plan recommends is as "
                    f"liable to be out as one already installed. Coverage lost is measured "
                    f"against the {baseline:.1%} the full network reaches when nothing fails."
                )
            st.caption(
                "The aggregate view: uniform random failures. Real failures cluster, so this "
                "is optimistic — the named devices above are the specific risk."
            )

    with scenario_tab:
        st.markdown(
            "**The size of the gain is robust; the identity of the sites is not.** "
            "No site survives 90% of scenarios."
        )
        st.caption(
            "This is a **deterministic** sensitivity analysis, not a sampled one. "
            "Every scenario below — each retrieval threshold, demand proxy and "
            "operating-hours assumption — is enumerated exhaustively, so these "
            "results do not move with the simulation seed. That is why re-rolling "
            "the seed changes the incident and outage views but leaves this tab "
            "unchanged."
        )
        stability, candidates = art.get("site_stability"), art.get("candidates")
        if stability is not None and not stability.empty and "label" in stability.columns:
            st.markdown("**Which sites survive the sweep, and where they are**")
            st.pydeck_chart(
                viz.deck([viz.stability_points(candidates, stability)], candidates),
                key=f"map_stability_{s['town']}",
            )
            st.markdown(
                viz.legend(
                    [("Chosen often across scenarios — bigger = more robust", [*viz.PROPOSED, 215])]
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(
                stability.head(12)[["site_id", "label", "times_chosen", "share"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "site_id": st.column_config.TextColumn("ID", width="small"),
                    "label": "Site",
                    "times_chosen": "Scenarios",
                    "share": st.column_config.ProgressColumn(
                        "Robust", min_value=0.0, max_value=1.0, format="percent"
                    ),
                },
            )

        proxies = art.get("proxy_agreement")
        if proxies is not None and not proxies.empty and "area" in proxies.columns:
            lowest = proxies.set_index(proxies.columns[0])["area"].drop("area", errors="ignore")
            if not lowest.empty:
                st.caption(
                    f"Demand-proxy choice moves the answer: the residential-area proxy and "
                    f"`{lowest.idxmin()}` rank-correlate at only {lowest.min():.2f}."
                )

        for key, caption in (
            ("threshold_sweep", "Coverage at 2, 3 and 4-minute budgets."),
            ("hours_scenario", "Counting only devices stating round-the-clock access."),
            ("cross_proxy_sensitivity", "Solved under one demand assumption, scored under all."),
        ):
            frame = art.get(key)
            if frame is not None and not frame.empty:
                with st.expander(key.replace("_", " ").title()):
                    st.dataframe(frame.head(12), width="stretch", hide_index=True)
                    st.caption(caption)


def screen_evidence(art: dict, preset: str, budget: int) -> None:
    """Why believe it, in three tiers: DECIDE, VERIFY, PROVENANCE.

    Ordered so a planner meets the decisions first and the method defence second.
    Every headline states the takeaway; every DECIDE panel ends in the decision it
    feeds.
    """
    st.caption(
        "Why trust the list: how it compares with simpler methods, how stable it is "
        "when the inputs change, and where the data comes from."
    )
    s = art["summary"]
    worst = s.get("worst_radius_before", 0)
    reach = s["threshold_m"]

    # ---- the through-line ---------------------------------------------------
    calibration = art.get("eligibility_validation")
    outside = 0.0
    nomap = 0.0
    if calibration is not None:
        total = calibration["existing_aeds"].sum()
        eligible = calibration.loc[calibration["decision"] == "eligible", "existing_aeds"].sum()
        nomap = float(
            calibration.loc[calibration["decision"] == "no OSM building", "existing_aeds"].sum()
            / total
        )
        outside = float((total - eligible) / total)

    st.markdown(
        f"### The binding constraint is the candidate pool, not the algorithm\n"
        f"Three separate results say the same thing. The worst-served point sits "
        f"**{worst:.0f} m** from any device while a device reaches **{reach:.0f} m**, and the "
        f"nearest *eligible building* to it is further than that — so no objective at any "
        f"weighting can cover it. Across every ordering the worst case barely moves. And "
        f"**{outside:.0%}** of the devices SCDF actually deployed sit outside our candidate "
        f"pool ({nomap:.0%} in buildings OpenStreetMap does not have at all, the rest refused "
        f"by our eligibility rules).\n\n"
        "**Widening the pool — surveying sites, or admitting building types we currently "
        "abstain on — would do more than any change to the solver.**"
    )
    st.divider()

    # ================= DECIDE ================================================
    st.markdown("## Decide")

    queue = queue_for(art, preset)
    if queue is not None and not queue.empty:
        cumulative = queue["cumulative_coverage"].to_numpy()
        knee = viz.knee_of(cumulative)
        gains = queue["coverage_added_pp"].to_numpy()
        ninety = int((gains.cumsum() >= 0.9 * gains.sum()).argmax()) + 1
        st.markdown(
            f"**Fund about {knee} devices. Past that the curve flattens.** "
            f"The first {knee} take coverage from {s['baseline_coverage']:.0%} to "
            f"{cumulative[knee - 1]:.0%}; reaching the ceiling of {cumulative[-1]:.0%} needs "
            f"{len(queue)} — {len(queue) - knee} more devices for "
            f"{(cumulative[-1] - cumulative[knee - 1]) * 100:.0f} further points."
        )
        st.altair_chart(
            viz.coverage_curve(queue, s["baseline_coverage"], budget, knee),
            width="stretch",
        )
        st.caption(
            f"Amber line: the knee. Grey dashed: your current budget of {budget}. "
            f"90% of everything achievable arrives by {ninety}."
        )
        st.info(
            f"**Decision:** set the programme at ~{knee} devices, or at {ninety} if the "
            "budget stretches. Beyond that, money buys more from widening the candidate "
            "pool than from more devices."
        )
        st.divider()

    objectives = art.get("objectives")
    if objectives is not None and not objectives.empty:
        frame = objectives[objectives["objective"] != "do nothing"].copy()
        varying = [c for c in ("coverage", "max_m", "mean_m") if frame[c].nunique() > 1]
        st.markdown(
            "**No ordering closes the biggest hole — that needs new candidate buildings.** "
            "The orderings differ on average trip and on which neighbourhoods gain, not on "
            "whether the worst-served point gets served."
        )
        if varying:
            st.altair_chart(
                viz.priority_comparison(
                    frame[["objective", *varying]].rename(columns={"objective": "priority"})
                ),
                width="stretch",
            )
        st.info(
            "**Decision:** choose the ordering by whose problem you are solving — most "
            "people reached, or the worst-served areas first. It is a policy choice, and "
            "the coverage cost of choosing equity is small."
        )
        st.divider()

    subzones = art.get("subzones")
    if subzones is not None and len(subzones):
        worst_before = subzones["coverage_before"].min()
        worst_after = subzones["coverage_after"].min()
        untouched = subzones[subzones["delta_pp"] <= 0.01]["subzone"].tolist()
        st.markdown(
            f"**The worst-served subzone goes from {worst_before:.0%} to {worst_after:.0%}.** "
            "The minimum, not the average, is what equity means."
        )
        left, right = st.columns(2)
        for column, stage, panel in (
            ("coverage_before", "Today", left),
            ("coverage_after", "After the plan", right),
        ):
            with panel:
                st.caption(f"**{stage}**")
                st.pydeck_chart(
                    viz.deck([viz.subzone_choropleth(subzones, column)], subzones),
                    key=f"map_subzone_{column}_{s['town']}",
                )
        st.markdown(
            viz.legend([("0% covered", [214, 70, 60, 205]), ("100% covered", [64, 180, 100, 205])]),
            unsafe_allow_html=True,
        )
        if untouched:
            st.info(
                f"**Decision:** {len(untouched)} subzone(s) gain nothing at this budget — "
                f"{', '.join(untouched)}. Either accept that, or switch to worst-served-first "
                "and re-read the queue."
            )
        with st.expander("Numbers"):
            st.dataframe(
                subzones[["subzone", "coverage_before", "coverage_after", "delta_pp"]].sort_values(
                    "coverage_after"
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "subzone": "Subzone",
                    "coverage_before": st.column_config.NumberColumn("Today", format="percent"),
                    "coverage_after": st.column_config.NumberColumn("After", format="percent"),
                    "delta_pp": st.column_config.NumberColumn("Change (pp)", format="%.1f"),
                },
            )
        st.divider()

    # ================= VERIFY ================================================
    st.markdown("## Verify")

    nesting = art.get("nesting_loss")
    if nesting is not None and not nesting.empty:
        largest = nesting.iloc[-1]
        st.markdown(
            f"**You can fund in tranches at no cost.** Buying {int(largest['k'])} devices in "
            f"stages instead of all at once forfeits "
            f"{largest['nesting_loss_pp']:.1f} percentage points, because the queue is nested — "
            "each year's choices stay optimal when the next year's are added."
        )
        st.dataframe(
            nesting[["k", "ilp_coverage", "greedy_coverage", "nesting_loss_pp"]],
            width="stretch",
            hide_index=True,
            column_config={
                "k": "Devices",
                "ilp_coverage": st.column_config.NumberColumn("Buy all at once", format="percent"),
                "greedy_coverage": st.column_config.NumberColumn("In tranches", format="percent"),
                "nesting_loss_pp": st.column_config.NumberColumn("Cost (pp)", format="%.2f"),
            },
        )
        st.caption("An integer program certifies the all-at-once optimum, so the gap is proven.")
        st.divider()

    comparison = art.get("method_vs_baseline")
    if comparison is not None:
        band = s.get("random_k_band", {})
        gain = comparison.iloc[2]["coverage"] - comparison.iloc[1]["coverage"]
        st.markdown(
            f"**The method beats the required baseline by {gain * 100:.1f} points** at the same "
            "budget, and clears the random-placement band."
        )
        st.altair_chart(viz.method_bar(comparison), width="stretch")
        st.caption(
            f"Random placement over 50 seeds reaches {band.get('mean', 0):.1%} "
            f"(p95 {band.get('p95', 0):.1%})."
        )
        st.divider()

    # ================= PROVENANCE ===========================================
    st.markdown("## Provenance")
    st.caption("Where the numbers come from. Collapsed because none of it changes a decision.")

    with st.expander("What 'demand' actually means"):
        st.markdown(
            "Demand is **building footprint area per grid cell**, used as a proxy for where "
            "people are. It is **not** population and **not** cardiac-arrest incidence: the "
            "pipeline contains no population count and no incidence rate, so coverage cannot "
            "honestly be restated as arrests reached per year, and no such figure is offered. "
            "Every coverage number is a share of this modelled demand surface."
        )

    if calibration is not None:
        with st.expander(f"Candidate rules — {outside:.0%} of real deployments fall outside"):
            st.altair_chart(
                viz.simple_bar(calibration, "existing_aeds", "decision", "Existing devices"),
                width="stretch",
            )
            st.caption(
                f"Of the {outside:.0%} outside our pool, {nomap:.0%} sit in buildings "
                "OpenStreetMap does not contain at all — a map-data gap no rule change fixes. "
                "The remainder are refused by our eligibility rules."
            )

    labels = art.get("label_quality")
    if labels is not None:
        with st.expander("Where site names come from"):
            st.dataframe(labels, width="stretch", hide_index=True)
            st.caption("A constructed label says where a building is, not what it is called.")

    with st.expander("Registry data quality"):
        st.write(
            {
                "operating hours": s["hours"],
                "postal cross-validation": s["postal_validation"],
                "co-located devices": s["colocation"],
                "sites needing field verification": s["abstained_sites"],
            }
        )

    sites = art.get("sites")
    if sites is not None:
        st.download_button(
            "Download site reference (id, label, coordinates)",
            sites.to_csv(index=False),
            file_name=f"aed_sites_{s['town']}.csv",
            mime="text/csv",
        )


def main() -> None:
    """Render the app."""
    towns = available_towns()
    if not towns:
        banner()
        st.warning(
            safe_failure(
                "no pipeline artifacts were found. Run "
                "`uv run python -m aed_siting.pipeline --all-towns --seed 42` first."
            )
        )
        return

    with st.sidebar:
        st.markdown("### 1. Pick an area")
        slug = st.selectbox(
            "Town",
            towns,
            index=towns.index(DEFAULT_TOWN) if DEFAULT_TOWN in towns else 0,
            format_func=town_label,
            label_visibility="collapsed",
        )
        art = load_town(slug)
        # Must run before the override multiselects below are instantiated.
        apply_pending_overrides(slug)

        # The budget depends on the queue, which depends on the settings below,
        # so reserve its place at the top and fill it once the queue exists.
        st.markdown("### 2. How many new AEDs can you fund?")
        budget_slot = st.container()

        st.markdown("### 3. Fine-tune (optional)")
        advanced = st.expander("How sites are ranked")
        advanced.caption(
            "The default suits most planners. Change it to see how a different goal "
            "reorders the list."
        )
        method = advanced.selectbox(
            "Ranking method", list(ORDERABLE), format_func=METHOD_LABELS.get, key="method"
        )
        advanced.caption(
            "Three more methods (an exact optimiser, random picks, and doing nothing) "
            "appear as comparisons in Evidence. They give a set, not an ordered list."
        )
        preset = advanced.selectbox("What matters most", ["Custom", *PRESETS], key="preset")
        default_keys = list(PRESETS.get(preset, PRESETS[next(iter(PRESETS))]))
        levels = st.session_state.setdefault("levels", max(len(default_keys), 2))
        # columns([1, 1, 3]) gave each button ~40px, so the labels wrapped one
        # character per line. Equal full-width columns keep them on one line.
        add, drop = advanced.columns(2)
        if add.button("Add level", width="stretch", help="Add a tie-break level") and levels < 4:
            st.session_state["levels"] = levels + 1
            levels += 1
        if drop.button("Drop level", width="stretch", help="Remove a level") and levels > 1:
            st.session_state["levels"] = levels - 1
            levels -= 1

        choices = ["— none —", *KEYS]
        picks: list[str] = []
        for position in range(levels):
            fallback = default_keys[position] if position < len(default_keys) else "— none —"
            if preset != "Custom":
                st.session_state[f"key_{position}"] = fallback
            selection = advanced.selectbox(
                f"Priority {position + 1}",
                choices,
                index=choices.index(fallback),
                format_func=lambda k: KEY_LABELS.get(k, k),
                key=f"key_{position}",
            )
            if selection != "— none —" and selection not in picks:
                picks.append(selection)
        if not picks:
            picks = [default_keys[0]]

        art["_custom_queue"] = art["_custom_steps"] = None
        pinned = tuple(st.session_state.get(f"pins_{slug}", []))
        vetoed = tuple(st.session_state.get(f"vetos_{slug}", []))
        solved = solve_method(slug, method, tuple(picks), 400, pinned, vetoed)
        if solved is None:
            st.warning("Needs solver.npz — re-run the pipeline.")
        else:
            art["_custom_queue"], art["_custom_steps"] = solved

        # Keep the unconstrained solve so the *cost* of each override can be
        # shown at whatever budget the planner lands on. An override is only
        # informed if its price is visible.
        art["_unconstrained"] = None
        if pinned or vetoed:
            free = solve_method(slug, method, tuple(picks), 400)
            art["_unconstrained"] = free[0] if free is not None else None
        art["_override_counts"] = (len(pinned), len(vetoed))

        queue = queue_for(art, preset)
        max_k = max(len(queue) if queue is not None else 0, 1)
        budget = budget_slot.number_input(
            "New AEDs to fund",
            min_value=1,
            max_value=max_k,
            value=min(10, max_k),
            step=1,
        )
        if queue is not None and not queue.empty:
            ceiling = float(queue["cumulative_coverage"].iloc[-1])
            gains = queue["coverage_added_pp"].to_numpy()
            enough = int((gains.cumsum() >= 0.9 * gains.sum()).argmax()) + 1
            adds_coverage = int((gains > 0).sum())
            note = (
                f"Coverage stops improving after **{adds_coverage}** devices "
                f"({ceiling:.1%}); the rest only shorten the average trip."
                if adds_coverage < max_k
                else f"Saturates at **{max_k}** devices ({ceiling:.1%})."
            )
            budget_slot.caption(f"{note} 90% of the gain arrives by **{enough}**.")

        overrides = st.expander("Force a site in or out")
        overrides.caption(
            "Local knowledge the model cannot see: a closure, a landlord, a promise. "
            "You can also click a site on the Plan map."
        )
        candidates = art.get("candidates")
        options = candidates["site_id"].tolist() if candidates is not None else []
        labels = (
            dict(zip(candidates["site_id"], candidates["label"], strict=True))
            if candidates is not None
            else {}
        )
        overrides.multiselect(
            "Must include (pin)",
            options,
            format_func=lambda sid: f"{sid} — {labels.get(sid, '')}",
            key=f"pins_{slug}",
        )
        overrides.multiselect(
            "Must exclude (veto)",
            options,
            format_func=lambda sid: f"{sid} — {labels.get(sid, '')}",
            key=f"vetos_{slug}",
        )

        sim = st.expander("Simulation seed")
        seed_col, roll_col = sim.columns([3, 2])
        seed = seed_col.number_input(
            "Seed",
            min_value=0,
            max_value=99_999,
            value=int(st.session_state.get("seed_value", SEED)),
            step=1,
            label_visibility="collapsed",
        )
        if roll_col.button("Re-roll", width="stretch", help="Draw a new seed"):
            st.session_state["seed_value"] = int(np.random.SeedSequence().entropy % 100_000)
            st.rerun()
        st.session_state["seed_value"] = seed
        sim.caption(
            f"Seed **{seed}** draws the synthetic incidents and the outage trials, "
            "and nothing else. The install queue, the coverage figures and the "
            "robustness sweeps are computed offline and do not move when you "
            "re-roll — see the pipeline seed in the footer. Same seed and same "
            "inputs always reproduce the same result."
        )

        maps = st.expander("Map layers")
        layers_on = {
            "demand": maps.checkbox("Demand and gaps", value=True),
            "existing": maps.checkbox("Covered today", value=True),
            "proposed": maps.checkbox("What the plan adds", value=True),
            "worst": maps.checkbox("Worst-served point", value=False),
        }

        with st.expander("Data, assumptions and limits"):
            for line in DISCLOSURES:
                st.markdown(f"- {line}")
            st.markdown(f"- {OFFICIAL_CHANNELS}")

    s = art["summary"]
    after = (
        float(queue["cumulative_coverage"].iloc[budget - 1])
        if queue is not None and not queue.empty and budget <= len(queue)
        else s["baseline_coverage"]
    )
    st.title("Where should the next AEDs go?")
    st.markdown(
        f"**{s['town_label']}** has **{s['existing_aeds']} public AEDs** (heart "
        f"defibrillators). **{s['baseline_coverage']:.0%}** of people (modelled) are within a "
        f"{s['budget_min']:.0f}-minute round trip on foot of one. Funding **{budget} more** "
        f"in the best places would raise that to **{after:.0%}**. "
        "New here? Open **Start here**."
    )

    intro, existing, plan, test, evidence = st.tabs(
        ["Start here", "Existing AEDs", "Plan new AEDs", "Stress-test", "Evidence"]
    )
    with intro:
        banner()
        screen_intro(art, preset, budget)
    with existing:
        banner()
        screen_existing(art)
    with plan:
        banner()
        screen_plan(art, preset, budget, layers_on)
    with test:
        banner()
        screen_test(art, preset, layers_on, budget, seed)
    with evidence:
        banner()
        screen_evidence(art, preset, budget)

    st.caption(
        f"Artifacts built with pipeline seed {s['seed']} · {s['threshold_m']:.0f} m reach "
        f"· demand proxy `{s['primary_proxy']}` · source data February 2020 "
        "· all figures are modelled planning estimates."
    )


main()
