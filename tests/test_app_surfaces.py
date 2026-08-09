"""Gates on the user-facing surface.

The safety gate is pass/fail before scoring, so these tests treat the banner and
the point-of-use disclosures as correctness requirements, not presentation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from aed_siting.config import PROCESSED_DIR, ROOT
from aed_siting.safety import DISCLOSURES, safe_failure

APP = ROOT / "app.py"


@pytest.fixture(scope="module")
def app_source() -> str:
    assert APP.exists(), "app.py is missing"
    return APP.read_text()


@pytest.fixture(scope="module")
def app_tree(app_source: str) -> ast.Module:
    return ast.parse(app_source)


@pytest.mark.gate
def test_banner_on_every_screen(app_tree, app_source):
    """Every view function must render the banner, not just the entry point.

    Enforced structurally: ``main`` calls ``banner()`` once before anything else
    and again inside each screen's tab context, so a new screen that forgets it
    is visible as a count mismatch.
    """
    screens = [
        node.name
        for node in ast.walk(app_tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("screen_")
    ]
    assert screens, "no screen functions found"
    assert app_source.count("banner()") >= len(screens) + 2, (
        f"{len(screens)} screens but too few banner() calls"
    )
    assert app_source.count("banner()") >= 4, "banner is not rendered on every screen"
    assert "def banner" in app_source
    assert "st.error(BANNER)" in app_source


@pytest.mark.gate
def test_app_uses_the_verbatim_banner_constant(app_source):
    """The app must import the constant, never re-type the wording."""
    assert "from aed_siting.safety import" in app_source
    assert "BANNER" in app_source
    # A paraphrase would appear as a literal string containing the key phrase.
    assert "not for emergency use" not in app_source.replace("BANNER", "")


@pytest.mark.gate
def test_point_of_use_disclosures(app_source):
    """Dataset date, assumptions, uncertainty and failure modes must appear in the UI."""
    assert "DISCLOSURES" in app_source
    assert "OFFICIAL_CHANNELS" in app_source
    joined = " ".join(DISCLOSURES)
    assert "February 2020" in joined
    assert "not proof" in joined


@pytest.mark.gate
def test_failure_states_are_safe(app_source):
    """Missing or empty artifacts must abstain, never render a blank or crash."""
    assert app_source.count("safe_failure(") >= 4, "not every view has a safe failure state"
    message = safe_failure("operating hours could not be parsed.")
    assert "No recommendation" in message and "abstains" in message


@pytest.mark.gate
def test_app_reads_only_frozen_artifacts(app_source):
    """No network access path may exist from the UI."""
    for forbidden in ("requests", "urllib", "osmnx", "overpass", "http://", "https://api"):
        assert forbidden not in app_source, f"app.py references {forbidden!r}"
    assert "PROCESSED_DIR" in app_source


@pytest.mark.gate
def test_no_nearest_aed_lookup(app_tree):
    """The Lane 1 entry point is deliberately absent.

    The explainability panel answers "why is this cell covered?"; a free-text or
    coordinate box answering "where is my nearest AED?" would be live emergency
    guidance in all but name, which the safety gate forbids.

    Checked against the parsed call graph rather than the raw text, so prose
    *describing* the omission does not trip the gate.
    """
    # Free-text widgets could accept an address; a bounded numeric input for a
    # device count could not. The gate targets location entry, which is the
    # actual safety concern, rather than banning every input on principle.
    forbidden_widgets = {"text_input", "text_area", "chat_input"}
    calls = [
        node.func.attr
        for node in ast.walk(app_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    ]
    offending = forbidden_widgets.intersection(calls)
    assert not offending, f"app.py exposes a free-entry lookup surface: {sorted(offending)}"

    # No *widget* may ask for a place. Checked against widget labels only:
    # scanning every string constant instead flagged the prose explaining that
    # site names are built from an address, which is documentation, not a prompt.
    widgets = {"number_input", "selectbox", "radio", "slider", "multiselect"}
    location_words = ("address", "postcode", "postal", "latitude", "longitude", "coordinate")
    for node in ast.walk(app_tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in widgets
        ):
            continue
        labels = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        labels += [
            kw.value.value
            for kw in node.keywords
            if kw.arg == "label" and isinstance(kw.value, ast.Constant)
        ]
        for label in labels:
            for word in location_words:
                assert word not in label.lower(), (
                    f"widget {node.func.attr} prompts for a location: {label!r}"
                )


@pytest.mark.gate
def test_all_towns_produce_artifacts():
    """Every town offered in the selector must have a complete artifact set."""
    from aed_siting.config import TOWNS, processed_dir

    built = [slug for slug in TOWNS if (processed_dir(slug) / "run_summary.json").exists()]
    if not built:
        pytest.skip("pipeline has not been run in this checkout")
    for slug in built:
        directory = processed_dir(slug)
        summary = json.loads((directory / "run_summary.json").read_text())
        for name in (
            "install_queue.csv",
            "sites.csv",
            "method_vs_baseline.csv",
            "objectives.csv",
            "site_stability.csv",
            "nesting_loss.csv",
            "equity_before.csv",
            "equity_after.csv",
            "incidents_before.csv",
            "incidents_after.csv",
            "outage_stress.csv",
            "cells.csv",
        ):
            assert (directory / name).exists(), f"{slug}: missing artifact {name}"
        assert summary["existing_aeds"] > 0
        assert summary["candidates"] > 0
        assert 0.0 <= summary["baseline_coverage"] <= 1.0
        assert summary["town"] == slug


@pytest.mark.gate
def test_recommendations_never_claim_availability(app_source):
    """No UI string may assert a device is available, working, or inspected."""
    lowered = app_source.lower()
    for phrase in ("currently available", "is working", "verified available", "open now"):
        assert phrase not in lowered, f"availability claim in UI: {phrase!r}"


@pytest.mark.gate
def test_only_orderable_methods_drive_the_queue():
    """Methods without a meaningful order must not be offered as install queues.

    An ILP yields an optimal *set* per budget with no internal order, random-k is
    a distribution and do-nothing is empty. Presenting any of them with a `rank`
    column would invent an ordering the method never produced.
    """
    import app

    assert set(app.ORDERABLE) == {
        "Marginal-coverage greedy",
        "Demand-ranked top-k (baseline)",
        "Greedy k-center (equity)",
    }
    for name in app.COMPARISON_ONLY:
        assert name not in app.ORDERABLE, f"{name} must not drive the queue"


@pytest.mark.gate
def test_app_binds_to_this_checkouts_package():
    """`app.py` must import the `aed_siting` sitting beside it, not an installed copy.

    Regression gate: the deployed app kept a stale ``site-packages`` copy from an
    earlier build. It imported cleanly, so the old path guard -- which only acted
    when the import *failed* -- did nothing, and a freshly-pulled ``app.py`` called
    into last build's module. ``demand_hexes()`` received four arguments, and the
    live app died with a ``TypeError`` that could never reproduce locally, where
    the editable install always matched. The guard now prefers ``src/``
    unconditionally so the two cannot drift apart.
    """
    import app  # noqa: F401

    import aed_siting

    resolved = Path(aed_siting.__file__).resolve()
    assert resolved.is_relative_to((ROOT / "src").resolve()), (
        f"app.py bound to {resolved}, not this checkout's src/ -- version skew is possible"
    )
    source = APP.read_text()
    assert "sys.path.insert(0, str(_SRC))" in source, "the src/ path guard was removed"
    assert "except ModuleNotFoundError" not in source.split("from aed_siting")[0], (
        "the guard is conditional again; a stale installed copy would win"
    )


@pytest.mark.gate
@pytest.mark.slow
def test_map_click_override_reaches_the_solver():
    """A map click must land in the override widgets without Streamlit raising.

    The override multiselects live in the sidebar, which renders *before* the
    tabs holding the map. Writing to a widget's session-state key after that
    widget exists raises ``StreamlitAPIException``, so the click records an
    intent and ``apply_pending_overrides`` folds it in at the top of the next
    run. That ordering is invisible from the code and fails only at runtime,
    which is exactly what this gate covers.
    """
    from streamlit.testing.v1 import AppTest

    from aed_siting.config import DEFAULT_TOWN, processed_dir

    if not (processed_dir(DEFAULT_TOWN) / "run_summary.json").exists():
        pytest.skip("pipeline has not been run in this checkout")

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.session_state[f"_pending_pins_{DEFAULT_TOWN}"] = "TP-006"
    at.session_state[f"_pending_vetos_{DEFAULT_TOWN}"] = "TP-051"
    at.run()

    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    assert "TP-006" in at.session_state[f"pins_{DEFAULT_TOWN}"], "pinned site never reached state"
    assert "TP-051" in at.session_state[f"vetos_{DEFAULT_TOWN}"], "vetoed site never reached state"
    # The intent is consumed, so a later rerun cannot re-apply it in a loop.
    assert f"_pending_pins_{DEFAULT_TOWN}" not in at.session_state

    # An override the planner cannot price is not an informed override, so the
    # cost of the constrained solve has to reach the screen, not just the state.
    warnings = " ".join(w.value for w in at.warning)
    assert "1 pinned, 1 vetoed" in warnings, f"override cost never surfaced; saw: {warnings[:200]}"


def test_processed_artifacts_ship_with_the_repo():
    """Derived artifacts are committed so a reviewer can run the tool on clone.

    This reverses an earlier rule that ignored ``data/processed/``. The app reads
    *only* pre-computed artifacts, so ignoring them meant a fresh clone rendered
    the empty state until a ~25-minute pipeline run finished -- a reviewer with
    limited time would reasonably conclude the tool did not work. They remain
    fully regenerable from the checksummed raw inputs; committing them trades
    repository tidiness for the thing that actually matters here.
    """
    gitignore = (ROOT / ".gitignore").read_text()
    ignored = [
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "data/processed/" not in ignored, "processed artifacts must ship with the repo"
    assert Path(PROCESSED_DIR).name == "processed"
