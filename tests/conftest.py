"""Shared fixtures.

The socket block enforces the brief's supplemental-data rule 3: every judged
result must reproduce from frozen inputs, without a live service. A test that
silently reaches the network would hide a reproducibility failure.
"""

from __future__ import annotations

import socket

import pytest

from aed_siting.config import AED_PATH, PLANNING_AREAS_PATH
from aed_siting.io.aed import load_aed
from aed_siting.io.boundary import load_town_boundary

_ALLOWED = {socket.AF_UNIX} if hasattr(socket, "AF_UNIX") else set()


class _BlockedSocket(socket.socket):
    """Socket that refuses any non-local address family."""

    def __init__(self, family=socket.AF_INET, *args, **kwargs):
        if family not in _ALLOWED:
            raise RuntimeError(
                "Network access is blocked during tests. Every judged result must "
                "reproduce from the frozen inputs in data/raw/."
            )
        super().__init__(family, *args, **kwargs)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, request):
    """Block outbound sockets unless a test is explicitly marked ``network``."""
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setattr(socket, "socket", _BlockedSocket)


@pytest.fixture(scope="session")
def town_artifacts():
    """Pipeline artifacts for the default town.

    Skips rather than fails when the pipeline has not been run, so a fresh clone
    can still execute the unit tests before building anything.
    """
    import json

    import geopandas as gpd
    import pandas as pd

    from aed_siting.config import DEFAULT_TOWN, processed_dir

    directory = processed_dir(DEFAULT_TOWN)
    if not (directory / "run_summary.json").exists():
        pytest.skip("pipeline artifacts absent; run `make pipeline` first")
    out = {"summary": json.loads((directory / "run_summary.json").read_text())}
    for path in directory.glob("*.csv"):
        out[path.stem] = pd.read_csv(path)
    for path in directory.glob("*.geojson"):
        out[path.stem] = gpd.read_file(path)
    return out


@pytest.fixture(scope="session")
def boundary():
    """Toa Payoh planning-area polygon, EPSG:3414."""
    return load_town_boundary(PLANNING_AREAS_PATH)


@pytest.fixture(scope="session")
def aed_town(boundary):
    """AED registry clipped to the study area, plus its load report."""
    return load_aed(AED_PATH, boundary=boundary)
