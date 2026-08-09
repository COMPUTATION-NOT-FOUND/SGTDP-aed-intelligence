"""Distance models.

One contract, two implementations. The stored matrix is **one-way
network-estimate metres**: ``DetourEuclidean`` produces them by scaling
straight-line separation, and a walking-network model would produce true
shortest-path metres. Because both speak the same unit, the coverage threshold
never changes when one is swapped for the other.

Round-trip is accounted for in exactly one place: the ``/ 2`` inside
:func:`aed_siting.config.threshold_m`, which converts a round-trip time budget
into the one-way distance it buys. Neither this module nor
:mod:`aed_siting.metrics` doubles anything -- doing so as well would halve the
effective radius.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from aed_siting.config import CRS_SVY21, DETOUR_FACTOR

UNIT = "network_metres_oneway"


@runtime_checkable
class DistanceModel(Protocol):
    """One-way distance between demand cells and sites."""

    unit: ClassVar[str]

    def matrix(self, origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame) -> np.ndarray:
        """Return shape ``(len(origins), len(destinations))`` in metres, one-way."""
        ...


def _require_svy21(frame: gpd.GeoDataFrame, label: str) -> np.ndarray:
    """Extract coordinates, refusing anything not already projected to metres."""
    if frame.crs is None or frame.crs.to_epsg() != CRS_SVY21:
        raise ValueError(
            f"{label} must be EPSG:{CRS_SVY21} (metres); got {frame.crs}. "
            "Distances are never computed in degrees."
        )
    return np.c_[frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()]


class DetourEuclidean:
    """Straight-line separation scaled by a detour factor.

    The shipped model. A detour factor converts straight-line distance into an
    estimate of walked network distance, which is what the threshold is expressed
    in.

    Assumptions:
        A single global factor stands in for street-network circuity. 1.35 is a
        conventional value for dense urban grids; it is swept in the sensitivity
        analysis rather than treated as known.

    Limits:
        Cannot see barriers. A canal, expressway or fenced boundary between two
        points is invisible to it, so this model is optimistic exactly where the
        walking network would be most pessimistic. That is the single largest
        modelled-versus-real gap in the pipeline and is the reason the
        walking-network implementation exists as an upgrade path.
    """

    unit: ClassVar[str] = UNIT

    def __init__(self, factor: float = DETOUR_FACTOR) -> None:
        """Store the detour factor, refusing values that beat a straight line."""
        if factor < 1.0:
            raise ValueError("detour factor must be >= 1.0; a route cannot beat a straight line")
        self.factor = factor

    def matrix(self, origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame) -> np.ndarray:
        """Return one-way network-estimate metres, shape (n_origins, n_destinations)."""
        a = _require_svy21(origins, "origins")
        b = _require_svy21(destinations, "destinations")
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=float)
        straight = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return straight * self.factor


def nearest_distance(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    model: DistanceModel | None = None,
    penalties_m: np.ndarray | None = None,
) -> np.ndarray:
    """One-way distance from each origin to its nearest destination.

    Args:
        origins: Demand cells, EPSG:3414.
        destinations: Sites, EPSG:3414.
        model: Distance model; defaults to :class:`DetourEuclidean`.
        penalties_m: Optional per-destination retrieval penalty in metres, added
            before taking the minimum so a penalised site can lose to a further
            but unpenalised one.

    Returns:
        Array of length ``len(origins)``; ``inf`` where there are no destinations.

    Limits:
        One-way. Round-trip is applied by the caller, once.
    """
    if len(destinations) == 0:
        return np.full(len(origins), np.inf)
    model = model or DetourEuclidean()
    dist = model.matrix(origins, destinations)
    if penalties_m is not None:
        dist = dist + np.asarray(penalties_m, dtype=float)[None, :]
    return dist.min(axis=1)


def within_radius_pairs(
    origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, radius_m: float
) -> list[tuple[int, int]]:
    """Origin/destination index pairs closer than ``radius_m`` straight-line.

    A blocking helper for the optimisers: coverage sets only ever need pairs
    inside the threshold, and a KD-tree finds them without materialising the full
    matrix. The radius is *straight-line*, so callers must pre-divide a
    network-metre threshold by the detour factor to avoid discarding pairs the
    full model would have kept.
    """
    a = _require_svy21(origins, "origins")
    b = _require_svy21(destinations, "destinations")
    tree = cKDTree(b)
    return [
        (i, j)
        for i, neighbours in enumerate(tree.query_ball_point(a, r=radius_m))
        for j in neighbours
    ]
