"""Pinned constants for the whole pipeline.

Everything that could silently change an answer lives here: coordinate systems,
the retrieval-time budget the coverage threshold is derived from, the detour
factor, the seed, and the dataset ids. Nothing downstream hard-codes these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DIR: Final[Path] = ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = ROOT / "data" / "processed"
MANIFEST_PATH: Final[Path] = ROOT / "data" / "manifest.json"

# --- Coordinate systems ----------------------------------------------------
# Distances are metres in SVY21. Nothing is ever measured in degrees.
CRS_WGS84: Final[int] = 4326
CRS_SVY21: Final[int] = 3414

# --- Study areas -----------------------------------------------------------
# Four planning areas spanning the full coverage range measured across 15 areas,
# so the town selector demonstrates transfer rather than re-skinning one result.
# Coverage figures are the uniform-demand first pass used to choose them.
TOWNS: Final[dict[str, dict[str, str]]] = {
    "toa-payoh": {"area": "TOA PAYOH", "prefix": "TP", "label": "Toa Payoh"},
    "ang-mo-kio": {"area": "ANG MO KIO", "prefix": "AMK", "label": "Ang Mo Kio"},
    "bedok": {"area": "BEDOK", "prefix": "BD", "label": "Bedok"},
    "jurong-east": {"area": "JURONG EAST", "prefix": "JE", "label": "Jurong East"},
}
DEFAULT_TOWN: Final[str] = "toa-payoh"

#: Backwards-compatible alias; the planning-area name of the default town.
TOWN: Final[str] = TOWNS[DEFAULT_TOWN]["area"]

H3_RESOLUTION: Final[int] = 10  # ~0.0150 km^2/hex -> ~545 hexes over 8.18 km^2


def town_area(slug: str = DEFAULT_TOWN) -> str:
    """Planning-area name (``PLN_AREA_N``) for a town slug."""
    return TOWNS[slug]["area"]


def town_prefix(slug: str = DEFAULT_TOWN) -> str:
    """Short prefix used to build stable site ids, e.g. ``TP`` -> ``TP-041``."""
    return TOWNS[slug]["prefix"]


def town_label(slug: str = DEFAULT_TOWN) -> str:
    """Human-readable town name for the UI."""
    return TOWNS[slug]["label"]


def osm_path(slug: str = DEFAULT_TOWN) -> Path:
    """Frozen OSM extract for one town."""
    return RAW_DIR / f"osm_{slug}.parquet"


def processed_dir(slug: str = DEFAULT_TOWN) -> Path:
    """Per-town artifact directory, so towns never overwrite each other."""
    return PROCESSED_DIR / slug


# --- Distance model --------------------------------------------------------
# The stored matrix is ONE-WAY network-estimate metres. DetourEuclidean produces
# them as euclid * DETOUR_FACTOR; a walking-network model would produce true
# shortest-path metres. The threshold is expressed in the same unit and is never
# rescaled -- that is what keeps the two implementations interchangeable.
DETOUR_FACTOR: Final[float] = 1.35
WALK_SPEED_MS: Final[float] = 1.4

# Round-trip retrieval budget. Pre-registered headline is 3 minutes; the sweep is
# reported in full regardless of which value flatters the result.
RETRIEVAL_BUDGET_MIN: Final[float] = 3.0
RETRIEVAL_BUDGET_SWEEP_MIN: Final[tuple[float, ...]] = (2.0, 3.0, 4.0)

# Lift-and-return allowance for an AED that is not at ground level. Applied to
# EXISTING devices only: a new install is specified at ground floor, so candidate
# sites carry zero penalty. This asymmetry flatters new installs and is disclosed
# in the method card rather than left to be discovered.
FLOOR_PENALTY_SECONDS: Final[float] = 45.0

# --- Determinism -----------------------------------------------------------
SEED: Final[int] = 42

# --- Source register -------------------------------------------------------
# Verified against the live dataset pages. The AED layer has near-duplicate
# siblings on data.gov.sg; the 16-field schema check is the discriminator.
DATASET_IDS: Final[dict[str, str]] = {
    "aed": "d_4e6b82c58a8a832f6f1fee5dfa6d47ea",
    "planning_areas": "d_4765db0e87b9c86336792efe8a1f7a66",
    "subzones": "d_8594ae9ff96d0c708bc2af633048edfb",
}

AED_PATH: Final[Path] = RAW_DIR / "aed_public_access.geojson"
PLANNING_AREAS_PATH: Final[Path] = RAW_DIR / "planning_areas_no_sea.geojson"
SUBZONES_PATH: Final[Path] = RAW_DIR / "subzones_no_sea.geojson"
OSM_PATH: Final[Path] = RAW_DIR / f"osm_{DEFAULT_TOWN}.parquet"

# The 16 published attributes named in the challenge brief.
AED_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "OBJECTID",
    "AED_ID",
    "OPERATING_HOURS",
    "HOUSE_NUMBER",
    "ROAD_NAME",
    "BUILDING_NAME",
    "UNIT_NUMBER",
    "POSTAL_CODE",
    "AED_LOCATION_DESCRIPTION",
    "AED_LOCATION_FLOOR_LEVEL",
    "LATITUDE",
    "LONGITUDE",
    "XVAL",
    "YVAL",
    "INC_CRC",
    "FMEL_UPD_D",
)

# Fields with no authoritative data dictionary. The brief forbids assigning them
# operational meaning, so the pipeline must never consume them. XVAL/YVAL were
# verified to be SVY21 (median offset 0.083 m) but that is recorded as a finding
# only -- all geometry is derived from LATITUDE/LONGITUDE.
BANNED_FIELDS: Final[frozenset[str]] = frozenset({"XVAL", "YVAL", "INC_CRC", "FMEL_UPD_D"})

# Address-like fields that must never be parsed as numbers.
ADDRESS_FIELDS: Final[tuple[str, ...]] = ("POSTAL_CODE", "HOUSE_NUMBER", "UNIT_NUMBER")
POSTAL_CODE_LENGTH: Final[int] = 6

# Free-text field belonging to Lane 1. Loaded for provenance, never parsed.
LANE1_ONLY_FIELDS: Final[frozenset[str]] = frozenset({"AED_LOCATION_DESCRIPTION"})


def threshold_m(budget_min: float = RETRIEVAL_BUDGET_MIN) -> float:
    """One-way network metres reachable within a round-trip retrieval budget.

    Args:
        budget_min: Round-trip retrieval budget in minutes.

    Returns:
        One-way distance in metres, directly comparable to the distance matrix.

    Assumptions:
        Travel time only, at ``WALK_SPEED_MS``. Recognition, cabinet unlocking and
        return-to-patient handling are excluded and documented as excluded.
        The detour factor is NOT applied here -- it already lives in the matrix,
        and applying it twice would tighten the threshold by 35%.
    """
    return (budget_min * 60.0 * WALK_SPEED_MS) / 2.0


def floor_penalty_m(seconds: float = FLOOR_PENALTY_SECONDS) -> float:
    """Convert the lift-and-return allowance into matrix units (metres)."""
    return seconds * WALK_SPEED_MS
