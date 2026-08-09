"""Loading, repairing and clipping the SCDF Public Access AED registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from aed_siting.config import (
    ADDRESS_FIELDS,
    AED_PATH,
    AED_REQUIRED_FIELDS,
    BANNED_FIELDS,
    CRS_SVY21,
    CRS_WGS84,
    POSTAL_CODE_LENGTH,
    floor_penalty_m,
)

_GROUND_LEVEL = "1"
_NULLISH = {"", "nan", "none", "null", "<na>"}


@dataclass
class LoadReport:
    """Everything the loader changed or discarded, for the manifest and method card.

    Nothing here is cosmetic: each count is a transformation that a reviewer is
    entitled to see, and several are reported as metrics in their own right.
    """

    records_in: int = 0
    postal_codes_zero_padded: int = 0
    house_number_nulls: int = 0
    unit_number_nulls: int = 0
    dropped_missing_geometry: int = 0
    records_in_town: int = 0
    dropped_duplicate_ids: int = 0
    colocated_records: int = 0
    floor_above_ground: int = 0
    floor_unparseable: int = 0
    notes: list[str] = field(default_factory=list)

    def as_transformations(self) -> list[str]:
        """Render the report as manifest-ready transformation strings."""
        return [
            f"Read {', '.join(ADDRESS_FIELDS)} as text, never numeric.",
            f"Left-padded {self.postal_codes_zero_padded} POSTAL_CODE values to "
            f"{POSTAL_CODE_LENGTH} characters. The leading zero was destroyed by "
            "numeric serialisation before publication, so this is an inferred "
            "repair, not a recovery.",
            "Rebuilt point geometry from LATITUDE/LONGITUDE only; XVAL/YVAL, "
            "INC_CRC and FMEL_UPD_D are dropped and never consumed.",
            f"Dropped {self.dropped_missing_geometry} records with unusable coordinates.",
            f"Clipped to the study area: {self.records_in_town} of {self.records_in} retained.",
            f"Dropped {self.dropped_duplicate_ids} records with a repeated AED_ID "
            "(the authoritative device key).",
            f"Retained {self.colocated_records} records that share a coordinate with "
            "another device: these are distinct registered devices geocoded to one "
            "building centroid, not duplicates.",
            f"Flagged {self.floor_unparseable} records whose floor level could not "
            "be parsed; these carry the retrieval penalty conservatively.",
        ]


def parse_floor_penalty(value: object) -> tuple[float, bool]:
    """Convert a stated floor level into a retrieval penalty in metres.

    Args:
        value: Raw ``AED_LOCATION_FLOOR_LEVEL`` cell.

    Returns:
        ``(penalty_metres, parsed_ok)``. Level 1 is ground in Singapore and
        carries no penalty. Upper floors and basements both carry the full
        lift-and-return allowance.

    Assumptions:
        A single flat allowance, not a per-storey cost: the dominant term is
        waiting for a lift, which does not scale with height.

    Limits:
        Applies to EXISTING devices only. Candidate sites carry zero penalty
        because a new install is specified at ground level — an asymmetry that
        flatters new installs and is disclosed in the method card.
        Unparseable values are penalised conservatively and flagged, never guessed.
    """
    text = str(value).strip().upper()
    if text.lower() in _NULLISH:
        return floor_penalty_m(), False
    if text == _GROUND_LEVEL:
        return 0.0, True
    if re.fullmatch(r"B\d+", text) or re.fullmatch(r"\d+", text):
        return floor_penalty_m(), True
    return floor_penalty_m(), False


def _clean_address_fields(frame: pd.DataFrame, report: LoadReport) -> pd.DataFrame:
    """Read address-like fields as text and repair lost leading zeros."""
    for column in ADDRESS_FIELDS:
        frame[column] = (
            frame[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        )
        frame.loc[frame[column].str.lower().isin(_NULLISH), column] = pd.NA

    postal = frame["POSTAL_CODE"]
    short = postal.notna() & (postal.str.len() < POSTAL_CODE_LENGTH)
    report.postal_codes_zero_padded = int(short.sum())
    frame.loc[short, "POSTAL_CODE"] = postal[short].str.zfill(POSTAL_CODE_LENGTH)

    report.house_number_nulls = int(frame["HOUSE_NUMBER"].isna().sum())
    report.unit_number_nulls = int(frame["UNIT_NUMBER"].isna().sum())
    return frame


def load_aed(
    path=AED_PATH,
    boundary: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, LoadReport]:
    """Load the AED registry, repair it, and optionally clip it to a study area.

    Args:
        path: Frozen ``Public Access AEDs`` GeoJSON.
        boundary: Single-row study-area frame in EPSG:3414. If omitted, all of
            Singapore is returned.

    Returns:
        ``(frame, report)`` where ``frame`` is in EPSG:3414 with a
        ``floor_penalty_m`` column, and ``report`` records every transformation.

    Assumptions:
        Geometry is derived from LATITUDE/LONGITUDE only. XVAL/YVAL were verified
        to be SVY21 (median offset 0.083 m over 3,000 records) but the brief
        forbids assigning operational meaning to undocumented fields, so they are
        dropped rather than trusted.

    Limits:
        A record here is not evidence that a device is present, accessible,
        inspected, or working. This is a February 2020 registry snapshot.

    Raises:
        ValueError: If any of the brief's 16 published attributes is absent.
    """
    raw = gpd.read_file(path)
    report = LoadReport(records_in=len(raw))

    missing = [c for c in AED_REQUIRED_FIELDS if c not in raw.columns]
    if missing:
        raise ValueError(f"AED layer is missing published attributes: {missing}")

    frame = _clean_address_fields(pd.DataFrame(raw.drop(columns="geometry")), report)

    lat = pd.to_numeric(frame["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
    usable = lat.notna() & lon.notna()
    report.dropped_missing_geometry = int((~usable).sum())
    frame, lat, lon = frame[usable], lat[usable], lon[usable]

    penalties = frame["AED_LOCATION_FLOOR_LEVEL"].map(parse_floor_penalty)
    frame["floor_penalty_m"] = [p for p, _ in penalties]
    frame["floor_parsed"] = [ok for _, ok in penalties]

    # Banned fields are physically removed so nothing downstream can consume them.
    frame = frame.drop(columns=[c for c in BANNED_FIELDS if c in frame.columns])

    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(lon, lat),
        crs=CRS_WGS84,
    ).to_crs(CRS_SVY21)

    if boundary is not None:
        polygon = boundary.to_crs(CRS_SVY21).union_all()
        gdf = gdf[gdf.geometry.within(polygon)]
    report.records_in_town = len(gdf)

    # AED_ID is unique across all 9,644 registry records and is the authoritative
    # device key, so it -- not geometry -- decides identity. Records sharing a
    # coordinate are distinct devices geocoded to one building centroid (e.g.
    # 310480-002 on level 2 and 310480-003 on level 1 of the same block).
    # Dropping those would delete real devices, understate the existing network,
    # and inflate the apparent value of every recommended placement.
    before = len(gdf)
    gdf = gdf[~gdf["AED_ID"].duplicated()]
    report.dropped_duplicate_ids = before - len(gdf)
    report.colocated_records = int(gdf.geometry.duplicated(keep=False).sum())

    # Floor counts are taken AFTER clipping and dedup so they describe the devices
    # the model actually uses, not the all-Singapore file.
    report.floor_unparseable = int((~gdf["floor_parsed"]).sum())
    report.floor_above_ground = int((gdf["floor_penalty_m"] > 0).sum())

    return gdf.reset_index(drop=True), report
