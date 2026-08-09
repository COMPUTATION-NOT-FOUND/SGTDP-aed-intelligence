"""Registry data-quality checks.

Borrowed from challenge Lane 3 (registry and readiness), kept only where it
serves the Lane 2 decision.

What this module deliberately does *not* do is fuzzy near-duplicate detection.
That was built first and then removed, because the registry turns out to carry an
authoritative device key: ``AED_ID`` is unique across all 9,644 records, and its
prefix is the six-digit postal code. Two records with different ``AED_ID`` values
are different registered devices, however close together they sit. A proximity-
and-name matcher over this data does not find duplicates -- it finds schools and
community clubs that legitimately host four to six devices, and "collapsing"
those would delete real AEDs from the model.

What remains are two checks that are real:

* cross-field validation of ``POSTAL_CODE`` against the ``AED_ID`` prefix, which
  turns the 456 zero-pad repairs from an inference into a verified reconstruction;
* co-located-site detection, which feeds the Lane 2 catchment-load and
  redundancy analysis rather than any notion of data defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from aed_siting.config import POSTAL_CODE_LENGTH


@dataclass(frozen=True)
class PostalValidation:
    """Cross-field agreement between POSTAL_CODE and the AED_ID prefix."""

    records: int
    agreeing: int
    repaired_and_confirmed: int
    mismatches: pd.DataFrame

    @property
    def agreement_rate(self) -> float:
        """Share of records whose two independent postal fields agree."""
        return self.agreeing / self.records if self.records else 0.0

    def summary(self) -> str:
        """One-line summary for the tracker and the data-quality doc."""
        return (
            f"{self.agreeing}/{self.records} records ({self.agreement_rate:.2%}) agree "
            f"between POSTAL_CODE and the AED_ID prefix; {self.repaired_and_confirmed} "
            f"zero-pad repairs independently confirmed; {len(self.mismatches)} unresolved."
        )


def validate_postal_codes(frame: gpd.GeoDataFrame | pd.DataFrame) -> PostalValidation:
    """Check POSTAL_CODE against the postal code embedded in AED_ID.

    ``AED_ID`` is formatted ``<6-digit postal code>-<sequence>`` (e.g.
    ``098136-004``), giving a second, independent record of the postal code in
    every row. Where ``POSTAL_CODE`` lost its leading zero to numeric
    serialisation upstream, the ``AED_ID`` prefix did not.

    Args:
        frame: AED records after loading, with POSTAL_CODE already repaired.

    Returns:
        A :class:`PostalValidation` carrying the agreement rate, the number of
        repairs independently confirmed, and any unresolved mismatches.

    Assumptions:
        The ``AED_ID`` prefix is a postal code. This is inferred from format and
        then *tested* here rather than asserted — the agreement rate is the
        evidence, and a low rate would falsify the assumption rather than pass
        silently.

    Limits:
        Agreement shows the registry is internally consistent. It says nothing
        about whether either field is correct in the real world, and nothing at
        all about whether a device is present or working.
    """
    postal = frame["POSTAL_CODE"].astype("string").str.strip()
    prefix = frame["AED_ID"].astype("string").str.split("-").str[0].str.strip()
    padded = postal.str.zfill(POSTAL_CODE_LENGTH)

    agrees = padded == prefix
    was_short = postal.str.len() < POSTAL_CODE_LENGTH

    mismatches = pd.DataFrame(
        {
            "AED_ID": frame["AED_ID"][~agrees],
            "POSTAL_CODE": postal[~agrees],
            "id_prefix": prefix[~agrees],
            "BUILDING_NAME": frame["BUILDING_NAME"][~agrees],
        }
    ).reset_index(drop=True)

    return PostalValidation(
        records=len(frame),
        agreeing=int(agrees.sum()),
        repaired_and_confirmed=int((was_short & agrees).sum()),
        mismatches=mismatches,
    )


@dataclass(frozen=True)
class ColocationReport:
    """Sites hosting more than one registered device."""

    sites: pd.DataFrame
    n_records: int

    @property
    def n_multi_device_sites(self) -> int:
        """Distinct coordinates carrying two or more devices."""
        return len(self.sites)

    @property
    def n_records_colocated(self) -> int:
        """Records sitting at a shared coordinate."""
        return int(self.sites["devices"].sum()) if not self.sites.empty else 0

    def summary(self) -> str:
        """One-line summary for the tracker and the data-quality doc."""
        return (
            f"{self.n_multi_device_sites} coordinates host 2+ devices, covering "
            f"{self.n_records_colocated} of {self.n_records} records; "
            f"largest site holds {int(self.sites['devices'].max()) if not self.sites.empty else 0}."
        )


def find_colocated_sites(frame: gpd.GeoDataFrame) -> ColocationReport:
    """Group records that share an exact coordinate.

    Args:
        frame: AED records in EPSG:3414.

    Returns:
        A :class:`ColocationReport` listing each shared coordinate, how many
        devices it carries, and the building name.

    Assumptions:
        Shared coordinates arise from geocoding several devices in one building
        to a single centroid, not from duplicate data entry — established by the
        uniqueness of ``AED_ID`` and confirmed by floor levels differing within a
        group.

    Limits:
        Co-location is a *modelling* fact, not a defect. Two devices at one
        coordinate deliver the same modelled coverage as one, so coverage is
        unaffected; device counts, catchment load and redundancy are not. Nothing
        here should be reported as a data-quality problem.
    """
    working = frame.copy()
    working["_wkb"] = working.geometry.to_wkb()

    grouped = (
        working.groupby("_wkb")
        .agg(
            devices=("AED_ID", "size"),
            building=("BUILDING_NAME", "first"),
            floors=("AED_LOCATION_FLOOR_LEVEL", lambda s: sorted({str(v) for v in s})),
        )
        .reset_index(drop=True)
    )
    sites = (
        grouped[grouped["devices"] > 1]
        .sort_values("devices", ascending=False)
        .reset_index(drop=True)
    )
    return ColocationReport(sites=sites, n_records=len(frame))
