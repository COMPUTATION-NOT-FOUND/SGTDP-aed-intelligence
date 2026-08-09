"""Deterministic bucketing of the free-text OPERATING_HOURS field.

Operating-hours normalisation is challenge Lane 1's named technique; it is used
here for the reason Lane 2 names, "hours of access" as a placement constraint. A
device that is locked inside a school at 03:00 does not cover its neighbourhood
at 03:00.

The field is free text, but in practice it is highly regular, so this is a rule
parser rather than an NLP model. Anything the rules do not recognise is
**excluded and flagged**, never guessed — the parse rate is reported as the
submission's safety/error metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Bucket = Literal["always_open", "extended", "daytime", "closed", "unparseable"]

#: A device is treated as always available only if it states cover for the whole
#: day, every day. 23:59 is how this registry writes midnight.
_ALWAYS = re.compile(r"mon\s*-\s*sun\s+00:00\s*-\s*23:5[89]", re.IGNORECASE)
_CLOSED_ALL = re.compile(r"mon\s*-\s*sun\s+closed", re.IGNORECASE)
_SPAN = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")

#: Daily open hours at or above this are "extended" rather than "daytime".
_EXTENDED_HOURS = 14.0


@dataclass(frozen=True)
class HoursReport:
    """Parse outcome across a set of records."""

    counts: pd.Series
    total: int

    @property
    def parse_rate(self) -> float:
        """Share of records the rules resolved to a usable bucket."""
        if not self.total:
            return 0.0
        return 1.0 - (int(self.counts.get("unparseable", 0)) / self.total)

    @property
    def always_open(self) -> int:
        """Devices stating 24/7 cover."""
        return int(self.counts.get("always_open", 0))

    def summary(self) -> str:
        """One-line summary for the tracker and the data-quality doc."""
        return (
            f"{self.total} records: parse rate {self.parse_rate:.1%}, "
            f"{self.always_open} stated 24/7, "
            f"{int(self.counts.get('unparseable', 0))} unparseable (excluded, flagged)."
        )


def bucket_hours(value: object) -> Bucket:
    """Classify one OPERATING_HOURS string.

    Args:
        value: Raw cell contents.

    Returns:
        One of ``always_open``, ``extended``, ``daytime``, ``closed`` or
        ``unparseable``.

    Assumptions:
        ``Mon - Sun 00:00-23:59`` denotes continuous availability. Where several
        day-ranges are stated, the **longest** daily span decides the bucket, so a
        site open late on one day is not credited with that cover all week —
        the bucket is then used conservatively by the 24/7 scenario, which admits
        only ``always_open``.

    Limits:
        These are *stated* hours from a February 2020 snapshot, not verified
        present access. Nothing here may be rendered as "open now" or
        "available". Unparseable input returns ``unparseable`` so the caller can
        exclude and flag it; it is never coerced into a usable bucket.
    """
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return "unparseable"
    if _ALWAYS.search(text):
        return "always_open"
    if _CLOSED_ALL.search(text):
        return "closed"

    spans = _SPAN.findall(text)
    if not spans:
        return "unparseable"

    longest = 0.0
    for start_h, start_m, end_h, end_m in spans:
        start = int(start_h) + int(start_m) / 60
        end = int(end_h) + int(end_m) / 60
        if end < start:  # crosses midnight
            end += 24
        longest = max(longest, end - start)

    if longest <= 0:
        return "unparseable"
    if longest >= 23.9:
        return "always_open"
    return "extended" if longest >= _EXTENDED_HOURS else "daytime"


def classify_frame(frame: pd.DataFrame, column: str = "OPERATING_HOURS") -> pd.Series:
    """Bucket every record's stated hours."""
    return frame[column].map(bucket_hours).astype("string")


def hours_report(frame: pd.DataFrame, column: str = "OPERATING_HOURS") -> HoursReport:
    """Bucket a frame and summarise the outcome."""
    buckets = classify_frame(frame, column)
    return HoursReport(counts=buckets.value_counts(), total=len(frame))


def always_open_mask(frame: pd.DataFrame, column: str = "OPERATING_HOURS") -> pd.Series:
    """Boolean mask of devices stating round-the-clock cover.

    Drives the 24/7-only constraint scenario. Conservative by construction:
    anything not positively identified as 24/7 — including unparseable records —
    is excluded rather than assumed available.
    """
    return classify_frame(frame, column) == "always_open"
