"""Manifest loading and checksum verification.

Deliverable 5 of the challenge brief asks for frozen input versions *or*
checksums. Asserting the checksums in a test is what makes them load-bearing
rather than decorative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from aed_siting.config import MANIFEST_PATH, ROOT

REQUIRED_ENTRY_FIELDS = (
    "name",
    "publisher",
    "licence",
    "page_url",
    "retrieved_utc",
    "path",
    "sha256",
    "role",
    "transformations",
)


@dataclass(frozen=True)
class ChecksumResult:
    """Outcome of verifying one frozen input."""

    key: str
    path: Path
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        """True when the file on disk matches the manifest."""
        return self.expected == self.actual


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load the data manifest.

    Raises:
        FileNotFoundError: If the manifest is absent. Run
            ``scripts/fetch_sources.py`` once to create it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python scripts/fetch_sources.py` first."
        )
    return json.loads(path.read_text())


def verify_checksums(path: Path = MANIFEST_PATH) -> list[ChecksumResult]:
    """Re-hash every frozen input and compare against the manifest.

    Returns:
        One result per source, in manifest order.

    Limits:
        Proves the bytes are unchanged since the manifest was written. It does
        not prove they match the organiser's event-provided frozen copy — that
        substitution is disclosed separately in the evaluation report.
    """
    manifest = load_manifest(path)
    results = []
    for entry in manifest["sources"]:
        file_path = ROOT / entry["path"]
        results.append(
            ChecksumResult(
                key=entry["key"],
                path=file_path,
                expected=entry["sha256"],
                actual=sha256_of(file_path) if file_path.exists() else "<missing>",
            )
        )
    return results
