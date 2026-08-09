"""Scan the repository for credentials before submission.

Safety-gate item 10: no credentials or private API keys in the prototype, logs,
repository, or presentation materials. None of this project's data sources
requires authentication, so any hit is a defect.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "AWS secret": re.compile(r"aws_secret_access_key\s*=\s*\S+", re.IGNORECASE),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    "generic assignment": re.compile(
        r"\b(api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['\"][^'\"]{12,}['\"]",
        re.IGNORECASE,
    ),
}

# The frozen data.gov.sg download URLs are signed and time-limited; they expired
# long before submission and are not credentials. They are excluded by path.
SKIP_SUFFIXES = {".geojson", ".png", ".jpg", ".pdf", ".lock", ".mp4"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}


def tracked_files() -> list[Path]:
    """Files git tracks, so untracked scratch work is not scanned."""
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def scan() -> list[tuple[Path, str, int]]:
    """Return every (path, pattern name, line number) hit."""
    hits: list[tuple[Path, str, int]] = []
    for path in tracked_files():
        if path.suffix in SKIP_SUFFIXES or set(path.parts) & SKIP_DIRS:
            continue
        if path.name == Path(__file__).name:  # this file defines the patterns
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    hits.append((path.relative_to(ROOT), label, line_number))
    return hits


def main() -> int:
    """Print findings and exit non-zero if any credential-like string is tracked."""
    hits = scan()
    for path, label, line_number in hits:
        print(f"{path}:{line_number}: possible {label}")
    if hits:
        print(f"\n{len(hits)} potential secret(s) found.")
        return 1
    print("No credentials found in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
