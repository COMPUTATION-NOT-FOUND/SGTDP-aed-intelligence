"""Fetch and freeze every raw input, then write data/manifest.json.

Run once, at the start of the build. Every downstream step reads only the frozen
files, so the pipeline reproduces with no network access (challenge brief,
supplemental-data rule 3).

Sources are pinned by dataset id -- see docs/sources.md for the register.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

API_BASE = "https://api-open.data.gov.sg/v1/public/api/datasets"
TIMEOUT = 120
RATE_LIMIT_SLEEP = 15  # 5 req/min unauthenticated; stay well under

# --- Source register -------------------------------------------------------
# Verified against the live dataset pages. Do not edit ids without re-verifying
# the field legend: the AED layer has near-duplicate siblings on data.gov.sg.
DATAGOV_SOURCES = [
    {
        "key": "aed",
        "filename": "aed_public_access.geojson",
        "dataset_id": "d_4e6b82c58a8a832f6f1fee5dfa6d47ea",
        "name": "Public Access AEDs",
        "publisher": "Singapore Civil Defence Force (SCDF)",
        "licence": "Singapore Open Data Licence",
        "page_url": "https://data.gov.sg/datasets/d_4e6b82c58a8a832f6f1fee5dfa6d47ea/view",
        "coverage_date": "2020-02",
        "page_last_updated": "2025-11-13",
        "role": "Existing public-access AED network (the Feb 2020 registry snapshot).",
    },
    {
        "key": "planning_areas",
        "filename": "planning_areas_no_sea.geojson",
        "dataset_id": "d_4765db0e87b9c86336792efe8a1f7a66",
        "name": "Master Plan 2019 Planning Area Boundary (No Sea)",
        "publisher": "Urban Redevelopment Authority (URA)",
        "licence": "Singapore Open Data Licence",
        "page_url": "https://data.gov.sg/datasets/d_4765db0e87b9c86336792efe8a1f7a66/view",
        "coverage_date": "2019",
        "page_last_updated": "2025-12-05",
        "role": "Toa Payoh planning-area boundary; clips every layer. 'No Sea' variant "
        "is required so hexes are never generated over water.",
    },
    {
        "key": "subzones",
        "filename": "subzones_no_sea.geojson",
        "dataset_id": "d_8594ae9ff96d0c708bc2af633048edfb",
        "name": "Master Plan 2019 Subzone Boundary (No Sea)",
        "publisher": "Urban Redevelopment Authority (URA)",
        "licence": "Singapore Open Data Licence",
        "page_url": "https://data.gov.sg/datasets/d_8594ae9ff96d0c708bc2af633048edfb/view",
        "coverage_date": "2019",
        "page_last_updated": None,
        "role": "Subzone polygons for the worst-served-subzone equity metric.",
    },
]

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Buildings carry the candidate footprints and the floor-area demand proxy;
# the POI tags carry the tag-allowlist signal and the POI-density proxy.
OSM_TAGS = {
    "building": True,
    "amenity": True,
    "shop": True,
    "leisure": True,
    "public_transport": True,
    "railway": ["station"],
}

# Four planning areas spanning the coverage range measured across 15 areas.
# The only OSM tags the pipeline consumes. The raw Overpass response carries 274
# columns; keeping all of them puts ~170 MB of unread tags in the repository, so
# the extract is slimmed to these before freezing. Recorded as a transformation.
OSM_KEEP = [
    "geometry",
    "building",
    "building:levels",
    "amenity",
    "shop",
    "leisure",
    "public_transport",
    "railway",
    "name",
    "addr:housename",
    "addr:housenumber",
    "addr:street",
]

TOWNS = {
    "toa-payoh": "TOA PAYOH",
    "ang-mo-kio": "ANG MO KIO",
    "bedok": "BEDOK",
    "jurong-east": "JURONG EAST",
}


def sha256_of(path: Path) -> str:
    """Return the hex SHA256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_datagov(source: dict) -> Path:
    """Download one data.gov.sg dataset and freeze it to data/raw/.

    GeoJSON datasets may skip the initiate-download step and poll directly. The
    poll response carries a short-lived signed URL, so the file is downloaded in
    the same run -- the URL itself is never persisted.
    """
    dest = RAW / source["filename"]
    poll = f"{API_BASE}/{source['dataset_id']}/poll-download"

    for attempt in range(1, 5):
        resp = requests.get(poll, timeout=TIMEOUT)
        # The endpoint answers 201 (not 200) and returns the signed URL directly;
        # there is no intermediate "status" to poll for GeoJSON datasets.
        if resp.ok:
            payload = resp.json().get("data", {})
            url = payload.get("url")
            if url:
                blob = requests.get(url, timeout=TIMEOUT)
                blob.raise_for_status()
                dest.write_bytes(blob.content)
                return dest
            print(f"    no url in payload (status={payload.get('status')!r}), retrying")
        else:
            print(f"    HTTP {resp.status_code}, retrying")
        time.sleep(RATE_LIMIT_SLEEP * attempt)

    raise RuntimeError(f"could not download {source['name']} ({source['dataset_id']})")


def fetch_osm(boundary_path: Path, slug: str, area: str) -> tuple[Path, str]:
    """Fetch OSM buildings + POIs inside one planning-area boundary and freeze them.

    OSM is unversioned, so each extract is frozen with a retrieval timestamp and
    hashed; a rerun tomorrow returns different data. Tries each mirror in turn.
    """
    import geopandas as gpd
    import osmnx as ox

    areas = gpd.read_file(boundary_path)
    name_col = next((c for c in areas.columns if c.upper() == "PLN_AREA_N"), None)
    if name_col is None:
        raise RuntimeError(
            "PLN_AREA_N absent -- attributes are probably inside an HTML table in "
            "'Description'. Parse it with pandas.read_html (see plan R3)."
        )

    town = areas[areas[name_col].str.upper().str.strip() == area]
    if len(town) == 0:
        raise RuntimeError(f"no polygon matched {area!r} in {name_col}")
    polygon = town.to_crs(4326).union_all()

    # Some URA boundaries carry ring self-intersections -- BEDOK does, at
    # (103.9321, 1.3055) -- and osmnx refuses an invalid polygon outright. A
    # zero-width buffer repairs the ring without moving the boundary.
    if not polygon.is_valid:
        repaired = polygon.buffer(0)
        print(
            f"    repaired invalid {area} boundary "
            f"(area change {abs(repaired.area - polygon.area) / polygon.area:.2e})"
        )
        polygon = repaired

    dest = RAW / f"osm_{slug}.parquet"
    last_error: Exception | None = None
    for mirror in OVERPASS_MIRRORS:
        try:
            print(f"    trying {mirror}")
            ox.settings.overpass_url = mirror.rsplit("/api/", 1)[0] + "/api"
            feats = ox.features_from_polygon(polygon, tags=OSM_TAGS).reset_index()
            keep = [c for c in OSM_KEEP if c in feats.columns]
            feats = feats[keep]
            for col in keep:
                if col != "geometry":
                    feats[col] = feats[col].astype("string")
            # GeoParquet: ~10x smaller than GeoJSON and read natively by geopandas.
            feats.to_parquet(dest, index=False)
            return dest, mirror
        except Exception as exc:
            last_error = exc
            print(f"    failed: {type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(5)

    raise RuntimeError(f"all Overpass mirrors failed; last error: {last_error}")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(UTC).isoformat(timespec="seconds")
    entries = []

    for source in DATAGOV_SOURCES:
        print(f"[{source['key']}] {source['name']}")
        path = fetch_datagov(source)
        digest = sha256_of(path)
        print(f"    {path.name}  {path.stat().st_size:,} bytes  sha256={digest[:16]}...")
        entries.append(
            {
                **{k: v for k, v in source.items() if k != "filename"},
                "path": f"data/raw/{path.name}",
                "format": "GeoJSON",
                "bytes": path.stat().st_size,
                "sha256": digest,
                "retrieved_utc": retrieved,
                "transformations": [],
            }
        )
        time.sleep(RATE_LIMIT_SLEEP)

    boundary = RAW / "planning_areas_no_sea.geojson"
    for slug, area in TOWNS.items():
        print(f"[osm:{slug}] OpenStreetMap buildings + POIs ({area})")
        osm_file, mirror = fetch_osm(boundary, slug, area)
        osm_digest = sha256_of(osm_file)
        print(
            f"    {osm_file.name}  {osm_file.stat().st_size:,} bytes  sha256={osm_digest[:16]}..."
        )
        entries.append(_osm_entry(slug, area, osm_file, osm_digest, mirror, retrieved))

    MANIFEST.write_text(
        json.dumps(
            {
                "generated_utc": retrieved,
                "towns": TOWNS,
                "note": "Every judged result is reproducible from these frozen files "
                "with no network access.",
                "sources": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nmanifest -> {MANIFEST.relative_to(ROOT)}  ({len(entries)} sources)")
    return 0


def _osm_entry(slug: str, area: str, path: Path, digest: str, mirror: str, retrieved: str) -> dict:
    """Manifest entry for one town's frozen OSM extract."""
    return {
        "key": f"osm:{slug}",
        "dataset_id": None,
        "name": f"OpenStreetMap buildings and POIs, {area} planning area",
        "publisher": "OpenStreetMap contributors",
        "licence": "ODbL 1.0 (attribution and share-alike required)",
        "page_url": mirror,
        "coverage_date": None,
        "page_last_updated": None,
        "role": "Candidate building footprints, tag allowlist signal, and the "
        "floor-area / POI-density demand proxies.",
        "path": f"data/raw/{path.name}",
        "format": "GeoParquet",
        "bytes": path.stat().st_size,
        "sha256": digest,
        "retrieved_utc": retrieved,
        "transformations": [
            f"Clipped to the MP2019 {area} planning-area polygon (No Sea).",
            f"Retained only the {len(OSM_KEEP) - 1} tags the pipeline consumes; the "
            "remaining OSM tag columns were dropped unread.",
            "Written as GeoParquet rather than GeoJSON for size.",
        ],
        "notes": "OSM is unversioned: a later refetch returns different data. "
        "This frozen extract, not the live API, is the source of truth.",
    }


if __name__ == "__main__":
    sys.exit(main())
