# SPDX-License-Identifier: MIT
"""Parallel manifest builder for the GES UVES fetcher.

The upstream sequential builder in src.fetch.fetch_ges.build_manifest issues one
ESO TAP query per starlist row in a single thread, accumulating results in
memory and writing the manifest CSV only after the full loop completes. With
roughly 6,143 stars and one network round-trip per star, the sequential build
runs about two hours.

This wrapper parallelizes per-star queries via concurrent.futures and ALSO
inlines two corrections that work around bugs in
src.fetch.fetch_ges.query_eso_tap (which is kept unchanged):

1. ESO ObsCore exposes spatial coordinates as s_ra and s_dec (the IVOA standard
   prefix), not ra and dec. The upstream ADQL template uses o.ra and o.dec,
   which makes ESO's parser return HTTP 400 with "Encountered '.' Was expecting
   one of: <EOF> ',' ';' 'ASC' 'DESC'" for every query. Fix: use s_ra and
   s_dec in both the SELECT clause and the CONTAINS spatial predicate.

2. ObsCore's access_url for ESO Phase 3 spectra is a DataLink endpoint
   (application/x-votable+xml), not the FITS itself. The upstream code passes
   access_url straight into the downloader, which would write VOTable XML to a
   .fits filename. Fix: construct the FITS URL from dp_id as
   https://dataportal.eso.org/dataportal_new/file/<dp_id> (verified to return
   application/octet-stream with the FITS bytes).

Both bugs are documented in.

Usage:
    python -u -m scripts.fetch_ges_parallel \
        --starlist data/common/manifests/starlist_30k.parquet \
        --manifest data/ges/manifests/ges_manifest.csv \
        --workers 12 --concurrency 4 --mode both
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
from typing import Optional

import pandas as pd
import requests

from src.fetch.common import ensure_dir, http_head, write_manifest
from src.fetch.fetch_ges import download_from_manifest

ESO_TAP_SYNC = "https://archive.eso.org/tap_obs/sync"
ESO_FILE_BASE = "https://dataportal.eso.org/dataportal_new/file"

# Corrected ADQL: s_ra / s_dec are the standard ObsCore spatial columns at ESO.
# Returns the highest-calib_level UVES spectrum within radius_deg of (ra, dec).
ADQL_TEMPLATE = (
    "SELECT TOP 1 obs_publisher_did, dataproduct_type, calib_level, "
    "instrument_name, access_url, access_format, s_ra, s_dec, dp_id "
    "FROM ivoa.obscore "
    "WHERE instrument_name LIKE 'UVES%' "
    "AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))=1 "
    "AND dataproduct_type='spectrum' "
    "ORDER BY calib_level DESC"
)


def query_eso_tap_fixed(
    ra: float,
    dec: float,
    radius_arcsec: float = 1.0,
    timeout_s: float = 30.0,
) -> list[dict]:
    """Query ESO TAP for UVES spectra near (ra, dec). Corrected ADQL."""
    radius_deg = radius_arcsec / 3600.0
    adql = ADQL_TEMPLATE.format(ra=ra, dec=dec, radius_deg=radius_deg)
    resp = requests.post(
        ESO_TAP_SYNC,
        data={"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    cols = [c["name"] for c in data.get("metadata", [])]
    rows = data.get("data") or []
    return [{cols[i]: row[i] for i in range(len(cols))} for row in rows]


def query_one(ra: float, dec: float, base_dir: str = "data") -> Optional[dict]:
    """Run one ESO TAP query plus an HTTP HEAD; return one manifest row or None."""
    try:
        cands = query_eso_tap_fixed(ra, dec, radius_arcsec=1.0)
    except Exception:
        return None
    if not cands:
        return None
    best = cands[0]
    dp_id = best.get("dp_id")
    if not dp_id:
        return None
    remote = f"{ESO_FILE_BASE}/{dp_id}"
    code, size = http_head(remote, timeout=20)
    local_name = f"ges_uves_{ra:.6f}_{dec:.6f}.fits"
    local_path = os.path.join(base_dir, "ges", "uves", local_name)
    return {
        "remote_url": remote,
        "local_path": local_path,
        "status": "pending",
        "http_status": code,
        "bytes": size,
    }


def build_manifest_parallel(
    starlist_parquet: str,
    out_csv: str,
    base_dir: str = "data",
    workers: int = 12,
    progress_every: int = 100,
) -> int:
    """Concurrent manifest builder. Returns the number of manifest rows written."""
    df = pd.read_parquet(starlist_parquet)
    if not {"ra", "dec"}.issubset(df.columns):
        raise ValueError("starlist parquet missing 'ra'/'dec'")
    coords = df[["ra", "dec"]].dropna().drop_duplicates().reset_index(drop=True)
    n_total = len(coords)
    print(
        f"[parallel] querying ESO TAP for {n_total} stars with {workers} workers...",
        flush=True,
    )

    rows: list[dict] = []
    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(query_one, float(r["ra"]), float(r["dec"]), base_dir)
            for _, r in coords.iterrows()
        ]
        for fut in cf.as_completed(futures):
            completed += 1
            r = fut.result()
            if r is not None:
                rows.append(r)
            if completed % progress_every == 0 or completed == n_total:
                print(
                    f"[parallel] build progress: {completed}/{n_total} "
                    f"({100 * completed / n_total:.1f}%) hits={len(rows)}",
                    flush=True,
                )

    ensure_dir(os.path.dirname(out_csv))
    write_manifest(rows, out_csv)
    print(
        f"[parallel] manifest written: {len(rows)} entries -> {out_csv}",
        flush=True,
    )
    return len(rows)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Parallel manifest builder + reuse of downstream download phase"
    )
    p.add_argument(
        "--starlist",
        default=os.path.join("data", "common", "manifests", "starlist_30k.parquet"),
    )
    p.add_argument(
        "--manifest",
        default=os.path.join("data", "ges", "manifests", "ges_manifest.csv"),
    )
    p.add_argument("--base-dir", default="data")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--mode", choices=["build", "download", "both"], default="both")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--downloader", choices=["python", "wget"], default="python")
    args = p.parse_args(argv)

    ensure_dir(os.path.dirname(args.manifest))
    if args.mode in {"build", "both"}:
        build_manifest_parallel(
            args.starlist,
            args.manifest,
            base_dir=args.base_dir,
            workers=args.workers,
        )
    if args.mode in {"download", "both"}:
        download_from_manifest(
            args.manifest,
            concurrency=args.concurrency,
            downloader=args.downloader,
        )


if __name__ == "__main__":
    main()
