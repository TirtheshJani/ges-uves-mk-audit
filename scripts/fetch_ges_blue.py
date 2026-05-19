# SPDX-License-Identifier: MIT
"""Fetch UVES U580 BLUE-chip products from the ESO archive.

Companion to scripts/fetch_ges_parallel.py. The original parallel fetcher used
ADQL with `TOP 1 ORDER BY calib_level DESC`, which returned only the RED chip
(582-683 nm) per star because ESO's database orders the two products
of a U580 observation that way by default. The 4800-6800 A wavelength window
locked in needs both chips. This script fetches the missing BLUE
chip (477-580 nm) per star and writes them to filenames suffixed with `_BLUE`
so they don't collide with the existing red-chip files.

The blue chip is identified at the obscore level by `em_min < 5e-7` (i.e.
the spectrum starts below 500 nm; the U580 red chip starts at 582 nm so this
filter is exclusive).

Usage:
    python -u -m scripts.fetch_ges_blue \
        --starlist data/common/manifests/starlist_30k.parquet \
        --manifest data/ges/manifests/ges_manifest_blue.csv \
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

ADQL_TEMPLATE_BLUE = (
    "SELECT TOP 1 obs_publisher_did, dataproduct_type, calib_level, "
    "instrument_name, dp_id, em_min, em_max "
    "FROM ivoa.obscore "
    "WHERE instrument_name LIKE 'UVES%' "
    "AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))=1 "
    "AND dataproduct_type='spectrum' "
    "AND em_min < 5e-7 "
    "ORDER BY calib_level DESC"
)


def query_one_blue(ra: float, dec: float, base_dir: str = "data") -> Optional[dict]:
    """Query for the BLUE chip spectrum and return one manifest row."""
    radius_deg = 1.0 / 3600.0
    adql = ADQL_TEMPLATE_BLUE.format(ra=ra, dec=dec, radius_deg=radius_deg)
    try:
        resp = requests.post(
            ESO_TAP_SYNC,
            data={"QUERY": adql, "FORMAT": "json", "LANG": "ADQL"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    rows = data.get("data") or []
    if not rows:
        return None
    cols = [c["name"] for c in data.get("metadata", [])]
    row = {cols[i]: rows[0][i] for i in range(len(cols))}
    dp_id = row.get("dp_id")
    if not dp_id:
        return None
    remote = f"{ESO_FILE_BASE}/{dp_id}"
    code, size = http_head(remote, timeout=20)
    local_name = f"ges_uves_{ra:.6f}_{dec:.6f}_BLUE.fits"
    local_path = os.path.join(base_dir, "ges", "uves", local_name)
    return {
        "remote_url": remote,
        "local_path": local_path,
        "status": "pending",
        "http_status": code,
        "bytes": size,
    }


def build_blue_manifest(
    starlist_parquet: str,
    out_csv: str,
    base_dir: str = "data",
    workers: int = 12,
    progress_every: int = 100,
) -> int:
    df = pd.read_parquet(starlist_parquet)
    coords = df[["ra", "dec"]].dropna().drop_duplicates().reset_index(drop=True)
    n_total = len(coords)
    print(f"[blue] querying ESO TAP for {n_total} stars (BLUE chip) ...", flush=True)

    rows: list[dict] = []
    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(query_one_blue, float(r["ra"]), float(r["dec"]), base_dir)
            for _, r in coords.iterrows()
        ]
        for fut in cf.as_completed(futures):
            completed += 1
            r = fut.result()
            if r is not None:
                rows.append(r)
            if completed % progress_every == 0 or completed == n_total:
                print(
                    f"[blue] build progress: {completed}/{n_total} "
                    f"({100 * completed / n_total:.1f}%) hits={len(rows)}",
                    flush=True,
                )

    ensure_dir(os.path.dirname(out_csv))
    write_manifest(rows, out_csv)
    print(f"[blue] manifest written: {len(rows)} entries -> {out_csv}", flush=True)
    return len(rows)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Fetch UVES U580 BLUE-chip spectra")
    p.add_argument(
        "--starlist",
        default=os.path.join("data", "common", "manifests", "starlist_30k.parquet"),
    )
    p.add_argument(
        "--manifest",
        default=os.path.join("data", "ges", "manifests", "ges_manifest_blue.csv"),
    )
    p.add_argument("--base-dir", default="data")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--mode", choices=["build", "download", "both"], default="both")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--downloader", choices=["python", "wget"], default="python")
    args = p.parse_args(argv)

    ensure_dir(os.path.dirname(args.manifest))
    if args.mode in {"build", "both"}:
        build_blue_manifest(
            args.starlist, args.manifest, base_dir=args.base_dir, workers=args.workers
        )
    if args.mode in {"download", "both"}:
        download_from_manifest(
            args.manifest, concurrency=args.concurrency, downloader=args.downloader
        )


if __name__ == "__main__":
    main()
