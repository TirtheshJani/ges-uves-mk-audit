#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build MK-class labels for Gaia-ESO UVES spectra and write to parquet."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.interpret.labels import build_labels


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path,
                   help="regridded HDF5 from build_hdf5.py")
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="directory for the ESO TAP catalog cache parquet")
    p.add_argument("--out", required=True, type=Path,
                   help="output parquet path")
    p.add_argument("--match-radius-arcsec", type=float, default=0.5)
    p.add_argument("--min-per-class", type=int, default=50)
    p.add_argument("--warn-per-class", type=int, default=200)
    p.add_argument("--allow-drop-underfilled", action="store_true",
                   help="drop a class below min-per-class rather than raising")
    p.add_argument("--e-teff-max", type=float, default=200.0,
                   help="maximum E_TEFF in K ")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    labels_df, stats = build_labels(
        h5_path=args.h5,
        cache_dir=args.cache_dir,
        match_radius_arcsec=args.match_radius_arcsec,
        min_per_class=args.min_per_class,
        warn_per_class=args.warn_per_class,
        allow_drop_underfilled=args.allow_drop_underfilled,
        e_teff_max_k=args.e_teff_max,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(args.out, index=False)
    stats_path = args.out.with_suffix(".stats.json")
    stats_dict = asdict(stats)
    with stats_path.open("w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"wrote {len(labels_df)} labels -> {args.out}")
    print(f"stats -> {stats_path}")
    print(f"per-class: {stats.per_class}")
    print(f"n_dropped_e_teff: {stats.n_dropped_e_teff} (per-class: "
          f"{stats.per_class_dropped_e_teff})")
    if stats.ghost_audit is not None:
        ga = stats.ghost_audit
        print(
            f"ghost_audit: n_catalog_rows={ga.n_catalog_rows} "
            f"n_ghost={ga.n_ghost} pure_u580={ga.n_ghost_pure_u580} "
            f"mixed_setup={ga.n_ghost_mixed_setup}/{ga.n_mixed_setup_total} "
            f"rate={ga.mixed_setup_ghost_rate:.4f}"
        )


if __name__ == "__main__":
    main()
