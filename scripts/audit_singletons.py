#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Audit singleton-group composition of the spatial CV partition.

. the manuscript's StratifiedGroupKFold
pass produces a 0.010 macro-F1 gap relative to plain StratifiedKFold, and
draft2 Section 6 (line 217) claims this "bounds the residual cluster-leakage
contribution per Blanco-Cuaresma 2019." The reviewer asks what fraction of
the 1926 spatial groups are singletons and what fraction of training rows
live in singleton groups; if most rows are singletons, the group
stratification is partial and the 0.010 gap is a lower bound on leakage,
not a tight constraint.

This script re-derives the spatial groups from artifacts/features.npz on the
train+val partition (the partition trained CV on), computes the
singleton statistics, and patches the relevant fields into
artifacts/metrics.json so the manuscript dossier can cite them.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.interpret.classifier import derive_spatial_groups, singleton_group_stats

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/features.npz"),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/metrics.json"),
    )
    p.add_argument("--eps-deg", type=float, default=0.1)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument(
        "--partition",
        choices=("trainval", "train"),
        default="trainval",
        help="match spatial-CV scope; trainval combines train + val",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    train_idx = fp["train_idx"]
    val_idx = fp["val_idx"]
    if args.partition == "trainval":
        idx = np.concatenate([train_idx, val_idx])
    else:
        idx = train_idx
    ra = fp["ra_deg"][idx]
    dec = fp["dec_deg"][idx]
    logger.info(
        "re-deriving spatial groups on %d rows (%s partition) "
        "with eps_deg=%.4f min_samples=%d",
        len(idx), args.partition, args.eps_deg, args.min_samples,
    )
    groups = derive_spatial_groups(
        ra, dec, eps_deg=args.eps_deg, min_samples=args.min_samples
    )
    stats = singleton_group_stats(groups)
    logger.info(
        "singleton-group transparency: "
        "%d/%d groups are singletons (%.1f%%); "
        "%d/%d rows in singleton groups (%.1f%%)",
        stats["n_singleton_groups"], stats["n_unique_groups"],
        100 * stats["singleton_group_fraction"],
        stats["n_rows_in_singleton_groups"], stats["n_total_rows"],
        100 * stats["singleton_row_fraction"],
    )

    if args.metrics.exists():
        with args.metrics.open() as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics["cv_spatial_singleton_stats"] = {
        "partition": args.partition,
        "eps_deg": float(args.eps_deg),
        "min_samples": int(args.min_samples),
        **stats,
    }
    with args.metrics.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("patched %s under cv_spatial_singleton_stats", args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
