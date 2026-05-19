#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure Mg b triplet equivalent widths on the held-out test set per class.

. Replaces the saturation rebuttal in draft2 manuscript
line 196 ("The Pickles K templates in our benchmark show Mg b equivalent
widths comparable to or exceeding the G templates in the same library")
with a measurement on the actual 145 K-class and 230 G-class test rows.

Output: ``artifacts/ablation/mg_b_ew_test.json`` with per-class summary
(n, median, IQR, mean, std), per-row arrays, and a two-sample KS test of
K vs G distributions.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.interpret.ew import (
    DEFAULT_CONTINUUM_HALF_WIDTH_AA,
    DEFAULT_LINE_HALF_WIDTH_AA,
    MG_B_LINES_AA,
    ks_two_sample,
    measure_mg_b_triplet,
    summary_per_class,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/features.npz"),
        help="path to features.npz with X, y, wave_centers, test_idx",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ablation/mg_b_ew_test.json"),
    )
    p.add_argument(
        "--line-half-width-aa",
        type=float,
        default=DEFAULT_LINE_HALF_WIDTH_AA,
        help="half-width (A) of the line window used to integrate (1-F/F_cont)",
    )
    p.add_argument(
        "--continuum-half-width-aa",
        type=float,
        default=DEFAULT_CONTINUUM_HALF_WIDTH_AA,
        help="half-width (A) of the local-continuum window around each line",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"]
    wave_centers = fp["wave_centers"]
    test_idx = fp["test_idx"]
    X_test = X[test_idx]
    y_test = y[test_idx]
    logger.info(
        "measuring Mg b triplet EW on %d test rows; line half-width=%.1f A, "
        "continuum half-width=%.1f A; lines at %s",
        len(test_idx), args.line_half_width_aa, args.continuum_half_width_aa,
        MG_B_LINES_AA,
    )
    out = measure_mg_b_triplet(
        X_test, wave_centers,
        line_half_width_aa=args.line_half_width_aa,
        continuum_half_width_aa=args.continuum_half_width_aa,
    )
    ew_total = out["ew_mg_b_total"]
    summary = summary_per_class(ew_total, y_test)
    per_line_summaries = {
        key: summary_per_class(out[f"ew_mg_b{i + 1}"], y_test)
        for i, key in enumerate(("ew_mg_b1", "ew_mg_b2", "ew_mg_b3"))
    }
    g_vals = ew_total[(y_test == 2) & np.isfinite(ew_total)]
    k_vals = ew_total[(y_test == 3) & np.isfinite(ew_total)]
    d_stat, p_val = ks_two_sample(g_vals, k_vals)
    g_median = float(np.median(g_vals)) if len(g_vals) else float("nan")
    k_median = float(np.median(k_vals)) if len(k_vals) else float("nan")
    payload = {
        "n_test": int(len(test_idx)),
        "mg_b_lines_aa": list(MG_B_LINES_AA),
        "line_half_width_aa": float(args.line_half_width_aa),
        "continuum_half_width_aa": float(args.continuum_half_width_aa),
        "mean_bin_width_aa": float(np.mean(np.diff(wave_centers))),
        "summary_ew_mg_b_total": summary,
        "summary_ew_mg_b_per_line": per_line_summaries,
        "ks_test_K_vs_G": {
            "n_K": int(len(k_vals)),
            "n_G": int(len(g_vals)),
            "median_K_aa": k_median,
            "median_G_aa": g_median,
            "K_minus_G_median_aa": k_median - g_median,
            "d_statistic": float(d_stat),
            "p_value": float(p_val),
            "interpretation": (
                "If K_minus_G_median > 0, K spectra carry STRONGER Mg b absorption "
                "than G (refuting the saturation rebuttal); if < 0, K is WEAKER "
                "than G (consistent with low-EW saturation); KS p < 0.05 means "
                "the two distributions differ significantly."
            ),
        },
        "per_row": {
            "ew_mg_b_total": ew_total.tolist(),
            "ew_mg_b1": out["ew_mg_b1"].tolist(),
            "ew_mg_b2": out["ew_mg_b2"].tolist(),
            "ew_mg_b3": out["ew_mg_b3"].tolist(),
            "y_test": y_test.astype(int).tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)
    logger.info(
        "per-class median Mg b total EW: F=%.3f A  G=%.3f A  K=%.3f A",
        summary["F"]["median_aa"], summary["G"]["median_aa"],
        summary["K"]["median_aa"],
    )
    logger.info(
        "K vs G KS test: D=%.4f  p=%.4g  K_median - G_median = %.3f A",
        d_stat, p_val, k_median - g_median,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
