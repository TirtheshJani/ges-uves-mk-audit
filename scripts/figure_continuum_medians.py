#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Figure: per-class continuum-median distributions, production vs polynomial_n5.

Renders the per-class continuum-median spread (0.929 F, 0.919 G, 0.902 K on
the held-out test set) as a two-panel box-and-whisker figure. The shortcut
this figure surfaces is described in the manuscript Results section on the
continuum-level shortcut on K and G.

Two-panel figure:
  left: per-class continuum-median distribution under production
        (Gaia-ESO pipeline) normalisation
  right: same distribution after independent polynomial_n5 re-derivation

Each panel shows three box-plots (F, G, K) with per-class medians and
IQRs. The production panel includes pairwise KS-test annotations (F-G,
G-K, F-K) on the per-row medians.

Output: artifacts/figures/continuum_medians.pdf and submission/.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from src.interpret.benchmark import _continuum_polynomial_clipped

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "F", 2: "G", 3: "K"}
CLASS_COLORS = {"F": "#1f77b4", "G": "#2ca02c", "K": "#d62728"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/figures/continuum_medians.pdf"),
    )
    p.add_argument("--polynomial-order", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    test_idx = fp["test_idx"]
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"]

    X_test = X[test_idx]
    y_test = y[test_idx]

    logger.info("renormalising %d test rows", len(X_test))
    X_renorm = np.empty_like(X_test)
    for i, row in enumerate(X_test):
        X_renorm[i] = _continuum_polynomial_clipped(
            wave_centers, row.astype(np.float64), order=args.polynomial_order,
        )

    per_class_prod = {}
    per_class_poly = {}
    for cls in (1, 2, 3):
        m = y_test == cls
        per_class_prod[CLASS_NAMES[cls]] = np.nanmedian(
            X_test[m][:, ~gap_mask], axis=1
        )
        per_class_poly[CLASS_NAMES[cls]] = np.nanmedian(
            X_renorm[m][:, ~gap_mask], axis=1
        )

    ks_fg = ks_2samp(per_class_prod["F"], per_class_prod["G"])
    ks_gk = ks_2samp(per_class_prod["G"], per_class_prod["K"])
    ks_fk = ks_2samp(per_class_prod["F"], per_class_prod["K"])
    logger.info(
        "production KS: F-G p=%.2e  G-K p=%.2e  F-K p=%.2e",
        ks_fg.pvalue, ks_gk.pvalue, ks_fk.pvalue,
    )

    fig, (ax_prod, ax_poly) = plt.subplots(
        1, 2, figsize=(9.0, 4.4), sharey=False,
    )

    def _box_panel(ax, per_class_dict, title, label_below: bool = False):
        positions = [1, 2, 3]
        data = [per_class_dict[c] for c in ("F", "G", "K")]
        bp = ax.boxplot(
            data, positions=positions, widths=0.55, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, cls in zip(bp["boxes"], ("F", "G", "K")):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.55)
        ax.set_xticks(positions)
        ax.set_xticklabels(["F", "G", "K"], fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax.grid(axis="y", color="lightgrey", lw=0.4, alpha=0.5)
        # Per-class median annotations: place clearly outside the box so
        # they cannot overlap each other even when the medians are close.
        # On the production panel we place them just to the right of each
        # box; on the polynomial panel (where medians cluster near 1.00)
        # we stack them in a single row beneath the panel.
        if label_below:
            text_lines = [
                f"med$_{{{cls}}}$ = {float(np.nanmedian(per_class_dict[cls])):.3f}"
                for cls in ("F", "G", "K")
            ]
            ax.text(
                0.5, -0.18, "    ".join(text_lines),
                transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
            )
        else:
            for cls, pos in zip(("F", "G", "K"), positions):
                med = float(np.nanmedian(per_class_dict[cls]))
                ax.annotate(
                    f"med = {med:.3f}", xy=(pos, med),
                    xytext=(pos + 0.32, med),
                    fontsize=9, ha="left", va="center",
                    color=CLASS_COLORS[cls], fontweight="bold",
                )

    _box_panel(
        ax_prod, per_class_prod,
        "Production (Gaia-ESO pipeline normalisation)",
        label_below=False,
    )
    _box_panel(
        ax_poly, per_class_poly,
        r"Independent 5th-order Legendre re-derivation",
        label_below=True,
    )

    # Pairwise KS p-values: place them in a clean, free area inside the
    # production panel rather than crammed into the title.
    ks_lines = [
        r"Pairwise KS test (per-row medians):",
        f"  F vs G:  $p$ = {ks_fg.pvalue:.2e}",
        f"  G vs K:  $p$ = {ks_gk.pvalue:.2e}",
        f"  F vs K:  $p$ = {ks_fk.pvalue:.2e}",
    ]
    ax_prod.text(
        0.03, 0.03, "\n".join(ks_lines),
        transform=ax_prod.transAxes,
        ha="left", va="bottom", fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey",
                  lw=0.5, alpha=0.95),
    )

    ax_prod.set_ylabel("per-row median of normalised flux", fontsize=10)
    ax_poly.set_ylabel("")
    # Match y-limits across panels so the visual scale stays comparable
    # without forcing sharey (which would push the polynomial panel into a
    # tiny strip near 1.0).
    ax_prod.set_ylim(0.78, 1.04)
    ax_poly.set_ylim(0.94, 1.04)
    fig.suptitle(
        "Per-class continuum-median distributions: production vs polynomial re-derivation",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
