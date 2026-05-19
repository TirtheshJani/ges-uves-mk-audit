#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Figure 6: per-class continuum-median distributions under production and polynomial_n5.

review item 3. The reviewer asked for the per-class
continuum-median spread (0.929 F, 0.919 G, 0.902 K on the test set) to
be promoted from prose into a figure, since it is now central evidence
for the continuum-level shortcut described in Section 5.

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
        1, 2, figsize=(8.5, 4.0), sharey=True,
    )

    def _box_panel(ax, per_class_dict, title):
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
        ax.set_xticklabels(["F", "G", "K"])
        ax.set_ylabel("per-row continuum median (\\AA-1 dex-1 normalised)")
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="grey", lw=0.6, ls="--", alpha=0.6)
        for cls, pos in zip(("F", "G", "K"), positions):
            med = float(np.nanmedian(per_class_dict[cls]))
            ax.annotate(
                f"med={med:.3f}", xy=(pos, med),
                xytext=(pos + 0.30, med),
                fontsize=8, ha="left", va="center",
                color=CLASS_COLORS[cls],
            )

    _box_panel(
        ax_prod, per_class_prod,
        f"production (Gaia-ESO pipeline)\nKS F-G p={ks_fg.pvalue:.2e}  G-K p={ks_gk.pvalue:.2e}  F-K p={ks_fk.pvalue:.2e}",
    )
    _box_panel(
        ax_poly, per_class_poly,
        "polynomial_n5 re-derivation",
    )

    ax_prod.set_ylabel("per-row median of normalised flux")
    ax_poly.set_ylabel("")
    fig.suptitle(
        "Per-class continuum-median distributions: production vs polynomial_n5",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
