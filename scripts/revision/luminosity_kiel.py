#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Luminosity-class mix of the F/G/K sample, Kiel diagram and continuum-median figure.

Joins the Gaia-ESO stellar parameters (Teff, log g, [Fe/H]) onto the
feature rows by exact (ra_deg, dec_deg) match and reports, per class and
per split, the sample size, dwarf fraction (log g > 3.5), and median Teff,
log g and [Fe/H].  The production model's test accuracy is split by
luminosity class (dwarf vs giant) within each MK class, and the distance to
the nearest class Teff boundary is summarised on the test set.

Figures (vector PDF):
  * Kiel diagram (log g vs Teff, both axes inverted) coloured by class,
    with the class boundaries at 5300 K and 6000 K and the dwarf/giant
    divide at log g = 3.5; test rows are drawn with a distinct marker.
  * Per-class continuum-median box plots, production vs polynomial_n5,
    with every printed median and KS p-value read from
    artifacts/revision/scale_decomposition.json.

Outputs: artifacts/revision/luminosity_mix.json (+ luminosity_mix.csv),
submission/kiel_diagram.pdf, artifacts/figures/kiel_diagram.pdf,
submission/continuum_medians.pdf, artifacts/figures/continuum_medians.pdf.
Run from the repo root: ``python scripts/revision/luminosity_kiel.py``
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.interpret.benchmark import continuum_normalize  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "F", 2: "G", 3: "K"}
CLASSES = (1, 2, 3)
# Colour-blind safe (Okabe-Ito) hues threaded through both figures.
CLASS_COLORS = {"F": "#0072B2", "G": "#E69F00", "K": "#CC79A7"}
TEFF_BOUNDARIES_K = (5300.0, 6000.0)
LOGG_DWARF_DIVIDE = 3.5
BOUNDARY_NEAR_K = 150.0
SEED = 42

FONT_BASE, FONT_SMALL, FONT_TICK = 8, 7, 6


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_BASE,
        "axes.labelsize": FONT_BASE,
        "legend.fontsize": FONT_SMALL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def summarise(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "n_dwarf": int(df["dwarf_flag"].sum()),
        "dwarf_fraction": float(df["dwarf_flag"].mean()) if len(df) else float("nan"),
        "median_logg": float(df["logg"].median()),
        "median_teff_k": float(df["teff_k"].median()),
        "median_feh": float(df["feh"].median()),
    }


def kiel_diagram(df: pd.DataFrame, out_paths: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))  # avoid one class painting over another
    d = df.iloc[order]
    is_test = d["split"].to_numpy() == "test"
    for cls in ("F", "G", "K"):
        m = (d["mk_class"].to_numpy() == cls)
        ax.scatter(d.loc[m & ~is_test, "teff_k"], d.loc[m & ~is_test, "logg"],
                   s=4, c=CLASS_COLORS[cls], alpha=0.45, linewidths=0, rasterized=True)
        ax.scatter(d.loc[m & is_test, "teff_k"], d.loc[m & is_test, "logg"],
                   s=9, facecolors="none", edgecolors=CLASS_COLORS[cls], linewidths=0.5,
                   alpha=0.9, rasterized=True)
    for t in TEFF_BOUNDARIES_K:
        ax.axvline(t, color="0.35", lw=0.7, ls="--")
    ax.axhline(LOGG_DWARF_DIVIDE, color="0.35", lw=0.7, ls=":")
    ax.set_xlim(7400, 3800)
    ax.set_ylim(5.2, 0.2)
    ax.set_xlabel("effective temperature $T_\\mathrm{eff}$ (K)")
    ax.set_ylabel("surface gravity $\\log g$ (dex)")
    ax.set_title("K is mostly giants; F and G are dwarfs", loc="left")
    # Direct class labels in free whitespace instead of a legend box.
    ax.text(6700, 1.0, "F", color=CLASS_COLORS["F"], fontsize=FONT_BASE, fontweight="bold", ha="center")
    ax.text(5650, 1.0, "G", color=CLASS_COLORS["G"], fontsize=FONT_BASE, fontweight="bold", ha="center")
    ax.text(5150, 1.0, "K", color=CLASS_COLORS["K"], fontsize=FONT_BASE, fontweight="bold", ha="center")
    ax.text(7350, LOGG_DWARF_DIVIDE - 0.06, "giants", fontsize=FONT_SMALL, color="0.35",
            ha="left", va="bottom", style="italic")
    ax.text(7350, LOGG_DWARF_DIVIDE + 0.06, "dwarfs", fontsize=FONT_SMALL, color="0.35",
            ha="left", va="top", style="italic")
    # Marker key for split, drawn as a compact two-entry legend in the empty lower-left corner.
    h_tv = ax.scatter([], [], s=4, c="0.4", alpha=0.6, linewidths=0, label="train / val")
    h_te = ax.scatter([], [], s=9, facecolors="none", edgecolors="0.4", linewidths=0.5, label="test (open)")
    ax.legend(handles=[h_tv, h_te], loc="lower left", handletextpad=0.2, borderaxespad=0.3)
    fig.tight_layout()
    for pth in out_paths:
        pth.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pth, bbox_inches="tight")
        logger.info("wrote %s", pth)
    plt.close(fig)


def continuum_medians_figure(per_class_prod: dict, per_class_poly: dict,
                             numbers: dict, out_paths: list[Path]) -> None:
    """Two-panel box plot; printed medians and KS p-values come from ``numbers``."""
    fig, (ax_prod, ax_poly) = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=False)
    med_prod = numbers["row_median_distributions"]["production"]["per_class_median_of_row_medians"]
    med_poly = numbers["row_median_distributions"]["polynomial_n5"]["per_class_median_of_row_medians"]
    ks_prod = numbers["row_median_distributions"]["production"]["ks_pairwise"]
    ks_poly = numbers["row_median_distributions"]["polynomial_n5"]["ks_pairwise"]
    n_cls = numbers["row_median_distributions"]["production"]["per_class_n"]

    def _panel(ax, data_dict, med_dict, ks_dict, title, ks_corner="lower"):
        positions = [1, 2, 3]
        data = [data_dict[c] for c in ("F", "G", "K")]
        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7),
                        flierprops=dict(marker="o", markersize=2, alpha=0.5, linewidth=0.4))
        for patch, cls in zip(bp["boxes"], ("F", "G", "K")):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.6)
            patch.set_linewidth(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{c}\n(n={n_cls[c]})" for c in ("F", "G", "K")])
        ax.set_title(title, loc="left")
        ax.axhline(1.0, color="0.5", lw=0.6, ls="--")
        for cls, pos in zip(("F", "G", "K"), positions):
            ax.annotate(f"{med_dict[cls]:.3f}", xy=(pos, med_dict[cls]),
                        xytext=(pos + 0.31, med_dict[cls]), fontsize=FONT_SMALL,
                        ha="left", va="center", color=CLASS_COLORS[cls], fontweight="bold")
        ks_lines = ["two-sample KS, per-row medians"] + [
            f"{pair.replace('-', ' vs ')}: p = {ks_dict[pair]['p_value']:.1e}" for pair in ("F-G", "G-K", "F-K")
        ]
        yy, va = (0.03, "bottom") if ks_corner == "lower" else (0.97, "top")
        ax.text(0.03, yy, "\n".join(ks_lines), transform=ax.transAxes, ha="left", va=va,
                fontsize=FONT_SMALL, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
        ax.set_xlim(0.45, 3.95)

    _panel(ax_prod, per_class_prod, med_prod, ks_prod, "Pipeline normalisation: K sits lowest",
           ks_corner="lower")
    _panel(ax_poly, per_class_poly, med_poly, ks_poly, "Polynomial re-derivation: order reverses, span 0.005",
           ks_corner="upper")
    ax_prod.set_ylabel("per-row median of normalised flux")
    ax_prod.set_ylim(0.78, 1.04)
    ax_poly.set_ylim(0.955, 1.045)
    fig.tight_layout(w_pad=1.5)
    for pth in out_paths:
        pth.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pth, bbox_inches="tight")
        logger.info("wrote %s", pth)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--labels", type=Path, default=Path("artifacts/ges_mk_labels.parquet"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--scale-json", type=Path, default=Path("artifacts/revision/scale_decomposition.json"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/luminosity_mix.json"))
    p.add_argument("--kiel-out", type=Path, nargs="+",
                   default=[Path("submission/kiel_diagram.pdf"), Path("artifacts/figures/kiel_diagram.pdf")])
    p.add_argument("--medians-out", type=Path, nargs="+",
                   default=[Path("submission/continuum_medians.pdf"), Path("artifacts/figures/continuum_medians.pdf")])
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    np.random.seed(SEED)
    apply_style()

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    gap_mask = fp["gap_mask"].astype(bool)
    wave_centers = fp["wave_centers"]
    split = np.full(len(y), "train", dtype=object)
    split[fp["val_idx"]] = "val"
    split[fp["test_idx"]] = "test"
    feat = pd.DataFrame({
        "row": np.arange(len(y)), "ra_deg": fp["ra_deg"], "dec_deg": fp["dec_deg"],
        "y": y, "split": split, "dwarf_flag_features": fp["dwarf_flag"].astype(bool),
        "boundary_distance_k": fp["boundary_distance_k"],
    })
    labels = pd.read_parquet(args.labels)
    df = feat.merge(labels[["ra_deg", "dec_deg", "teff_k", "logg", "feh", "mk_class", "mk_int", "dwarf_flag"]],
                    on=["ra_deg", "dec_deg"], how="left", validate="one_to_one")
    if df["teff_k"].isna().any():
        raise SystemExit(f"{int(df['teff_k'].isna().sum())} feature rows failed to join onto labels")
    if not (df["y"] == df["mk_int"]).all():
        raise SystemExit("mk_int in labels disagrees with y in features")
    df["dwarf_flag"] = df["dwarf_flag"].astype(bool)
    join_checks = {
        "n_joined": int(len(df)),
        "n_unmatched": 0,
        "dwarf_flag_matches_features": bool((df["dwarf_flag"] == df["dwarf_flag_features"]).all()),
        "dwarf_flag_equals_logg_gt_3p5": bool(((df["logg"] > LOGG_DWARF_DIVIDE) == df["dwarf_flag"]).all()),
    }

    # --- per class x split summary ---------------------------------------
    mix = {}
    rows_csv = []
    for sp in ("all", "train", "val", "test"):
        sub = df if sp == "all" else df[df["split"] == sp]
        mix[sp] = {}
        for cls in ("F", "G", "K"):
            s = summarise(sub[sub["mk_class"] == cls])
            mix[sp][cls] = s
            rows_csv.append({"split": sp, "mk_class": cls, **s})
        mix[sp]["ALL"] = summarise(sub)

    # --- production-model accuracy by luminosity class on the test set ---
    with args.model.open("rb") as f:
        model = pickle.load(f)
    test = df[df["split"] == "test"].copy()
    test["pred"] = model.predict(X[test["row"].to_numpy()])
    test["correct"] = test["pred"] == test["y"]
    lum_acc = {}
    for cls in ("F", "G", "K"):
        d = {}
        for lum, flag in (("dwarf", True), ("giant", False)):
            s = test[(test["mk_class"] == cls) & (test["dwarf_flag"] == flag)]
            d[lum] = {"n": int(len(s)),
                      "accuracy": float(s["correct"].mean()) if len(s) else None,
                      "n_correct": int(s["correct"].sum()),
                      "pred_counts": {CLASS_NAMES[c]: int((s["pred"] == c).sum()) for c in CLASSES}}
        lum_acc[cls] = d
    g_to_k = test[(test["mk_class"] == "G") & (test["pred"] == 3)]
    lum_acc["G_to_K_errors"] = {"n": int(len(g_to_k)), "n_dwarf": int(g_to_k["dwarf_flag"].sum()),
                                "median_logg": float(g_to_k["logg"].median()) if len(g_to_k) else None,
                                "median_teff_k": float(g_to_k["teff_k"].median()) if len(g_to_k) else None}

    # --- Teff boundary distances on the test set --------------------------
    boundary = {}
    for cls in ("F", "G", "K"):
        s = test[test["mk_class"] == cls]
        boundary[cls] = {"n": int(len(s)),
                         "median_boundary_distance_k": float(s["boundary_distance_k"].median()),
                         f"n_within_{int(BOUNDARY_NEAR_K)}K": int((s["boundary_distance_k"] <= BOUNDARY_NEAR_K).sum()),
                         f"accuracy_within_{int(BOUNDARY_NEAR_K)}K": float(
                             s.loc[s["boundary_distance_k"] <= BOUNDARY_NEAR_K, "correct"].mean())
                         if (s["boundary_distance_k"] <= BOUNDARY_NEAR_K).any() else None,
                         f"accuracy_beyond_{int(BOUNDARY_NEAR_K)}K": float(
                             s.loc[s["boundary_distance_k"] > BOUNDARY_NEAR_K, "correct"].mean())
                         if (s["boundary_distance_k"] > BOUNDARY_NEAR_K).any() else None}

    # --- figures -----------------------------------------------------------
    kiel_diagram(df, args.kiel_out)

    numbers = json.loads(args.scale_json.read_text())
    X_test = X[fp["test_idx"]]
    y_test = y[fp["test_idx"]]
    X_poly = np.empty_like(X_test)
    for i, row in enumerate(X_test):
        X_poly[i] = continuum_normalize(wave_centers, row.astype(np.float64), method="polynomial_n5")
    per_class_prod = {CLASS_NAMES[c]: np.nanmedian(X_test[y_test == c][:, ~gap_mask], axis=1) for c in CLASSES}
    per_class_poly = {CLASS_NAMES[c]: np.nanmedian(X_poly[y_test == c][:, ~gap_mask], axis=1) for c in CLASSES}
    # Consistency guard: the box-plot medians must equal the JSON medians.
    for cls in ("F", "G", "K"):
        for src, arr in (("production", per_class_prod), ("polynomial_n5", per_class_poly)):
            j = numbers["row_median_distributions"][src]["per_class_median_of_row_medians"][cls]
            if abs(float(np.nanmedian(arr[cls])) - j) > 1e-5:
                raise SystemExit(f"median mismatch for {src}/{cls}: {np.nanmedian(arr[cls])} vs JSON {j}")
    continuum_medians_figure(per_class_prod, per_class_poly, numbers, args.medians_out)

    payload = {
        "description": ("Luminosity-class composition of the F/G/K sample per split, production-model "
                        "test accuracy by luminosity class, Teff-boundary proximity on the test set, "
                        "and provenance of the Kiel and continuum-median figures."),
        "inputs": {"features": str(args.features), "labels": str(args.labels), "model": str(args.model),
                   "scale_decomposition_json": str(args.scale_json),
                   "join_key": "exact (ra_deg, dec_deg)", "dwarf_definition": f"log g > {LOGG_DWARF_DIVIDE}",
                   "teff_boundaries_k": list(TEFF_BOUNDARIES_K), "boundary_near_threshold_k": BOUNDARY_NEAR_K,
                   "seed": SEED},
        "join_checks": join_checks,
        "luminosity_mix": mix,
        "test_accuracy_by_luminosity_class": lum_acc,
        "test_boundary_distance": boundary,
        "figures": {"kiel_diagram": [str(p) for p in args.kiel_out],
                    "continuum_medians": [str(p) for p in args.medians_out],
                    "continuum_medians_numbers_source": str(args.scale_json)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(rows_csv).to_csv(args.out.with_suffix(".csv"), index=False)
    logger.info("wrote %s", args.out)
    for cls in ("F", "G", "K"):
        logger.info("%s: dwarf fraction all=%.3f test=%.3f | test acc dwarf=%s (n=%d) giant=%s (n=%d)",
                    cls, mix["all"][cls]["dwarf_fraction"], mix["test"][cls]["dwarf_fraction"],
                    lum_acc[cls]["dwarf"]["accuracy"], lum_acc[cls]["dwarf"]["n"],
                    lum_acc[cls]["giant"]["accuracy"], lum_acc[cls]["giant"]["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
