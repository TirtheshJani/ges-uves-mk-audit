#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Run permutation importance, SHAP, and occlusion; report 3-way triangulation.

of the stellar-mk-audit. All three importance estimators run on
the validation split (test split is reserved for ablation).
(gap_mask) is enforced as a hard contract: the UVES inter-chip
gap [5769, 5834] A is excluded from every reported top-K.

Outputs (in --out-dir):
  - perm_importance.npz
  - shap_values.npz
  - shap_stability.json
  - occlusion.npz
  - triangulation_report.json (gates and red_flags)
  - per-class figure rendered to --figure-out (default
    artifacts/figures/importance_overlay_per_class.pdf) at 300 DPI.

Gates (acceptance):
  - pairwise_jaccard.perm_vs_shap >= 0.5
  - per_class_top_k contains no gap_mask bin
  - red_flags reported (may include "shortcut_learning_signal" when perm AND
    SHAP both miss Na D AND Ca I in G/K top-K post gap-mask)

References:
  - Sacco et al. 2014 A&A 565, A113 (UVES inter-chip gap)
  - Ness et al. 2015 ApJ 808, 16 (interpretability triangulation)
  - Pecaut and Mamajek 2013 ApJS 208, 9 (MK Teff bins)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg") # non-interactive backend; must precede pyplot import

import numpy as np

from src.interpret.classifier import load_model
from src.interpret.importance import compute_permutation_importance, save_importance
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS, lines_for_class
from src.interpret.occlusion import sliding_window_occlusion
from src.interpret.plotting import plot_per_class_importance
from src.interpret.shap_explain import (
    bootstrap_topk_stability,
    compute_shap_values,
    mean_abs_shap_per_class,
    save_shap,
    save_stability,
    stratified_subsample,
)
from src.interpret.triangulation import (
    pairwise_jaccard,
    per_class_topk,
    three_way_intersection,
    topk_excluding_mask,
)

logger = logging.getLogger(__name__)

# Tolerance for "near a diagnostic line" when scanning a class top-K
# against MK_LINES. 5 A matches the line-match precision tolerance
# and the typical perm-importance smoothing scale at 2.86 A/bin.
LINE_PROXIMITY_TOL_AA: float = 5.0


def _occ_score_per_bin(
    wave_centers: np.ndarray,
    occ_centers: np.ndarray,
    occ_delta: np.ndarray,
) -> np.ndarray:
    """Project sliding-window delta_acc onto the per-bin grid.

    delta_acc is negative when masking hurts; we flip sign so that
    larger score == more important (consistent with permutation and
    mean-abs SHAP).
    """
    return -np.interp(wave_centers, occ_centers, occ_delta)


def _topk_pre_mask(values: np.ndarray, k: int) -> set[int]:
    """Top-K without gap-mask filtering (diagnostic only)."""
    return set(int(i) for i in np.argsort(values)[-k:])


def _has_line_near(
    bins: list[int],
    wave_centers: np.ndarray,
    line_wavelengths_aa: list[float],
    tol_aa: float = LINE_PROXIMITY_TOL_AA,
) -> bool:
    """Return True if at least one ``bins`` index lies within ``tol_aa`` of any
    wavelength in ``line_wavelengths_aa``.
    """
    if not bins or not line_wavelengths_aa:
        return False
    bin_waves = np.asarray([wave_centers[i] for i in bins], dtype=float)
    targets = np.asarray(line_wavelengths_aa, dtype=float)
    # Vectorised: |bin_wave - target| min over targets per bin
    diffs = np.abs(bin_waves[:, None] - targets[None, :])
    return bool((diffs.min(axis=1) <= tol_aa).any())


def _bins_with_wavelengths(
    bins: list[int], wave_centers: np.ndarray
) -> list[dict[str, float]]:
    return [
        {"bin": int(i), "wavelength_aa": float(wave_centers[i])}
        for i in bins
    ]


def _shortcut_learning_red_flag(
    perm_per_class_topk: dict[str, list[int]],
    shap_per_class_topk: dict[str, list[int]],
    wave_centers: np.ndarray,
) -> str | None:
    """Gini carry-forward: if perm AND SHAP both miss Na D AND Ca I in
    G AND K top-K (post gap-mask), this is the shortcut-learning pivot
    signal per kill-criteria pivot.

    Returns the red-flag string when fired, else None. Does NOT pivot
    autonomously; the orchestrator decides whether to log a 
    pivot entry.
    """
    nad_waves = [5889.95, 5895.92]
    cai_waves = [6162.17, 6439.08]
    g_perm = perm_per_class_topk.get("G", [])
    k_perm = perm_per_class_topk.get("K", [])
    g_shap = shap_per_class_topk.get("G", [])
    k_shap = shap_per_class_topk.get("K", [])

    perm_misses_nad = (
        not _has_line_near(g_perm, wave_centers, nad_waves)
        and not _has_line_near(k_perm, wave_centers, nad_waves)
    )
    perm_misses_cai = (
        not _has_line_near(g_perm, wave_centers, cai_waves)
        and not _has_line_near(k_perm, wave_centers, cai_waves)
    )
    shap_misses_nad = (
        not _has_line_near(g_shap, wave_centers, nad_waves)
        and not _has_line_near(k_shap, wave_centers, nad_waves)
    )
    shap_misses_cai = (
        not _has_line_near(g_shap, wave_centers, cai_waves)
        and not _has_line_near(k_shap, wave_centers, cai_waves)
    )

    if (
        perm_misses_nad and perm_misses_cai
        and shap_misses_nad and shap_misses_cai
    ):
        return "shortcut_learning_signal"
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--figure-out",
        type=Path,
        default=Path("artifacts/figures/importance_overlay_per_class.pdf"),
        help="Per-class importance figure output path (PDF).",
    )
    p.add_argument("--shap-max-samples", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "run_interpret inputs: features=%s model=%s out_dir=%s top_k=%d seed=%d",
        args.features, args.model, args.out_dir, args.top_k, args.seed,
    )

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    val_idx = payload["val_idx"]

    if "gap_mask" not in payload.files:
        raise RuntimeError(
            "gap_mask missing from features.npz; contract violated"
        )
    gap_mask = payload["gap_mask"].astype(bool)
    logger.info(
        "gap_mask: %d bins True, range %.2f-%.2f A",
        int(gap_mask.sum()),
        float(wc[gap_mask].min()) if gap_mask.any() else float("nan"),
        float(wc[gap_mask].max()) if gap_mask.any() else float("nan"),
    )

    model = load_model(args.model)
    present = sorted(np.unique(y[val_idx]).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    logger.info("validation present classes: %s -> %s", present, class_labels)

    runtimes: dict[str, float] = {}

    # --- permutation importance on the validation set ---
    t0 = time.perf_counter()
    imp_mean, imp_std = compute_permutation_importance(
        model, X[val_idx], y[val_idx], n_repeats=10, seed=args.seed,
    )
    runtimes["perm_importance_s"] = time.perf_counter() - t0
    save_importance(args.out_dir / "perm_importance.npz", wc, imp_mean, imp_std)
    logger.info(
        "perm_importance: shape=%s std=%.6f runtime=%.2fs",
        imp_mean.shape, float(imp_mean.std()), runtimes["perm_importance_s"],
    )

    # --- SHAP on a stratified val subsample ---
    t0 = time.perf_counter()
    X_shap, y_shap, sub_idx = stratified_subsample(
        X[val_idx], y[val_idx], max_n=args.shap_max_samples, seed=args.seed,
    )
    shap_vals = compute_shap_values(model, X_shap)
    runtimes["shap_s"] = time.perf_counter() - t0
    save_shap(args.out_dir / "shap_values.npz", shap_vals, wc, sub_idx, y_shap)
    stability = bootstrap_topk_stability(
        shap_vals, top_k=args.top_k, n_bootstrap=100, seed=args.seed,
    )
    save_stability(args.out_dir / "shap_stability.json", stability, class_labels)
    mabs = mean_abs_shap_per_class(shap_vals)
    logger.info(
        "shap_values: shape=%s n_subsample=%d std=%.6f runtime=%.2fs",
        shap_vals.shape, X_shap.shape[0], float(mabs.std()), runtimes["shap_s"],
    )

    # --- sliding-window occlusion on the validation set ---
    # module contract: occlusion runs on val, not test. Test split is
    # reserved for the ablation null distribution.
    t0 = time.perf_counter()
    occ_centers, occ_delta = sliding_window_occlusion(
        model, X[val_idx], y[val_idx], wc, window_aa=50.0, stride_aa=25.0,
    )
    runtimes["occlusion_s"] = time.perf_counter() - t0
    np.savez_compressed(
        args.out_dir / "occlusion.npz",
        window_centers=occ_centers, delta_acc=occ_delta, wave_centers=wc,
    )
    logger.info(
        "occlusion: shape=%s std=%.6f runtime=%.2fs",
        occ_delta.shape, float(occ_delta.std()), runtimes["occlusion_s"],
    )

    # --- 3-way triangulation (global top-K, gap-mask filtered) ---
    shap_global = mabs.mean(axis=0)
    occ_score = _occ_score_per_bin(wc, occ_centers, occ_delta)

    perm_top = topk_excluding_mask(imp_mean, args.top_k, gap_mask)
    shap_top = topk_excluding_mask(shap_global, args.top_k, gap_mask)
    occ_top = topk_excluding_mask(occ_score, args.top_k, gap_mask)

    # Pre-mask diagnostic: how much gap_mask leakage would there have been?
    perm_top_pre = _topk_pre_mask(imp_mean, args.top_k)
    shap_top_pre = _topk_pre_mask(shap_global, args.top_k)
    occ_top_pre = _topk_pre_mask(occ_score, args.top_k)
    pre_mask_topk_diagnostic = {
        "perm_topk_pre_mask": sorted(int(i) for i in perm_top_pre),
        "perm_topk_pre_mask_in_gap": sorted(
            int(i) for i in perm_top_pre if gap_mask[i]
        ),
        "shap_topk_pre_mask": sorted(int(i) for i in shap_top_pre),
        "shap_topk_pre_mask_in_gap": sorted(
            int(i) for i in shap_top_pre if gap_mask[i]
        ),
        "occlusion_topk_pre_mask": sorted(int(i) for i in occ_top_pre),
        "occlusion_topk_pre_mask_in_gap": sorted(
            int(i) for i in occ_top_pre if gap_mask[i]
        ),
    }

    pairs = pairwise_jaccard({
        "perm": perm_top,
        "shap": shap_top,
        "occlusion": occ_top,
    })
    three_way = three_way_intersection(perm_top, shap_top, occ_top)

    # --- per-class top-K (post gap-mask) ---
    # perm importance is not per-class; reuse global perm score for each class
    # to provide a consistent record. mean |SHAP| is per-class natively.
    perm_per_class_scores = np.tile(imp_mean, (mabs.shape[0], 1))
    perm_per_class = per_class_topk(
        perm_per_class_scores, args.top_k, gap_mask, class_labels,
    )
    shap_per_class = per_class_topk(
        mabs, args.top_k, gap_mask, class_labels,
    )

    per_class_top_k_payload: dict[str, dict[str, list[dict[str, float]]]] = {
        "perm": {
            label: _bins_with_wavelengths(perm_per_class[label], wc)
            for label in class_labels
        },
        "shap": {
            label: _bins_with_wavelengths(shap_per_class[label], wc)
            for label in class_labels
        },
    }

    # --- red flags ---
    red_flags: list[str] = []
    # Hard-contract self-check (the test asserts this; keep it as a defence
    # in depth in case a future edit removes the gap-mask call upstream).
    for label in class_labels:
        for idx in perm_per_class[label] + shap_per_class[label]:
            if gap_mask[idx]:
                red_flags.append("gap_mask_leak")
                break
        if "gap_mask_leak" in red_flags:
            break

    # Jaccard gate (only perm-vs-shap is gated per amendment)
    if pairs["perm_vs_shap"] < 0.5:
        red_flags.append("perm_vs_shap_below_0.5")

    # Shortcut-learning carry-forward signal (Gini already missed
    # Na D and Ca I; this is the formal triangulation check).
    sl = _shortcut_learning_red_flag(perm_per_class, shap_per_class, wc)
    if sl is not None:
        red_flags.append(sl)

    triangulation = {
        "top_k": args.top_k,
        "class_labels": class_labels,
        "line_sets": list(LINE_SETS.keys()),
        "runtimes_s": runtimes,
        "gap_mask_n_bins_true": int(gap_mask.sum()),
        "gap_mask_wavelength_range_aa": [
            float(wc[gap_mask].min()) if gap_mask.any() else None,
            float(wc[gap_mask].max()) if gap_mask.any() else None,
        ],
        "pairwise_jaccard": pairs,
        "three_way_intersection_size": len(three_way),
        "three_way_intersection_bins": _bins_with_wavelengths(
            sorted(int(i) for i in three_way), wc,
        ),
        "per_class_top_k": per_class_top_k_payload,
        "pre_mask_topk_diagnostic": pre_mask_topk_diagnostic,
        "shap_stability_per_class": stability,
        "red_flags": red_flags,
    }
    out_report = args.out_dir / "triangulation_report.json"
    with out_report.open("w") as f:
        json.dump(triangulation, f, indent=2)
    logger.info("wrote triangulation report -> %s", out_report)

    # --- per-class figure ---
    plot_per_class_importance(
        wave_centers=wc,
        shap_per_class=mabs,
        class_labels=class_labels,
        gap_mask=gap_mask,
        out_path=args.figure_out,
    )
    logger.info("wrote per-class importance figure -> %s", args.figure_out)

    # --- console summary ---
    print("triangulation pairwise_jaccard:", json.dumps(pairs, indent=2))
    print(
        "three_way_intersection_size=%d bins=%s"
        % (len(three_way), sorted(int(i) for i in three_way))
    )
    print("per_class_top_k (post gap-mask):")
    for label in class_labels:
        perm_w = [round(wc[i], 2) for i in perm_per_class[label]]
        shap_w = [round(wc[i], 2) for i in shap_per_class[label]]
        diag_lines = [
            f"{ln.name}@{ln.wavelength_aa}"
            for ln in lines_for_class(label)
            if float(wc.min()) <= ln.wavelength_aa <= float(wc.max())
        ]
        print(f" {label}: perm bins -> {perm_per_class[label]}")
        print(f" {label}: perm wl -> {perm_w}")
        print(f" {label}: shap bins -> {shap_per_class[label]}")
        print(f" {label}: shap wl -> {shap_w}")
        print(f" {label}: class diag lines -> {diag_lines}")
    print("red_flags:", red_flags)


if __name__ == "__main__":
    main()
