#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Decompose the polynomial re-normalisation effect into scale and class-spread.

The production LightGBM model (``artifacts/lgbm_mk.pkl``) is held fixed and
the 456 held-out test rows are scored under a set of controlled transforms
of the input flux matrix:

  production            unchanged pipeline-normalised features
  polynomial_n5         every row re-normalised with the iteratively
                        sigma-clipped 5th-order Legendre fit used by the
                        audit-side re-derivation (fit over all 696 bins,
                        gap bins included, as in the existing scripts)
  polynomial_n5_gap_restored
                        as above, then the 22 inter-chip gap bins are reset
                        to their production values
  common_median         each row multiplied by (global test median / own
                        off-gap median): per-class spread removed, overall
                        scale kept
  uniform_x1.09         every row multiplied by 1.09: spread kept, scale shifted
  uniform_x_best        every row multiplied by M = median(polynomial_n5) /
                        median(production): the multiplier that reproduces
                        the polynomial_n5 median level
  gk_swap               G rows multiplied by med_K/med_G and K rows by
                        med_G/med_K (per-class median-of-row-medians swapped)

Multiplicative transforms act on the 674 off-gap bins only; gap bins are
left at their production values.  Also reported: a one-feature logistic
regression on the per-row off-gap median (train rows -> test rows), and the
per-class median-of-row-medians with pairwise two-sample KS p-values under
production and polynomial_n5.

Outputs: artifacts/revision/scale_decomposition.json and .csv
Run from the repo root: ``python scripts/revision/scale_decomposition.py``
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.interpret.benchmark import continuum_normalize  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "F", 2: "G", 3: "K"}
CLASSES = (1, 2, 3)
SEED = 42


def polynomial_renormalise(X: np.ndarray, wave_centers: np.ndarray, order: int = 5) -> np.ndarray:
    """Apply the sigma-clipped Legendre re-normalisation row by row."""
    out = np.empty_like(X)
    for i, row in enumerate(X):
        out[i] = continuum_normalize(
            wave_centers, row.astype(np.float64), method="polynomial_n5",
            polynomial_order=order,
        )
    return out


def row_medians(X: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    return np.nanmedian(X[:, ~gap_mask], axis=1)


def scale_rows(X: np.ndarray, factors: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    """Multiply off-gap bins of each row by its factor; gap bins unchanged."""
    out = X.copy()
    out[:, ~gap_mask] = X[:, ~gap_mask] * np.asarray(factors, dtype=X.dtype)[:, None]
    return out


def score(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X)
    cm = confusion_matrix(y, pred, labels=list(CLASSES))
    rec = recall_score(y, pred, labels=list(CLASSES), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(CLASSES), average="macro")),
        "per_class_recall": {CLASS_NAMES[c]: float(r) for c, r in zip(CLASSES, rec)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": [CLASS_NAMES[c] for c in CLASSES],
        "g_to_k": int(cm[1, 2]),
        "k_to_g": int(cm[2, 1]),
        "n_pred": {CLASS_NAMES[c]: int(np.sum(pred == c)) for c in CLASSES},
    }


def median_summary(X: np.ndarray, y: np.ndarray, gap_mask: np.ndarray) -> dict:
    rm = row_medians(X, gap_mask)
    per_class = {CLASS_NAMES[c]: rm[y == c] for c in CLASSES}
    ks = {}
    for a, b in (("F", "G"), ("G", "K"), ("F", "K")):
        res = ks_2samp(per_class[a], per_class[b])
        ks[f"{a}-{b}"] = {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
    return {
        "global_median_of_rows": float(np.nanmedian(X[:, ~gap_mask])),
        "per_class_median_of_row_medians": {
            k: float(np.nanmedian(v)) for k, v in per_class.items()
        },
        "per_class_iqr_of_row_medians": {
            k: [float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75))]
            for k, v in per_class.items()
        },
        "per_class_n": {k: int(len(v)) for k, v in per_class.items()},
        "ks_pairwise": ks,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/scale_decomposition.json"))
    p.add_argument("--polynomial-order", type=int, default=5)
    p.add_argument("--uniform-multiplier", type=float, default=1.09)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    np.random.seed(SEED)

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    train_idx, test_idx = fp["train_idx"], fp["test_idx"]
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"].astype(bool)
    with args.model.open("rb") as f:
        model = pickle.load(f)

    X_test, y_test = X[test_idx], y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    # --- transforms -------------------------------------------------------
    logger.info("polynomial_n%d re-normalisation of %d test rows", args.polynomial_order, len(X_test))
    X_poly = polynomial_renormalise(X_test, wave_centers, args.polynomial_order)
    X_poly_gap = X_poly.copy()
    X_poly_gap[:, gap_mask] = X_test[:, gap_mask]

    rm_prod = row_medians(X_test, gap_mask)
    global_med_prod = float(np.nanmedian(X_test[:, ~gap_mask]))
    global_med_poly = float(np.nanmedian(X_poly[:, ~gap_mask]))
    best_multiplier = global_med_poly / global_med_prod

    X_common = scale_rows(X_test, global_med_prod / rm_prod, gap_mask)
    X_u109 = scale_rows(X_test, np.full(len(X_test), args.uniform_multiplier), gap_mask)
    X_ubest = scale_rows(X_test, np.full(len(X_test), best_multiplier), gap_mask)

    med_cls = {c: float(np.nanmedian(rm_prod[y_test == c])) for c in CLASSES}
    swap_factors = np.ones(len(X_test))
    swap_factors[y_test == 2] = med_cls[3] / med_cls[2]
    swap_factors[y_test == 3] = med_cls[2] / med_cls[3]
    X_swap = scale_rows(X_test, swap_factors, gap_mask)

    interventions = {
        "production": (X_test, "unchanged pipeline-normalised features"),
        "polynomial_n5": (X_poly, "sigma-clipped 5th-order Legendre re-normalisation of every row; "
                                  "fit over all bins (gap included), all bins divided by the fit"),
        "polynomial_n5_gap_restored": (X_poly_gap, "polynomial_n5, then the 22 gap bins reset to production values"),
        "common_median": (X_common, "off-gap bins of each row scaled so the row median equals the "
                                    f"global test median {global_med_prod:.4f}"),
        f"uniform_x{args.uniform_multiplier:.2f}": (X_u109, f"off-gap bins of every row multiplied by {args.uniform_multiplier}"),
        "uniform_x_best": (X_ubest, f"off-gap bins of every row multiplied by {best_multiplier:.4f} "
                                    "(= median(polynomial_n5) / median(production))"),
        "gk_swap": (X_swap, f"G rows x {swap_factors[y_test == 2][0]:.4f}, K rows x {swap_factors[y_test == 3][0]:.4f} "
                            "(per-class median-of-row-medians swapped); F rows unchanged"),
    }
    results = {}
    for name, (Xv, desc) in interventions.items():
        results[name] = {"transform": desc, **score(model, Xv, y_test)}
        results[name]["median_level"] = float(np.nanmedian(Xv[:, ~gap_mask]))
        logger.info("%-28s macroF1=%.4f recall=%s G->K=%d", name, results[name]["macro_f1"],
                    {k: round(v, 3) for k, v in results[name]["per_class_recall"].items()},
                    results[name]["g_to_k"])

    # --- one-feature classifier on the per-row off-gap median -------------
    f_train = row_medians(X_train, gap_mask)[:, None]
    f_test = rm_prod[:, None]
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(f_train, y_train)
    pred_lr = lr.predict(f_test)
    gk = np.isin(y_test, (2, 3))
    # K is the positive class; lower row median -> K, so the score is -median.
    auc_gk_raw = float(roc_auc_score((y_test[gk] == 3).astype(int), -rm_prod[gk]))
    proba = lr.predict_proba(f_test)
    k_col = list(lr.classes_).index(3)
    g_col = list(lr.classes_).index(2)
    p_k_given_gk = proba[gk][:, k_col] / (proba[gk][:, k_col] + proba[gk][:, g_col])
    auc_gk_lr = float(roc_auc_score((y_test[gk] == 3).astype(int), p_k_given_gk))
    fg = np.isin(y_test, (1, 2))
    auc_fg_raw = float(roc_auc_score((y_test[fg] == 1).astype(int), rm_prod[fg]))
    one_feature = {
        "feature": "per-row median of off-gap normalised flux (production features)",
        "model": "sklearn LogisticRegression(class_weight='balanced', max_iter=1000), fit on train rows",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_accuracy_3class": float(accuracy_score(y_test, pred_lr)),
        "test_balanced_accuracy_3class": float(balanced_accuracy_score(y_test, pred_lr)),
        "test_macro_f1_3class": float(f1_score(y_test, pred_lr, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred_lr, labels=list(CLASSES)).tolist(),
        "roc_auc_g_vs_k_raw_median": auc_gk_raw,
        "roc_auc_g_vs_k_lr_conditional_prob": auc_gk_lr,
        "roc_auc_f_vs_g_raw_median": auc_fg_raw,
        "auc_orientation": "G vs K: K positive, score = -median (K rows have lower medians); "
                           "F vs G: F positive, score = +median",
        "n_g_test": int(np.sum(y_test == 2)),
        "n_k_test": int(np.sum(y_test == 3)),
    }

    gap_vals = X_test[:, gap_mask]
    payload = {
        "description": (
            "Scale-vs-spread decomposition of the polynomial re-normalisation effect "
            "on the fixed production LightGBM model, held-out test rows only."
        ),
        "inputs": {
            "features": str(args.features),
            "model": str(args.model),
            "n_test": int(len(test_idx)),
            "n_train_for_one_feature_classifier": int(len(train_idx)),
            "n_bins": int(X.shape[1]),
            "n_gap_bins": int(gap_mask.sum()),
            "polynomial_order": int(args.polynomial_order),
            "sigma_clip": 3.0,
            "sigma_clip_iters": 5,
            "seed": SEED,
            "uniform_multiplier": float(args.uniform_multiplier),
        },
        "gap_bin_treatment": {
            "polynomial_n5": "gap bins enter the Legendre fit (sigma-clipping removes strong outliers) "
                             "and are divided by the fitted continuum, as in the existing audit scripts",
            "polynomial_n5_gap_restored": "after re-normalisation the gap bins are reset to production values",
            "multiplicative_transforms": "only the off-gap bins are rescaled; gap bins keep production values",
            "gap_bin_values_production_test": {
                "mean": float(np.nanmean(gap_vals)),
                "median": float(np.nanmedian(gap_vals)),
                "std": float(np.nanstd(gap_vals)),
            },
        },
        "median_levels": {
            "production_global_test_median": global_med_prod,
            "polynomial_n5_global_test_median": global_med_poly,
            "best_uniform_multiplier": float(best_multiplier),
            "production_train_global_median": float(np.nanmedian(X_train[:, ~gap_mask])),
        },
        "interventions": results,
        "one_feature_classifier": one_feature,
        "row_median_distributions": {
            "production": median_summary(X_test, y_test, gap_mask),
            "polynomial_n5": median_summary(X_poly, y_test, gap_mask),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    rows = []
    for name, r in results.items():
        rows.append({
            "intervention": name,
            "recall_F": r["per_class_recall"]["F"],
            "recall_G": r["per_class_recall"]["G"],
            "recall_K": r["per_class_recall"]["K"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "g_to_k": r["g_to_k"],
            "k_to_g": r["k_to_g"],
            "median_level": r["median_level"],
            "confusion_flat_FGK": " ".join(str(v) for row in r["confusion_matrix"] for v in row),
        })
    pd.DataFrame(rows).to_csv(args.out.with_suffix(".csv"), index=False)
    logger.info("wrote %s and %s", args.out, args.out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
