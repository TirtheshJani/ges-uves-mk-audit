# SPDX-License-Identifier: MIT
"""Non-ML reference baselines for the MK classifier.

Motivating question: "Why is LightGBM at macro-F1=0.926 interesting? Run a
kNN-on-spectra baseline or a simple Pickles template-fit baseline on the
same test split. If LightGBM beats it by 0.05 macro-F1, the audit is
meaningful. If LightGBM ties or barely beats a non-ML baseline, the whole
framing changes."

Two baselines, each on the same train_idx / test_idx split as the LightGBM
production run:

  ``knn_predict``: scikit-learn KNeighborsClassifier with a correlation
    distance (1 - Pearson r) over the 696 feature bins. Returns predictions
    per k in {1, 5, 11}.

  ``pickles_template_predict``: per test row, return the MK class of the
    chi-squared best-matching Pickles UVKLIB template under one of the three
    continuum conventions. Reuses
    :func:`src.interpret.benchmark.load_pickles_library`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _correlation_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """1 - Pearson r between every pair of rows in X1 and X2.

    Standardising each row to zero mean and unit norm reduces the
    correlation distance to (1 - X1_norm @ X2_norm.T) / 2 on the
    correlation in [-1, 1]; equivalently we report 1 - Pearson r in [0, 2].
    """
    def _row_norm(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64) - X.mean(axis=1, keepdims=True)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    A = _row_norm(X1)
    B = _row_norm(X2)
    return (1.0 - A @ B.T).astype(np.float32)


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_values: Iterable[int] = (1, 5, 11),
) -> dict[int, np.ndarray]:
    """Predict for each k by majority vote over the k nearest training rows.

    Distance is 1 - Pearson correlation across feature bins. Returns
    ``{k: y_pred_test}``. Ties on majority vote fall back to the nearest
    neighbour's label.
    """
    dist = _correlation_distance(X_test, X_train)
    out: dict[int, np.ndarray] = {}
    sorted_idx = np.argsort(dist, axis=1)
    for k in k_values:
        k = int(k)
        nearest = sorted_idx[:, :k]
        nearest_labels = y_train[nearest]
        preds = np.empty(len(X_test), dtype=y_train.dtype)
        for i in range(len(X_test)):
            labs, counts = np.unique(nearest_labels[i], return_counts=True)
            top = labs[counts == counts.max()]
            if len(top) == 1:
                preds[i] = top[0]
            else:
                # Tie: pick the label of the single nearest neighbour.
                preds[i] = nearest_labels[i, 0]
        out[k] = preds
    return out


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: Iterable[int]
) -> float:
    """Unweighted macro-F1 across ``class_labels``."""
    class_labels = list(class_labels)
    f1s: list[float] = []
    for c in class_labels:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def pickles_template_predict(
    X_test: np.ndarray,
    wave_centers: np.ndarray,
    pickles_dir: Path,
    continuum_method: str = "polynomial_n5",
    restrict_to_fgk: bool = True,
) -> tuple[np.ndarray, dict]:
    """Per test row, return the MK class of the lowest-chi-squared Pickles template.

    When ``restrict_to_fgk`` is True, only F/G/K Pickles templates are
    candidates (so the chi-squared minimum cannot land on M or earlier
    types). This is the appropriate non-ML baseline when comparing
    against an FGK-only classifier; rows whose closest FGK template is
    still a poor fit get the closest FGK label regardless.

    Returns ``(y_pred_str, stats)``; stats carries the count of usable
    templates and the mean per-row chi-squared at rank 1.
    """
    from src.interpret.benchmark import load_pickles_library

    templates, _ = load_pickles_library(
        pickles_dir, wave_centers, continuum_method=continuum_method,
    )
    if restrict_to_fgk:
        templates = [t for t in templates if t.mk_class in ("F", "G", "K")]
    if not templates:
        raise RuntimeError("no Pickles templates available after filtering")
    classes = np.array([t.mk_class for t in templates])
    fluxes = np.stack([t.flux_on_grid for t in templates], axis=0)  # (n_templates, n_bins)
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=object)
    chi2_at_rank1 = np.empty(n_test, dtype=np.float64)
    for r in range(n_test):
        diffs = fluxes - X_test[r][None, :]
        chi2 = np.nansum(diffs ** 2, axis=1)
        j = int(np.argmin(chi2))
        y_pred[r] = classes[j]
        chi2_at_rank1[r] = float(chi2[j])
    stats = {
        "n_templates_used": int(len(templates)),
        "continuum_method": continuum_method,
        "restrict_to_fgk": bool(restrict_to_fgk),
        "chi2_rank1_mean": float(np.mean(chi2_at_rank1)),
        "chi2_rank1_median": float(np.median(chi2_at_rank1)),
    }
    return y_pred.astype(str), stats
