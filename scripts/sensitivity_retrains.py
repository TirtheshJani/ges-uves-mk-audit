#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Sensitivity retrains for the (Mg b, K) headline null.

Steps 8a (uniform-weights retest) and 8b (3x3 hyperparameter
sensitivity grid). Each cell retrains the LightGBM classifier with one
hyperparameter override, runs the masked_line_ablation on the (Mg b, K)
pair only on the held-out test set, and records whether the null persists.

The reviewer's Moderate concerns:

  8a: "Class-balanced weights not sensitivity-tested. The more direct test
     is to retrain with uniform weights and re-run the (Mg b, K) ablation."

  8b: "LightGBM hyperparameters not tuned. A reviewer will ask whether the
     K-class shortcut is an artefact of an under-trained model. Include a
     brief note: did you check that hyperparameter sensitivity preserves
     the (Mg b, K) null?"

Outputs:
  - artifacts/sensitivity/uniform_weights_mg_b_k.json
  - artifacts/sensitivity/hyperparam_grid_mg_b_k.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.interpret.classifier import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_HPARAMS,
    train,
)
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.occlusion import masked_line_ablation

logger = logging.getLogger(__name__)

MG_B_K_LINE_SETS: dict[str, list[tuple[float, float]]] = {"Mg_b": LINE_SETS["Mg_b"]}


def _retrain_and_ablate(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    wave_centers: np.ndarray,
    gap_mask: np.ndarray,
    class_labels: list[str],
    hparam_overrides: dict[str, Any],
    seed: int,
    n_bootstrap: int,
    n_random_controls: int,
    continuum_fill: float,
) -> dict[str, Any]:
    """Retrain on (train, val) under hparam_overrides; run (Mg b, K) ablation.

    Returns a dict carrying the trained-classifier macro-F1 on the
    held-out test set and the (Mg b, K) ablation row fields.
    """
    num_class = len(class_labels)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    t0 = time.time()
    model = train(
        X_train, y_train, X_val, y_val,
        num_class=num_class,
        hparams=hparam_overrides,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING,
    )
    fit_seconds = time.time() - t0
    y_pred_test = model.predict(X_test)
    test_acc = float(np.mean(y_pred_test == y_test))
    # Macro-F1 on test.
    f1s: list[float] = []
    for c in range(num_class):
        # y_train labels are 0..num_class-1; the test split should follow.
        # We will use unique class IDs present in y_test.
        pass
    unique_classes = sorted(np.unique(y_test).tolist())
    for c in unique_classes:
        tp = int(np.sum((y_test == c) & (y_pred_test == c)))
        fp = int(np.sum((y_test != c) & (y_pred_test == c)))
        fn = int(np.sum((y_test == c) & (y_pred_test != c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    rows = masked_line_ablation(
        model, X_test, y_test, wave_centers,
        MG_B_K_LINE_SETS, class_labels,
        per_class=True,
        n_bootstrap=n_bootstrap,
        n_random_controls=n_random_controls,
        seed=seed,
        gap_mask=gap_mask,
        continuum_fill=continuum_fill,
    )
    # Find the (Mg_b, K) row.
    mgk_row = next(
        (r for r in rows if r.line_set == "Mg_b" and r.mk_class == "K"), None
    )
    if mgk_row is None:
        raise RuntimeError("ablation did not produce (Mg_b, K) row")
    return {
        "hparam_overrides": hparam_overrides,
        "fit_seconds": fit_seconds,
        "test_macro_f1": macro_f1,
        "test_accuracy": test_acc,
        "mg_b_k": {
            "n_test": int(mgk_row.n_test),
            "baseline_acc": float(mgk_row.baseline_acc),
            "masked_acc_mean": float(mgk_row.masked_acc_mean),
            "delta_acc_mean": float(mgk_row.delta_acc_mean),
            "delta_acc_ci_low": float(mgk_row.delta_acc_ci_low),
            "delta_acc_ci_high": float(mgk_row.delta_acc_ci_high),
            "p_value_vs_random": float(mgk_row.p_value_vs_random),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/sensitivity"))
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--n-random-controls", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=("uniform_weights", "hyperparam_grid", "both"),
        default="both",
    )
    p.add_argument(
        "--continuum-fill",
        type=str,
        default="empirical-median",
        help="continuum fill value; same convention as scripts/ablation.py",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    train_idx = fp["train_idx"]
    val_idx = fp["val_idx"]
    test_idx = fp["test_idx"]
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"]
    # Pass the full ALLOWED_MK_CLASSES so that integer class ids in y index
    # correctly into class_labels even though A (label 0) was dropped at.
    # This matches scripts/ablation.py:307. Earlier iteration of this script
    # passed ["F","G","K"] which is off-by-one against the {1,2,3} encoding.
    class_labels = list(ALLOWED_MK_CLASSES)

    if args.continuum_fill == "empirical-median":
        fill = float(np.nanmedian(X[test_idx][:, ~gap_mask]))
    elif args.continuum_fill == "1.0":
        fill = 1.0
    else:
        fill = float(args.continuum_fill)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("uniform_weights", "both"):
        logger.info("Step 8a: uniform-weights retest")
        result = _retrain_and_ablate(
            X, y, train_idx, val_idx, test_idx, wave_centers, gap_mask,
            class_labels, {"class_weight": None},
            args.seed, args.n_bootstrap, args.n_random_controls, fill,
        )
        baseline_payload = {
            "step": "8a",
            "description": "uniform-weights retest (class_weight=None)",
            "continuum_fill": fill,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "n_random_controls": args.n_random_controls,
            **result,
        }
        path = args.out_dir / "uniform_weights_mg_b_k.json"
        with path.open("w") as f:
            json.dump(baseline_payload, f, indent=2)
        logger.info(
            "  uniform_weights: test_macro_f1=%.4f  (Mg_b,K) delta=%+.4f  CI=[%+.4f,%+.4f]  p=%.4f",
            result["test_macro_f1"], result["mg_b_k"]["delta_acc_mean"],
            result["mg_b_k"]["delta_acc_ci_low"], result["mg_b_k"]["delta_acc_ci_high"],
            result["mg_b_k"]["p_value_vs_random"],
        )
        logger.info(" wrote %s", path)

    if args.mode in ("hyperparam_grid", "both"):
        logger.info("Step 8b: 3x3 hyperparameter grid (9 retrains)")
        grid_cells: list[dict[str, Any]] = []
        for max_depth in (6, 8, 10):
            for num_leaves in (31, 63, 127):
                logger.info(
                    "  cell max_depth=%d num_leaves=%d", max_depth, num_leaves,
                )
                cell = _retrain_and_ablate(
                    X, y, train_idx, val_idx, test_idx, wave_centers, gap_mask,
                    class_labels,
                    {"max_depth": max_depth, "num_leaves": num_leaves},
                    args.seed, args.n_bootstrap, args.n_random_controls, fill,
                )
                grid_cells.append(cell)
                logger.info(
                    "    test_macro_f1=%.4f  (Mg_b,K) delta=%+.4f  CI=[%+.4f,%+.4f]  p=%.4f",
                    cell["test_macro_f1"], cell["mg_b_k"]["delta_acc_mean"],
                    cell["mg_b_k"]["delta_acc_ci_low"], cell["mg_b_k"]["delta_acc_ci_high"],
                    cell["mg_b_k"]["p_value_vs_random"],
                )
        grid_payload = {
            "step": "8b",
            "description": "hyperparameter sensitivity grid",
            "grid": {"max_depth": [6, 8, 10], "num_leaves": [31, 63, 127]},
            "continuum_fill": fill,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "n_random_controls": args.n_random_controls,
            "n_cells": len(grid_cells),
            "cells": grid_cells,
            "null_persists_all_cells": all(
                c["mg_b_k"]["p_value_vs_random"] > 0.05 for c in grid_cells
            ),
            "max_abs_delta_acc": float(max(
                abs(c["mg_b_k"]["delta_acc_mean"]) for c in grid_cells
            )),
            "min_p_value": float(min(
                c["mg_b_k"]["p_value_vs_random"] for c in grid_cells
            )),
        }
        path = args.out_dir / "hyperparam_grid_mg_b_k.json"
        with path.open("w") as f:
            json.dump(grid_payload, f, indent=2)
        logger.info(" wrote %s", path)
        logger.info(
            "  Step 8b summary: null persists in all %d cells: %s; max |delta|=%.4f; min p=%.4f",
            len(grid_cells), grid_payload["null_persists_all_cells"],
            grid_payload["max_abs_delta_acc"], grid_payload["min_p_value"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
