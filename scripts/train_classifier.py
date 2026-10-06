#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Train the LightGBM MK classifier from a features.npz payload.

driver: 5-fold StratifiedGroupKFold on combined train+val per,
final model fit on full train+val, evaluated on the held-out test partition,
boundary-filtered sensitivity check per.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from src.interpret.classifier import (
    boundary_filtered_accuracy,
    derive_spatial_groups,
    evaluate,
    metrics_to_dict,
    save_model,
    train,
    train_cv,
    write_metrics,
)
from src.interpret.lines import ALLOWED_MK_CLASSES


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model-out", required=True, type=Path)
    p.add_argument("--metrics-out", required=True, type=Path)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of stratified group-aware folds .")
    p.add_argument("--cv-seed", type=int, default=42,
                   help="Random seed for CV splitter.")
    p.add_argument("--per-fold-timeout-s", type=float, default=3600.0,
                   help="Per-fold runtime cap; exceeding falls through to single-split.")
    p.add_argument("--single-split", action="store_true",
                   help="Skip CV; fall through to legacy single-split fit .")
    p.add_argument("--boundary-threshold-k", type=float, default=200.0,
                   help="|boundary_distance_k| threshold for sensitivity subset.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("train_classifier")
    t0 = time.perf_counter()

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    train_idx = payload["train_idx"]
    val_idx = payload["val_idx"]
    test_idx = payload["test_idx"]
    groups = payload["groups"]
    boundary_distance_k = payload["boundary_distance_k"]

    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    num_class = len(present)
    logger.info(
        "classes present (sorted, encoding -> label): %s",
        {int(c): ALLOWED_MK_CLASSES[c] for c in present},
    )
    logger.info(
        "splits: train=%d val=%d test=%d  n_bins=%d",
        len(train_idx), len(val_idx), len(test_idx), X.shape[1],
    )

    # acknowledged in log: A class dropped at ; A<->K gate N/A.
    logger.info(
        " A class dropped at (n=1); A<->K confusion gate N/A.",
    )

    # Combined train+val for CV and for the final fit.
    tv_idx = np.concatenate([train_idx, val_idx])
    X_tv = X[tv_idx]
    y_tv = y[tv_idx]
    g_tv = groups[tv_idx]
    n_unique_groups_tv = int(len(np.unique(g_tv)))
    logger.info(
        "train+val combined: n=%d, unique groups=%d ",
        len(tv_idx), n_unique_groups_tv,
    )

    # 1. Cross-validation on train+val (or skip if --single-split).
    cv_results = None
    cv_spatial_results: dict | None = None
    if args.single_split:
        logger.info("--single-split set; skipping CV per fallback.")
    else:
        logger.info(
            "running %d-fold StratifiedGroupKFold (random_state=%d)",
            args.cv_folds, args.cv_seed,
        )
        cv_results = train_cv(
            X=X_tv,
            y=y_tv,
            groups=g_tv,
            num_class=num_class,
            n_splits=args.cv_folds,
            random_state=args.cv_seed,
            per_fold_timeout_s=args.per_fold_timeout_s,
            class_labels=class_labels,
        )
        logger.info(
            "CV: mean=%.4f std=%.4f folds=%d fallback=%s splitter=%s",
            cv_results["cv_macro_f1_mean"],
            cv_results["cv_macro_f1_std"],
            len(cv_results["per_fold"]),
            cv_results["cv_fallback_triggered"],
            cv_results["splitter"],
        )

        # 1b. spatial-group sensitivity CV. Derive spatial
        # groups from ra_deg/dec_deg in features.npz, then run a SECOND
        # StratifiedGroupKFold over those groups so cluster-mate spectra
        # cannot leak between folds (Blanco-Cuaresma 2019). The original
        # cv_results above is retained verbatim for transparency.
        if "ra_deg" not in payload.files or "dec_deg" not in payload.files:
            raise RuntimeError(
                "mitigation requires ra_deg and dec_deg keys in "
                "features.npz; rebuild with the updated scripts/build_features.py"
            )
        ra_deg = payload["ra_deg"]
        dec_deg = payload["dec_deg"]
        spatial_groups_full = derive_spatial_groups(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            eps_deg=0.1,
            min_samples=5,
        )
        g_spatial_tv = spatial_groups_full[tv_idx]
        n_spatial_unique_tv = int(len(np.unique(g_spatial_tv)))
        logger.info(
            " spatial groups derived (n_unique_full=%d, "
            "n_unique_train+val=%d); running SECOND CV with these groups",
            int(len(np.unique(spatial_groups_full))),
            n_spatial_unique_tv,
        )
        cv_spatial_results = train_cv(
            X=X_tv,
            y=y_tv,
            groups=g_spatial_tv,
            num_class=num_class,
            n_splits=args.cv_folds,
            random_state=args.cv_seed,
            per_fold_timeout_s=args.per_fold_timeout_s,
            class_labels=class_labels,
        )
        logger.info(
            "spatial-CV: mean=%.4f std=%.4f folds=%d fallback=%s splitter=%s",
            cv_spatial_results["cv_macro_f1_mean"],
            cv_spatial_results["cv_macro_f1_std"],
            len(cv_spatial_results["per_fold"]),
            cv_spatial_results["cv_fallback_triggered"],
            cv_spatial_results["splitter"],
        )

    # 2. Final fit on full train+val with internal val split for early stopping.
    # Use the original train_idx for fit and val_idx for early stopping.
    logger.info(
        "final fit: train rows=%d (early-stop val rows=%d)",
        len(train_idx), len(val_idx),
    )
    model = train(
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        num_class=num_class,
    )

    # 3. Evaluate on held-out test partition.
    metrics = evaluate(
        model,
        X[test_idx], y[test_idx],
        class_labels=class_labels,
        n_train=len(train_idx),
        n_val=len(val_idx),
    )
    logger.info(
        "test metrics: acc=%.4f macro_f1=%.4f per_class_recall=%s",
        metrics.accuracy, metrics.macro_f1, metrics.per_class_recall,
    )

    # 4. Boundary-filtered sensitivity (applied to the TEST split only).
    boundary = boundary_filtered_accuracy(
        model=model,
        X_test=X[test_idx],
        y_test=y[test_idx],
        boundary_distance_k=boundary_distance_k[test_idx],
        threshold_k=args.boundary_threshold_k,
    )
    logger.info(
        "boundary delta_acc=%.4f ci=[%.4f, %.4f] (n_filtered=%d/%d)",
        boundary["delta_acc"],
        boundary["delta_acc_ci_low"],
        boundary["delta_acc_ci_high"],
        int(boundary["n_filtered"]),
        int(boundary["n_test"]),
    )

    # 5. Compose flat metrics payload and persist artifacts.
    payload_out = metrics_to_dict(metrics, cv_results, boundary)

    # append cv_spatial_* fields next to the original cv_*
    # fields. The original cv_* block (StratifiedKFold over the zero-filled
    # groups) is retained verbatim for transparency.
    if cv_spatial_results is None:
        payload_out["cv_spatial_macro_f1_mean"] = float("nan")
        payload_out["cv_spatial_macro_f1_std"] = 0.0
        payload_out["cv_spatial_per_fold_macro_f1"] = []
        payload_out["cv_spatial_n_unique_groups"] = 0
        payload_out["cv_spatial_splitter"] = ""
        payload_out["cv_spatial_fallback_triggered"] = True
        payload_out["cv_spatial_fallback_reason"] = "spatial CV not run"
    else:
        per_fold_f1 = [
            float(f["macro_f1"]) for f in cv_spatial_results.get("per_fold", [])
        ]
        payload_out["cv_spatial_macro_f1_mean"] = float(
            cv_spatial_results.get("cv_macro_f1_mean", float("nan"))
        )
        payload_out["cv_spatial_macro_f1_std"] = float(
            cv_spatial_results.get("cv_macro_f1_std", 0.0)
        )
        payload_out["cv_spatial_per_fold_macro_f1"] = per_fold_f1
        payload_out["cv_spatial_n_unique_groups"] = int(n_spatial_unique_tv)
        payload_out["cv_spatial_splitter"] = str(
            cv_spatial_results.get("splitter", "")
        )
        payload_out["cv_spatial_fallback_triggered"] = bool(
            cv_spatial_results.get("cv_fallback_triggered", False)
        )
        payload_out["cv_spatial_fallback_reason"] = str(
            cv_spatial_results.get("cv_fallback_reason", "")
        )
        payload_out["cv_spatial_per_fold"] = list(
            cv_spatial_results.get("per_fold", [])
        )

    save_model(model, args.model_out)
    write_metrics(payload_out, args.metrics_out)

    total = float(time.perf_counter() - t0)
    logger.info("done; total runtime=%.1fs", total)
    print(f"accuracy={metrics.accuracy:.4f}  macro_f1={metrics.macro_f1:.4f}")
    print(f"per-class recall: {metrics.per_class_recall}")
    print(f"per-class F1: {metrics.per_class_f1}")
    if cv_results is not None:
        print(
            f"cv: mean={cv_results['cv_macro_f1_mean']:.4f}  "
            f"std={cv_results['cv_macro_f1_std']:.4f}  "
            f"fallback={cv_results['cv_fallback_triggered']}"
        )
    if cv_spatial_results is not None:
        print(
            f"cv_spatial: mean={cv_spatial_results['cv_macro_f1_mean']:.4f}  "
            f"std={cv_spatial_results['cv_macro_f1_std']:.4f}  "
            f"n_unique_groups={n_spatial_unique_tv}  "
            f"splitter={cv_spatial_results['splitter']}  "
            f"fallback={cv_spatial_results['cv_fallback_triggered']}"
        )
    print(
        f"boundary: full={boundary['full_test_acc']:.4f}  "
        f"filtered={boundary['boundary_filtered_acc']:.4f}  "
        f"delta={boundary['delta_acc']:.4f}  "
        f"ci=[{boundary['delta_acc_ci_low']:.4f}, {boundary['delta_acc_ci_high']:.4f}]"
    )


if __name__ == "__main__":
    main()
