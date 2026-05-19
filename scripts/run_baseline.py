#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Non-ML baseline drivers (kNN and Pickles template-fit) for the test set.

. Reviewer-moderate: a non-ML baseline scopes the audit's
substantive interest. If LightGBM beats kNN by > 0.05 macro-F1, the audit
remains methodologically valuable. If the gap is smaller, the manuscript
framing widens to "audit instrument independent of classifier performance".

Outputs:
  - artifacts/baseline/knn_baseline.json
  - artifacts/baseline/pickles_template_baseline.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.interpret.baseline import (
    knn_predict,
    macro_f1,
    pickles_template_predict,
)
from src.interpret.lines import ALLOWED_MK_CLASSES

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--pickles-dir", type=Path, default=Path("data/pickles"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/baseline"))
    p.add_argument(
        "--knn-k",
        type=int,
        nargs="+",
        default=[1, 5, 11],
        help="k values to try for kNN-on-spectra",
    )
    p.add_argument(
        "--pickles-continuum-method",
        choices=("median_filter_200", "median_filter_50", "polynomial_n5"),
        default="polynomial_n5",
        help="continuum method for the Pickles template-fit baseline",
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # kNN baseline: train on train+val (matches final-model convention),
    # evaluate on test.
    trainval = np.concatenate([train_idx, val_idx])
    X_train = X[trainval]
    y_train = y[trainval]
    X_test = X[test_idx]
    y_test = y[test_idx]
    class_labels = sorted(np.unique(y_train).tolist())
    int_to_name = {i + 1: c for i, c in enumerate(ALLOWED_MK_CLASSES[1:])}
    # ALLOWED_MK_CLASSES is ("A","F","G","K"); y uses 1..3 for F,G,K (A=0 was dropped).
    # Convert label int -> letter for readable output.
    label_letter = {1: "F", 2: "G", 3: "K"}

    logger.info(
        "kNN baseline: n_train=%d, n_test=%d, k=%s, class_labels=%s",
        len(trainval), len(test_idx), args.knn_k, class_labels,
    )
    knn_preds = knn_predict(X_train, y_train, X_test, k_values=args.knn_k)
    knn_results: dict[str, dict] = {}
    for k, preds in knn_preds.items():
        f1 = macro_f1(y_test, preds, class_labels)
        accuracy = float(np.mean(preds == y_test))
        per_class_acc = {
            label_letter[c]: float(np.mean(preds[y_test == c] == c))
            for c in class_labels
            if (y_test == c).any()
        }
        knn_results[f"k={k}"] = {
            "k": int(k),
            "macro_f1_test": float(f1),
            "accuracy_test": accuracy,
            "per_class_recall_test": per_class_acc,
            "n_test": int(len(test_idx)),
        }
        logger.info(
            "  kNN k=%d: macro_f1=%.4f acc=%.4f per_class_recall=%s",
            k, f1, accuracy, per_class_acc,
        )
    knn_payload = {
        "n_train": int(len(trainval)),
        "n_test": int(len(test_idx)),
        "k_values": args.knn_k,
        "results_by_k": knn_results,
        "lightgbm_macro_f1_test_reference": "see artifacts/metrics.json $.macro_f1",
        "distance_metric": "1 - pearson_correlation",
    }
    knn_path = args.out_dir / "knn_baseline.json"
    with knn_path.open("w") as f:
        json.dump(knn_payload, f, indent=2)
    logger.info("wrote %s", knn_path)

    # Pickles template-fit baseline: chi-squared on FGK templates only.
    logger.info(
        "Pickles template-fit baseline: continuum_method=%s",
        args.pickles_continuum_method,
    )
    y_pred_str, pickles_stats = pickles_template_predict(
        X_test, wave_centers,
        pickles_dir=args.pickles_dir,
        continuum_method=args.pickles_continuum_method,
        restrict_to_fgk=True,
    )
    letter_to_int = {v: k for k, v in label_letter.items()}
    y_pred_int = np.array(
        [letter_to_int.get(s, -1) for s in y_pred_str], dtype=np.int64
    )
    f1_pkl = macro_f1(y_test, y_pred_int, class_labels)
    acc_pkl = float(np.mean(y_pred_int == y_test))
    per_class_acc_pkl = {
        label_letter[c]: float(np.mean(y_pred_int[y_test == c] == c))
        for c in class_labels
        if (y_test == c).any()
    }
    pickles_payload = {
        "n_test": int(len(test_idx)),
        "macro_f1_test": float(f1_pkl),
        "accuracy_test": acc_pkl,
        "per_class_recall_test": per_class_acc_pkl,
        **pickles_stats,
    }
    pickles_path = args.out_dir / "pickles_template_baseline.json"
    with pickles_path.open("w") as f:
        json.dump(pickles_payload, f, indent=2)
    logger.info("wrote %s", pickles_path)
    logger.info(
        "  Pickles template-fit: macro_f1=%.4f acc=%.4f per_class_recall=%s",
        f1_pkl, acc_pkl, per_class_acc_pkl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
