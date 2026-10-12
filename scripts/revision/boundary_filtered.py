#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Test accuracy away from the class boundaries at two distance cuts.

``scripts/train_classifier.py`` reports the held-out accuracy on the test rows
whose effective temperature lies more than 200 K from the nearest edge of their
own class bin (``artifacts/metrics.json:boundary_filtered_accuracy``). The
stratified ablation of ``scripts/revision/ablation_stratified.py`` uses a 150 K
cut. This script reports the same accuracy at both cuts from the deposited
model and feature matrix so that the manuscript can quote them side by side.

The 200 K row must reproduce the deposited value exactly (same rows, same
model); the script asserts that. The bootstrap interval on the accuracy
difference uses the same convention as the deposited one (test rows resampled
with replacement, 2000 resamples, seed 42, percentile interval).

Inputs:  artifacts/features.npz, artifacts/lgbm_mk.pkl, artifacts/metrics.json
Output:  artifacts/revision/boundary_filtered.json

Run from the repository root: ``python scripts/revision/boundary_filtered.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.classifier import load_model  # noqa: E402

CUTS_K = (150.0, 200.0)
N_BOOTSTRAP = 2000
SEED = 42


def main() -> int:
    fp = np.load(REPO_ROOT / "artifacts/features.npz", allow_pickle=False)
    test_idx = fp["test_idx"]
    X_test = fp["X"][test_idx]
    y_test = fp["y"][test_idx].astype(np.int64)
    dist = fp["boundary_distance_k"][test_idx].astype(np.float64)
    model = load_model(REPO_ROOT / "artifacts/lgbm_mk.pkl")
    y_pred = np.asarray(model.predict(X_test)).astype(np.int64)
    correct = y_pred == y_test
    full_acc = float(correct.mean())

    with (REPO_ROOT / "artifacts/metrics.json").open(encoding="utf-8") as f:
        met = json.load(f)
    dep = met["boundary_filtered_accuracy"]

    rng = np.random.default_rng(SEED)
    out: dict = {
        "description": (
            "Held-out test accuracy on rows whose Teff lies more than the cut from the nearest "
            "edge of their own class bin, at the 150 K cut of the stratified ablation and the "
            "200 K cut of the deposited classifier report, from the deposited model."
        ),
        "inputs": {
            "features": "artifacts/features.npz",
            "model": "artifacts/lgbm_mk.pkl",
            "distance_field": "boundary_distance_k (distance to the nearest edge of the row's own bin)",
            "n_test": int(len(test_idx)),
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "full_test_accuracy": full_acc,
        "cuts": {},
    }
    for cut in CUTS_K:
        far = dist > cut
        n_far = int(far.sum())
        acc_far = float(correct[far].mean())
        deltas = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(y_test), size=len(y_test))
            c_b, f_b = correct[idx], far[idx]
            deltas[b] = c_b[f_b].mean() - c_b.mean() if f_b.any() else np.nan
        deltas = deltas[np.isfinite(deltas)]
        out["cuts"][f"{int(cut)}K"] = {
            "threshold_k": cut,
            "n_filtered": n_far,
            "n_within": int(len(y_test) - n_far),
            "accuracy_filtered": acc_far,
            "accuracy_within": float(correct[~far].mean()) if (~far).any() else None,
            "delta_acc": acc_far - full_acc,
            "delta_acc_ci_low": float(np.percentile(deltas, 2.5)),
            "delta_acc_ci_high": float(np.percentile(deltas, 97.5)),
        }
    rep = out["cuts"]["200K"]
    out["reproduction_of_deposited_200K"] = {
        "deposited_accuracy": dep["boundary_filtered_acc"],
        "deposited_n_filtered": dep["n_filtered"],
        "recomputed_accuracy": rep["accuracy_filtered"],
        "recomputed_n_filtered": rep["n_filtered"],
        "accuracy_matches": abs(rep["accuracy_filtered"] - dep["boundary_filtered_acc"]) < 1e-12,
        "n_matches": rep["n_filtered"] == int(dep["n_filtered"]),
    }
    if not (out["reproduction_of_deposited_200K"]["accuracy_matches"] and out["reproduction_of_deposited_200K"]["n_matches"]):
        raise SystemExit("200 K cut does not reproduce artifacts/metrics.json:boundary_filtered_accuracy")
    out_path = REPO_ROOT / "artifacts/revision/boundary_filtered.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for cut, d in out["cuts"].items():
        print(f"{cut}: n={d['n_filtered']} accuracy={d['accuracy_filtered']:.4f} delta={d['delta_acc']:+.4f} "
              f"[{d['delta_acc_ci_low']:+.4f}, {d['delta_acc_ci_high']:+.4f}]")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
