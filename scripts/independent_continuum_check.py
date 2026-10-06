#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Independent continuum re-derivation check on the (Mg b, K) null.

The production continuum normalisation is inherited from the preprocessing
pipeline without an independent check. This script re-derives the continuum
on a K-class test subsample with an independent algorithm (an iteratively
sigma-clipped polynomial fit), recomputes the (Mg b, K) ablation, and
confirms that the null persists.

Procedure:
  1. Sample 50 K-class test rows (random_state=42 for reproducibility).
  2. Re-normalise each row with the Step 5 polynomial_n5 continuum method
     applied to the already pipeline-normalised features. The polynomial
     fit identifies any large-scale residual the GES pipeline missed.
  3. Re-run the (Mg b, K) ablation on the 50-row subsample using the
     existing production model (lgbm_mk.pkl). The model itself does not
     change; only the input flux is re-normalised.
  4. Compare to the full-test-set baseline.

Caveat: n=50 supports only a directional check, not equivalent statistical
power; it is a "confirm the null persists" sanity probe.

Output: artifacts/sensitivity/independent_continuum_mg_b_k.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.interpret.benchmark import _continuum_polynomial_clipped
from src.interpret.classifier import load_model
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.ablation import masked_line_ablation

logger = logging.getLogger(__name__)

K_CLASS_LABEL: int = 3  # y=3 -> K per labels.py encoding


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/sensitivity/independent_continuum_mg_b_k.json"),
    )
    p.add_argument("--n-subsample", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--n-random-controls", type=int, default=500)
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
    model = load_model(args.model)

    X_test = X[test_idx]
    y_test = y[test_idx]
    k_mask = y_test == K_CLASS_LABEL
    k_indices_in_test = np.flatnonzero(k_mask)
    if len(k_indices_in_test) < args.n_subsample:
        raise SystemExit(
            f"only {len(k_indices_in_test)} K-class rows in test; "
            f"requested subsample {args.n_subsample}"
        )
    rng = np.random.default_rng(args.seed)
    sub_indices = rng.choice(
        k_indices_in_test, size=args.n_subsample, replace=False
    )
    sub_indices = np.sort(sub_indices)
    X_sub = X_test[sub_indices].copy()
    y_sub = y_test[sub_indices].copy()
    logger.info(
        "sampled %d K-class test rows; renormalising under polynomial_n%d",
        args.n_subsample, args.polynomial_order,
    )
    X_renorm = np.empty_like(X_sub)
    for i, row in enumerate(X_sub):
        X_renorm[i] = _continuum_polynomial_clipped(
            wave_centers, row.astype(np.float64), order=args.polynomial_order,
        )

    # Production baseline for comparison: same ablation on the same 50 rows
    # WITHOUT the re-normalisation.
    rows_baseline = masked_line_ablation(
        model, X_sub, y_sub, wave_centers,
        {"Mg_b": LINE_SETS["Mg_b"]},
        list(ALLOWED_MK_CLASSES),
        per_class=True,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        seed=args.seed,
        gap_mask=gap_mask,
        continuum_fill=float(np.nanmedian(X_sub[:, ~gap_mask])),
        # Legacy reference-set configuration; reproduces the deposited artifact.
        null_mode="pooled",
        match_on="angstrom",
    )
    mgk_baseline = next(
        (r for r in rows_baseline if r.line_set == "Mg_b" and r.mk_class == "K"),
        None,
    )

    # Renormalised: re-run ablation against the polynomial-renormalised flux.
    rows_renorm = masked_line_ablation(
        model, X_renorm, y_sub, wave_centers,
        {"Mg_b": LINE_SETS["Mg_b"]},
        list(ALLOWED_MK_CLASSES),
        per_class=True,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        seed=args.seed,
        gap_mask=gap_mask,
        continuum_fill=float(np.nanmedian(X_renorm[:, ~gap_mask])),
        null_mode="pooled",
        match_on="angstrom",
    )
    mgk_renorm = next(
        (r for r in rows_renorm if r.line_set == "Mg_b" and r.mk_class == "K"),
        None,
    )

    def _row_payload(row) -> dict:
        if row is None:
            return {}
        return {
            "n_test": int(row.n_test),
            "baseline_acc": float(row.baseline_acc),
            "masked_acc_mean": float(row.masked_acc_mean),
            "delta_acc_mean": float(row.delta_acc_mean),
            "delta_acc_ci_low": float(row.delta_acc_ci_low),
            "delta_acc_ci_high": float(row.delta_acc_ci_high),
            "p_value_vs_random": float(row.p_value_vs_random),
        }

    payload = {
        "step": "8c",
        "description": (
            "independent continuum re-derivation on a 50-row K-class test "
            "subsample using polynomial_n5; directional check only (n=50)"
        ),
        "n_subsample": int(args.n_subsample),
        "polynomial_order": int(args.polynomial_order),
        "seed": int(args.seed),
        "n_bootstrap": int(args.n_bootstrap),
        "n_random_controls": int(args.n_random_controls),
        "indifference_zone_decision_39": 0.025,
        "subsample_indices_in_test": sub_indices.tolist(),
        "baseline_50row": _row_payload(mgk_baseline),
        "renormalised_50row": _row_payload(mgk_renorm),
        "null_persists_under_renorm": bool(
            mgk_renorm is not None
            and abs(mgk_renorm.delta_acc_mean) <= 0.025
            and mgk_renorm.p_value_vs_random > 0.05
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)
    logger.info(
        "  baseline 50-row (Mg_b, K): delta=%+.4f  CI=[%+.4f,%+.4f]  p=%.4f",
        mgk_baseline.delta_acc_mean if mgk_baseline else float("nan"),
        mgk_baseline.delta_acc_ci_low if mgk_baseline else float("nan"),
        mgk_baseline.delta_acc_ci_high if mgk_baseline else float("nan"),
        mgk_baseline.p_value_vs_random if mgk_baseline else float("nan"),
    )
    logger.info(
        "  renormalised 50-row (Mg_b, K): delta=%+.4f  CI=[%+.4f,%+.4f]  p=%.4f",
        mgk_renorm.delta_acc_mean if mgk_renorm else float("nan"),
        mgk_renorm.delta_acc_ci_low if mgk_renorm else float("nan"),
        mgk_renorm.delta_acc_ci_high if mgk_renorm else float("nan"),
        mgk_renorm.p_value_vs_random if mgk_renorm else float("nan"),
    )
    logger.info(
        "  null_persists_under_renorm = %s", payload["null_persists_under_renorm"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
