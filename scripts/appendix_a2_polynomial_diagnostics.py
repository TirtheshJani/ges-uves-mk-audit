#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Appendix A2 diagnostics: ablation + confusion under polynomial_n5 continuum.

The multi-leg shortcut framing asserts more than the headline ablation
directly tests, so three additional analyses are run on the existing
artifacts:

  (a) Fe i / Cr i ablation under the polynomial_n5 continuum, per class.
      Does the K class still have non-Mg b dependence after the continuum
      signal is removed?
  (b) Full F / G / K confusion under polynomial_n5. G-class baseline
      collapses to 0.148; where do the predictions go?
  (c) F-class baseline under polynomial_n5 already computed in the
      continuum_shortcut_investigation diagnostic (drops to 0.728), but
      we also report Fe/Cr ablation on F for completeness.

Fe/Cr line set is defined here as the SHAP-identified K-class neutral-
metal multiplets cited in Section 4.2 of the manuscript:

  Cr i 5206.04 A (RMT 7)
  Cr i 5208.42 A (RMT 7)
  Fe i 5269.54 A (RMT 15)
  Cr i 5345.80 A (RMT 1)

Half-window of +/- 5 A around each gives the line set:

  (5201, 5215) Cr i g-multiplet at 5206/5208
  (5264, 5274) Fe i RMT 15 at 5269.54
  (5340, 5350) Cr i at 5345

Total width 30 A, comparable to Mg_b (21 A). The line set is added
post hoc; it is NOT part of the pre-
specified headline gate (Section 3.5) and we treat the resulting
ablation effects as confirmatory of the multi-leg framing rather than
as new headline claims.

Output: artifacts/sensitivity/appendix_a2_polynomial_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.interpret.benchmark import _continuum_polynomial_clipped
from src.interpret.classifier import load_model
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.ablation import masked_line_ablation

logger = logging.getLogger(__name__)

FE_CR_LINE_SET = [
    (5201.0, 5215.0),
    (5264.0, 5274.0),
    (5340.0, 5350.0),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/sensitivity/appendix_a2_polynomial_diagnostics.json"),
    )
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

    logger.info(
        "renormalising %d test rows with polynomial_n%d",
        len(X_test), args.polynomial_order,
    )
    X_renorm = np.empty_like(X_test)
    for i, row in enumerate(X_test):
        X_renorm[i] = _continuum_polynomial_clipped(
            wave_centers, row.astype(np.float64), order=args.polynomial_order,
        )

    yhat_prod = model.predict(X_test)
    yhat_renorm = model.predict(X_renorm)

    class_names = {1: "F", 2: "G", 3: "K"}

    per_class_baseline = {}
    for cls in (1, 2, 3):
        m = y_test == cls
        per_class_baseline[class_names[cls]] = {
            "n_test": int(m.sum()),
            "production_baseline_acc": float((yhat_prod[m] == y_test[m]).mean()),
            "polynomial_n5_baseline_acc": float((yhat_renorm[m] == y_test[m]).mean()),
            "delta_baseline_acc": float(
                (yhat_renorm[m] == y_test[m]).mean() - (yhat_prod[m] == y_test[m]).mean()
            ),
        }

    fgk_labels = [1, 2, 3]
    cm_prod = confusion_matrix(y_test, yhat_prod, labels=fgk_labels).tolist()
    cm_renorm = confusion_matrix(y_test, yhat_renorm, labels=fgk_labels).tolist()

    line_sets_to_run = {
        "H_balmer": LINE_SETS["H_balmer"],
        "Mg_b": LINE_SETS["Mg_b"],
        "Na_D": LINE_SETS["Na_D"],
        "Ca_I": LINE_SETS["Ca_I"],
        "Fe_Cr_K_multiplets": FE_CR_LINE_SET,
    }

    def _ablation_payload(X_input, label):
        logger.info(
            "running ablation on %s (n_test=%d, n_line_sets=%d)",
            label, len(X_input), len(line_sets_to_run),
        )
        rows = masked_line_ablation(
            model, X_input, y_test, wave_centers,
            line_sets_to_run, list(ALLOWED_MK_CLASSES),
            per_class=True,
            n_bootstrap=args.n_bootstrap,
            n_random_controls=args.n_random_controls,
            seed=args.seed,
            gap_mask=gap_mask,
            continuum_fill=float(np.nanmedian(X_input[:, ~gap_mask])),
            # Legacy reference-set configuration; reproduces the deposited artifact.
            null_mode="pooled",
            match_on="angstrom",
        )
        out = {}
        for r in rows:
            out.setdefault(r.line_set, {})[r.mk_class] = {
                "n_test": int(r.n_test),
                "baseline_acc": float(r.baseline_acc),
                "masked_acc_mean": float(r.masked_acc_mean),
                "delta_acc_mean": float(r.delta_acc_mean),
                "delta_acc_ci_low": float(r.delta_acc_ci_low),
                "delta_acc_ci_high": float(r.delta_acc_ci_high),
                "p_value_vs_random": float(r.p_value_vs_random),
            }
        return out

    ablation_production = _ablation_payload(X_test, "production")
    ablation_polynomial = _ablation_payload(X_renorm, "polynomial_n5")

    payload = {
        "round": "post-hoc follow-up (appendix A2)",
        "description": (
            "Fe/Cr ablation plus per-class baseline accuracy plus full F/G/K "
            "confusion under polynomial_n5 re-derivation of the continuum. "
            "Production-continuum analogues reported alongside for comparison."
        ),
        "fe_cr_line_set_aa": FE_CR_LINE_SET,
        "fe_cr_line_set_provenance": (
            "Cr i 5206.04 (RMT 7), Cr i 5208.42 (RMT 7), Fe i 5269.54 "
            "(RMT 15), Cr i 5345.80 (RMT 1); half-window +/-5 A. "
            "Post-hoc diagnostic; not part of pre-specified "
            "headline gate (Section 3.5)."
        ),
        "seed": int(args.seed),
        "n_bootstrap": int(args.n_bootstrap),
        "n_random_controls": int(args.n_random_controls),
        "polynomial_order": int(args.polynomial_order),
        "per_class_baseline": per_class_baseline,
        "confusion_matrix_production": cm_prod,
        "confusion_matrix_polynomial_n5": cm_renorm,
        "confusion_matrix_labels": ["F", "G", "K"],
        "ablation_under_production_continuum": ablation_production,
        "ablation_under_polynomial_n5_continuum": ablation_polynomial,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)

    logger.info("=== Per-class baseline under polynomial_n5 ===")
    for cls in ("F", "G", "K"):
        b = per_class_baseline[cls]
        logger.info(
            "  %s: production=%.4f  polynomial=%.4f  delta=%+.4f  n=%d",
            cls, b["production_baseline_acc"], b["polynomial_n5_baseline_acc"],
            b["delta_baseline_acc"], b["n_test"],
        )

    logger.info("=== Confusion under polynomial_n5 (rows=true, cols=pred F/G/K) ===")
    for i, name in enumerate(("F", "G", "K")):
        logger.info(" %s: %s", name, cm_renorm[i])

    logger.info("=== Fe/Cr ablation: production vs polynomial ===")
    for ls in ("Fe_Cr_K_multiplets",):
        for cls in ("F", "G", "K"):
            prod = ablation_production.get(ls, {}).get(cls)
            poly = ablation_polynomial.get(ls, {}).get(cls)
            if prod and poly:
                logger.info(
                    "  %s, %s: production dA=%+.4f p=%.4f  |  polynomial dA=%+.4f p=%.4f",
                    ls, cls, prod["delta_acc_mean"], prod["p_value_vs_random"],
                    poly["delta_acc_mean"], poly["p_value_vs_random"],
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
