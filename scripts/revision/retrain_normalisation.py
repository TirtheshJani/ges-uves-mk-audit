#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Retrain the MK classifier under alternative continuum normalisations.

LightGBM is retrained with the production hyperparameters (read from the
pickled production model's ``get_params()``), the same train/val/test row
partition and validation-based early stopping (50 rounds on multi-logloss)
for four feature matrices:

  production            the deposited features unchanged
  polynomial_n5         every row (train, val and test) re-normalised with the
                        sigma-clipped 5th-order Legendre fit (fit over all
                        bins, gap bins divided by the fit, as in the audit
                        scripts)
  polynomial_n5_gap_restored
                        as above, then the gap bins reset to production values
                        so the gap region cannot carry the row's scale factor
  row_median            each row's off-gap bins divided by that row's own
                        off-gap median (gap bins unchanged)

For every retrained model: test per-class recall, macro-F1, confusion
matrix, best_iteration, and the masked-line ablation deltas for (Mg b, K)
and (Mg b, G) with the legacy settings (500 random controls, 500 bootstrap
resamples, constant fill equal to that matrix's off-gap test median).  For
the production matrix a 5-fold StratifiedKFold macro-F1 on train+val is
also reported, and the retrained test macro-F1 is compared with the
deposited model.

Output: artifacts/revision/retrain_normalisation.json
Run from the repo root: ``python scripts/revision/retrain_normalisation.py``
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.interpret.benchmark import continuum_normalize  # noqa: E402
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS  # noqa: E402
from src.interpret.occlusion import masked_line_ablation  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "F", 2: "G", 3: "K"}
CLASSES = (1, 2, 3)
SEED = 42
EARLY_STOPPING_ROUNDS = 50
DEPOSITED_MACRO_F1 = 0.9262


def polynomial_renormalise(X: np.ndarray, wave_centers: np.ndarray, order: int = 5) -> np.ndarray:
    out = np.empty_like(X)
    for i, row in enumerate(X):
        out[i] = continuum_normalize(
            wave_centers, row.astype(np.float64), method="polynomial_n5",
            polynomial_order=order,
        )
    return out


def row_median_normalise(X: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    out = X.copy()
    med = np.nanmedian(X[:, ~gap_mask], axis=1)
    out[:, ~gap_mask] = X[:, ~gap_mask] / med[:, None]
    return out


def fit_lgbm(params: dict, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def test_metrics(model, X_te, y_te) -> dict:
    pred = model.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=list(CLASSES))
    rec = recall_score(y_te, pred, labels=list(CLASSES), average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(pred == y_te)),
        "macro_f1": float(f1_score(y_te, pred, labels=list(CLASSES), average="macro")),
        "per_class_recall": {CLASS_NAMES[c]: float(r) for c, r in zip(CLASSES, rec)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": [CLASS_NAMES[c] for c in CLASSES],
        "g_to_k": int(cm[1, 2]),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else None,
    }


def ablation_rows(model, X_te, y_te, wave_centers, gap_mask, n_controls, n_boot) -> dict:
    fill = float(np.nanmedian(X_te[:, ~gap_mask]))
    draw_stats: dict = {}
    rows = masked_line_ablation(
        model, X_te, y_te, wave_centers,
        {"Mg_b": LINE_SETS["Mg_b"]},
        list(ALLOWED_MK_CLASSES),
        per_class=True,
        n_bootstrap=n_boot,
        n_random_controls=n_controls,
        seed=SEED,
        gap_mask=gap_mask,
        continuum_fill=fill,
        draw_stats_out=draw_stats,
    )
    out = {"continuum_fill": fill, "draw_stats": draw_stats}
    for r in rows:
        if r.line_set == "Mg_b" and r.mk_class in ("F", "G", "K"):
            out[f"Mg_b_{r.mk_class}"] = {
                "n_test": int(r.n_test),
                "baseline_recall": float(r.baseline_acc),
                "masked_recall": float(r.masked_acc_mean),
                "delta": float(r.delta_acc_mean),
                "delta_ci_low": float(r.delta_acc_ci_low),
                "delta_ci_high": float(r.delta_acc_ci_high),
                "p_value_vs_random": float(r.p_value_vs_random),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/retrain_normalisation.json"))
    p.add_argument("--n-random-controls", type=int, default=500)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--cv-folds", type=int, default=5)
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
    train_idx, val_idx, test_idx = fp["train_idx"], fp["val_idx"], fp["test_idx"]
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"].astype(bool)
    with args.model.open("rb") as f:
        deposited = pickle.load(f)
    params = dict(deposited.get_params())
    params["random_state"] = SEED
    params["class_weight"] = "balanced"
    deposited_test = test_metrics(deposited, X[test_idx], y[test_idx])

    logger.info("building polynomial_n5 matrix for all %d rows", len(X))
    X_poly = polynomial_renormalise(X, wave_centers)
    X_poly_gap = X_poly.copy()
    X_poly_gap[:, gap_mask] = X[:, gap_mask]
    X_rowmed = row_median_normalise(X, gap_mask)

    matrices = {
        "production": (X, "deposited features unchanged"),
        "polynomial_n5": (X_poly, "sigma-clipped 5th-order Legendre re-normalisation of every row; "
                                  "gap bins included in the fit and divided by it"),
        "polynomial_n5_gap_restored": (X_poly_gap, "polynomial_n5 with gap bins reset to production values"),
        "row_median": (X_rowmed, "off-gap bins divided by the row's own off-gap median; gap bins unchanged"),
    }

    results = {}
    for name, (Xm, desc) in matrices.items():
        logger.info("retraining on %s", name)
        model = fit_lgbm(params, Xm[train_idx], y[train_idx], Xm[val_idx], y[val_idx])
        res = {"transform": desc, "test": test_metrics(model, Xm[test_idx], y[test_idx])}
        res["test_global_median_off_gap"] = float(np.nanmedian(Xm[test_idx][:, ~gap_mask]))
        logger.info(
            "  %s: macroF1=%.4f recall=%s best_iter=%s", name, res["test"]["macro_f1"],
            {k: round(v, 3) for k, v in res["test"]["per_class_recall"].items()},
            res["test"]["best_iteration"],
        )
        logger.info("  masked-line ablation (Mg b) on %s", name)
        res["ablation_Mg_b"] = ablation_rows(
            model, Xm[test_idx], y[test_idx], wave_centers, gap_mask,
            args.n_random_controls, args.n_bootstrap,
        )
        for cls in ("K", "G"):
            a = res["ablation_Mg_b"].get(f"Mg_b_{cls}", {})
            logger.info("    (Mg b, %s): delta=%+.4f CI=[%+.4f, %+.4f] p=%.3f", cls,
                        a.get("delta", float("nan")), a.get("delta_ci_low", float("nan")),
                        a.get("delta_ci_high", float("nan")), a.get("p_value_vs_random", float("nan")))
        if name == "production":
            tv_idx = np.concatenate([train_idx, val_idx])
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
            fold_f1 = []
            for k, (tr, va) in enumerate(skf.split(Xm[tv_idx], y[tv_idx])):
                m_k = fit_lgbm(params, Xm[tv_idx][tr], y[tv_idx][tr], Xm[tv_idx][va], y[tv_idx][va])
                fold_f1.append(float(f1_score(y[tv_idx][va], m_k.predict(Xm[tv_idx][va]),
                                              labels=list(CLASSES), average="macro")))
                logger.info("  CV fold %d macroF1=%.4f best_iter=%s", k, fold_f1[-1], m_k.best_iteration_)
            res["cv_train_val"] = {
                "splitter": f"StratifiedKFold(n_splits={args.cv_folds}, shuffle=True, random_state={SEED})",
                "n_rows": int(len(tv_idx)),
                "per_fold_macro_f1": fold_f1,
                "macro_f1_mean": float(np.mean(fold_f1)),
                "macro_f1_sd": float(np.std(fold_f1, ddof=1)),
                "note": "each fold's held-out part doubles as the early-stopping set",
            }
            res["reproduces_deposited_model"] = {
                "deposited_macro_f1": deposited_test["macro_f1"],
                "deposited_best_iteration": deposited_test["best_iteration"],
                "retrained_macro_f1": res["test"]["macro_f1"],
                "retrained_best_iteration": res["test"]["best_iteration"],
                "abs_difference_macro_f1": abs(res["test"]["macro_f1"] - deposited_test["macro_f1"]),
                "identical_confusion": res["test"]["confusion_matrix"] == deposited_test["confusion_matrix"],
            }
        results[name] = res

    payload = {
        "description": (
            "LightGBM retrained with the production hyperparameters on the same "
            "train/val/test partition under four continuum normalisations, with "
            "Mg b masked-line ablation deltas for each retrained model."
        ),
        "inputs": {
            "features": str(args.features),
            "hyperparameter_source": str(args.model),
            "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                for k, v in params.items()},
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_metric": "multi_logloss on val rows",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_random_controls": int(args.n_random_controls),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": SEED,
            "polynomial_order": 5,
            "sigma_clip": 3.0,
            "sigma_clip_iters": 5,
            "line_set_Mg_b_aa": LINE_SETS["Mg_b"],
        },
        "deposited_model_test": deposited_test,
        "retrained": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
