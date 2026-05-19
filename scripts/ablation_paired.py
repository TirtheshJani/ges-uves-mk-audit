#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Paired bootstrap, flip rate, and TOST equivalence on per-row ablation outputs.

Steps 2a (paired bootstrap), 2b (flip rate), 2c (TOST equivalence).
Consumes ``artifacts/ablation/per_row_predictions.npz`` produced by
``scripts/ablation.py`` (). The three analyses share the
same per-row baseline and masked predictions so they live in one driver.

, #2c (TOST equivalence), and #4 (paired
bootstrap on the substantive between-class difference) are addressed by
this script's three output JSONs.

Outputs:
  - artifacts/ablation/paired_bootstrap.json: Step 2a result
  - artifacts/ablation/flip_rates.json: Step 2b result
  - artifacts/ablation/tost.json: Step 2c result

The class label encoding is y=1 -> F, y=2 -> G, y=3 -> K (see
src/interpret/labels.py).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAME_TO_LABEL: dict[str, int] = {"F": 1, "G": 2, "K": 3}
TOST_INDIFFERENCE_ZONE: float = 0.025  # , locked 2026-05-16


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    mask = y_true == c
    if not mask.any():
        return float("nan")
    return float(np.mean(y_pred[mask] == c))


def paired_bootstrap_delta_difference(
    y_test: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_masked: np.ndarray,
    class_a: int,
    class_b: int,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI on delta_acc(class_a) - delta_acc(class_b) under the same masking.

    Resamples within each class independently with replacement, computes
    delta_acc per class on the resample, returns mean/CI/p-value on the
    difference.

    p-value tests H0: delta_a - delta_b >= 0 (one-sided). Smaller p means
    class_a's effect is more negative than class_b's by a larger margin.
     the substantive claim is "the model depends on the
    line set for class_a and does not for class_b", so class_a should be
    the class with the larger predicted effect.
    """
    rng = np.random.default_rng(seed)
    idx_a = np.flatnonzero(y_test == class_a)
    idx_b = np.flatnonzero(y_test == class_b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(f"empty support for class {class_a} or {class_b}")
    base_a = _per_class_accuracy(y_test, y_pred_base, class_a)
    masked_a = _per_class_accuracy(y_test, y_pred_masked, class_a)
    base_b = _per_class_accuracy(y_test, y_pred_base, class_b)
    masked_b = _per_class_accuracy(y_test, y_pred_masked, class_b)
    delta_a_observed = masked_a - base_a
    delta_b_observed = masked_b - base_b
    diff_observed = delta_a_observed - delta_b_observed

    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        ai = rng.choice(idx_a, size=len(idx_a), replace=True)
        bi = rng.choice(idx_b, size=len(idx_b), replace=True)
        d_a = float(np.mean(y_pred_masked[ai] == class_a)) - float(
            np.mean(y_pred_base[ai] == class_a)
        )
        d_b = float(np.mean(y_pred_masked[bi] == class_b)) - float(
            np.mean(y_pred_base[bi] == class_b)
        )
        diffs[b] = d_a - d_b
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    # One-sided p-value: fraction of bootstrap diffs >= 0 (testing H0 that
    # class_a's effect is no more negative than class_b's). With (B+1)
    # Phipson-Smyth correction.
    n_ge_zero = int(np.sum(diffs >= 0))
    p_one_sided = (n_ge_zero + 1) / (n_bootstrap + 1)
    return {
        "n_a": int(len(idx_a)),
        "n_b": int(len(idx_b)),
        "delta_a_observed": float(delta_a_observed),
        "delta_b_observed": float(delta_b_observed),
        "diff_observed": float(diff_observed),
        "diff_mean_bootstrap": float(diffs.mean()),
        "diff_ci_low_95": ci_low,
        "diff_ci_high_95": ci_high,
        "p_one_sided_diff_ge_zero": float(p_one_sided),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def flip_rate(
    y_test: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_masked: np.ndarray,
    c: int,
) -> dict[str, float | int]:
    """Compute per-class flip-rate decomposition under masking.

    identical aggregate accuracy may hide a balanced
    flip pattern. Reports n_correct_to_incorrect, n_incorrect_to_correct,
    n_flipped_total, n_total, and the rates.
    """
    mask = y_test == c
    n_total = int(mask.sum())
    if n_total == 0:
        return {
            "class_label": int(c),
            "n_total": 0,
            "n_correct_to_incorrect": 0,
            "n_incorrect_to_correct": 0,
            "n_unchanged_correct": 0,
            "n_unchanged_incorrect": 0,
            "n_flipped_total": 0,
            "flip_rate_total": float("nan"),
            "flip_rate_correct_to_incorrect": float("nan"),
            "flip_rate_incorrect_to_correct": float("nan"),
        }
    yt = y_test[mask]
    yb = y_pred_base[mask]
    ym = y_pred_masked[mask]
    base_correct = yb == yt
    masked_correct = ym == yt
    n_ci = int(np.sum(base_correct & ~masked_correct))
    n_ic = int(np.sum(~base_correct & masked_correct))
    n_uc = int(np.sum(base_correct & masked_correct))
    n_ui = int(np.sum(~base_correct & ~masked_correct))
    n_flipped = int(np.sum(yb != ym))
    return {
        "class_label": int(c),
        "n_total": n_total,
        "n_correct_to_incorrect": n_ci,
        "n_incorrect_to_correct": n_ic,
        "n_unchanged_correct": n_uc,
        "n_unchanged_incorrect": n_ui,
        "n_flipped_total": n_flipped,
        "flip_rate_total": float(n_flipped / n_total),
        "flip_rate_correct_to_incorrect": float(n_ci / n_total),
        "flip_rate_incorrect_to_correct": float(n_ic / n_total),
    }


def tost_equivalence(
    y_test: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_masked: np.ndarray,
    c: int,
    indifference_zone: float = TOST_INDIFFERENCE_ZONE,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Two one-sided t-test equivalence for per-class delta_acc.

    a CI of approximately +/-0.02 with n=145 is not the
    same as a TOST rejection of inequivalence; we test whether delta_acc
    lies entirely inside the +/-indifference_zone band.

    Procedure (bootstrap analogue of Schuirmann 1987 / Lakens 2017):
      - Bootstrap distribution of delta_acc_c from 500 resamples.
      - Compute 90 percent CI (5th, 95th percentiles): if both bounds in
        [-zone, +zone] -> reject inequivalence at alpha=0.05.
      - One-sided p-values: p_low = fraction of bootstrap deltas <= -zone,
        p_high = fraction >= +zone. p_tost = max(p_low, p_high). If
        p_tost < 0.05 -> equivalence established (Phipson-Smyth correction
        applied).
    """
    rng = np.random.default_rng(seed)
    mask = y_test == c
    if not mask.any():
        raise ValueError(f"empty support for class {c}")
    idx = np.flatnonzero(mask)
    base_acc = float(np.mean(y_pred_base[idx] == c))
    masked_acc = float(np.mean(y_pred_masked[idx] == c))
    delta_observed = masked_acc - base_acc

    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        bi = rng.choice(idx, size=len(idx), replace=True)
        boot[b] = float(np.mean(y_pred_masked[bi] == c)) - float(
            np.mean(y_pred_base[bi] == c)
        )
    ci90_low = float(np.percentile(boot, 5.0))
    ci90_high = float(np.percentile(boot, 95.0))
    ci95_low = float(np.percentile(boot, 2.5))
    ci95_high = float(np.percentile(boot, 97.5))
    n_le_neg = int(np.sum(boot <= -indifference_zone))
    n_ge_pos = int(np.sum(boot >= indifference_zone))
    p_low = (n_le_neg + 1) / (n_bootstrap + 1)
    p_high = (n_ge_pos + 1) / (n_bootstrap + 1)
    p_tost = max(p_low, p_high)
    ci_inside = (-indifference_zone <= ci90_low) and (ci90_high <= indifference_zone)
    p_inside = p_tost < 0.05
    if ci_inside and p_inside:
        verdict = "EQUIVALENT"
    elif not ci_inside and not p_inside:
        verdict = "INEQUIVALENT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "class_label": int(c),
        "n_test": int(mask.sum()),
        "delta_observed": float(delta_observed),
        "indifference_zone": float(indifference_zone),
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p_lower_bound": float(p_low),
        "p_upper_bound": float(p_high),
        "p_tost": float(p_tost),
        "ci_inside_zone": bool(ci_inside),
        "p_tost_significant": bool(p_inside),
        "verdict": verdict,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _line_sets_in_payload(pr: dict) -> list[str]:
    return sorted(
        k.removeprefix("y_pred_masked__")
        for k in pr.keys()
        if k.startswith("y_pred_masked__")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-row",
        type=Path,
        default=Path("artifacts/ablation/per_row_predictions.npz"),
        help="path to per_row_predictions.npz ()",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/ablation"),
    )
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tost-indifference-zone",
        type=float,
        default=TOST_INDIFFERENCE_ZONE,
        help="default is 0.025; override here for sensitivity probes",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pr = np.load(args.per_row, allow_pickle=False)
    payload = {k: pr[k] for k in pr.files}
    y_test = payload["y_test"].astype(np.int64)
    y_pred_base = payload["y_pred_base"].astype(np.int64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    line_sets = _line_sets_in_payload(payload)
    logger.info("loaded per-row payload with line sets: %s", line_sets)

    # --- Step 2a: paired bootstrap on (Mg b, G) vs (Mg b, K) ---
    if "y_pred_masked__Mg_b" not in payload:
        raise SystemExit("payload missing y_pred_masked__Mg_b; cannot run Step 2a")
    y_pred_masked_mgb = payload["y_pred_masked__Mg_b"].astype(np.int64)
    paired = paired_bootstrap_delta_difference(
        y_test, y_pred_base, y_pred_masked_mgb,
        class_a=CLASS_NAME_TO_LABEL["G"],
        class_b=CLASS_NAME_TO_LABEL["K"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    paired_payload = {
        "line_set": "Mg_b",
        "class_a": "G",
        "class_b": "K",
        "indifference_zone_decision_39": float(args.tost_indifference_zone),
        **paired,
    }
    paired_path = args.out_dir / "paired_bootstrap.json"
    with paired_path.open("w") as f:
        json.dump(paired_payload, f, indent=2)
    logger.info("wrote %s", paired_path)
    logger.info(
        "  delta_G=%+.4f  delta_K=%+.4f  diff=%+.4f  CI=[%+.4f,%+.4f]  p_one_sided=%.4f",
        paired["delta_a_observed"], paired["delta_b_observed"],
        paired["diff_observed"], paired["diff_ci_low_95"],
        paired["diff_ci_high_95"], paired["p_one_sided_diff_ge_zero"],
    )

    # --- Step 2b: flip rate for every (line set, class) pair ---
    flips: list[dict] = []
    label_to_name = {v: k for k, v in CLASS_NAME_TO_LABEL.items()}
    for set_name in line_sets:
        y_pred_masked = payload[f"y_pred_masked__{set_name}"].astype(np.int64)
        for label in (1, 2, 3):
            row = flip_rate(y_test, y_pred_base, y_pred_masked, label)
            row["line_set"] = set_name
            row["mk_class"] = label_to_name[label]
            flips.append(row)
    flip_payload = {
        "n_line_sets": len(line_sets),
        "n_pairs": len(flips),
        "rows": flips,
    }
    flip_path = args.out_dir / "flip_rates.json"
    with flip_path.open("w") as f:
        json.dump(flip_payload, f, indent=2)
    logger.info("wrote %s (%d rows)", flip_path, len(flips))

    # --- Step 2c: TOST equivalence on (Mg b, K), with companion runs on the
    # two pivot pairs for symmetry ---
    tost_results: list[dict] = []
    headline_tost = tost_equivalence(
        y_test, y_pred_base, y_pred_masked_mgb,
        c=CLASS_NAME_TO_LABEL["K"],
        indifference_zone=args.tost_indifference_zone,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    headline_tost["line_set"] = "Mg_b"
    headline_tost["mk_class"] = "K"
    headline_tost["headline"] = True
    tost_results.append(headline_tost)
    for set_name in ("Na_D", "Ca_I"):
        if f"y_pred_masked__{set_name}" in payload:
            y_pred_masked = payload[f"y_pred_masked__{set_name}"].astype(np.int64)
            res = tost_equivalence(
                y_test, y_pred_base, y_pred_masked,
                c=CLASS_NAME_TO_LABEL["K"],
                indifference_zone=args.tost_indifference_zone,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            res["line_set"] = set_name
            res["mk_class"] = "K"
            res["headline"] = False
            tost_results.append(res)
    tost_payload = {
        "indifference_zone": float(args.tost_indifference_zone),
        "decision_39_default": float(TOST_INDIFFERENCE_ZONE),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "results": tost_results,
    }
    tost_path = args.out_dir / "tost.json"
    with tost_path.open("w") as f:
        json.dump(tost_payload, f, indent=2)
    logger.info("wrote %s (%d pairs)", tost_path, len(tost_results))
    for r in tost_results:
        logger.info(
            "  TOST (%s, %s): verdict=%s  delta=%+.4f  90CI=[%+.4f,%+.4f]  p_tost=%.4f",
            r["line_set"], r["mk_class"], r["verdict"],
            r["delta_observed"], r["ci90_low"], r["ci90_high"], r["p_tost"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
