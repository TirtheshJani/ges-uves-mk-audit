#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Masked-line ablation driver: CSV of deltas, bootstrap CIs, Monte-Carlo p-values and paired flip statistics.

Consumes the ``gap_mask`` key written by ``build_features.py`` to forbid
the random-window sampler from drawing windows that land in the UVES
inter-chip gap. The headline gate evaluates the three pre-specified FGK
pairs (H_balmer, F), (Mg_b, G), (Mg_b, K). Two pivot pairs are reported
separately as confirmatory: (Na_D, K), (Ca_I, K).

A continuum-fill sensitivity pass is run on the headline pairs only,
comparing CONTINUUM_FILL=1.0 against the empirical median of the test
feature matrix outside the gap mask, and a fill-mode pass compares the
constant, noise and linear-interpolation fills.

Reference-set configuration (see ``src.interpret.ablation``):

  --null-mode class_matched (default) compares each class's observed recall
      delta against that class's own random-window recall deltas;
      --null-mode pooled compares against the all-class accuracy deltas.
  --match-on bins (default) draws random windows with the line set's exact
      per-segment bin counts, mutually non-overlapping; --match-on angstrom
      matches total width in Angstrom with equal-width segments.

Legacy invocation that reproduces the deposited ``artifacts/ablation``
outputs (pooled accuracy reference, Angstrom-matched windows, 500 random
windows, 500 bootstrap resamples, constant empirical-median fill):

    python scripts/ablation.py --features artifacts/features.npz \
        --model artifacts/lgbm_mk.pkl --out-dir artifacts/ablation \
        --null-mode pooled --match-on angstrom --n-random-controls 500 \
        --n-bootstrap 500 --fill-mode constant --continuum-fill empirical-median
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.interpret.classifier import load_model
from src.interpret.line_match import save_sweep, sweep_tolerances
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS
from src.interpret.ablation import (
    VALID_FILL_MODES,
    VALID_MATCH_ON,
    VALID_NULL_MODES,
    AblationRow,
    masked_line_ablation,
)

logger = logging.getLogger(__name__)

HEADLINE_PAIRS: list[tuple[str, str]] = [
    ("H_balmer", "F"),
    ("Mg_b", "G"),
    ("Mg_b", "K"),
]

PIVOT_PAIRS: list[tuple[str, str]] = [
    ("Na_D", "K"),
    ("Ca_I", "K"),
]

# Per-pair Bonferroni-corrected alpha for the headline gate. Family-wise
# alpha = 0.01 across the len(HEADLINE_PAIRS) = 3 headline pairs, so the
# per-pair alpha is 0.01 / 3 = 0.00333.... With 500 random windows the
# Phipson-Smyth floor is 1/501 = 0.00200; with the default 5000 windows it
# is 1/5001 = 0.00020.
GATE_FAMILYWISE_ALPHA: float = 0.01
GATE_PVALUE_MAX: float = GATE_FAMILYWISE_ALPHA / 3.0
GATE_DELTA_CI_HIGH_MAX: float = 0.0
HEADLINE_PASS_FLOOR: int = 2
SENSITIVITY_SHIFT_THRESHOLD: float = 0.02


def _write_csv(path: Path, rows: list[AblationRow]) -> None:
    """Write a list of AblationRow dataclasses to ``path`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    header = list(asdict(rows[0]).keys())
    lines_out = [",".join(header)]
    for r in rows:
        d = asdict(r)
        lines_out.append(",".join(
            f"{d[k]:.6g}" if isinstance(d[k], float) else str(d[k]) for k in header
        ))
    path.write_text("\n".join(lines_out) + "\n")


def _row_lookup(rows: list[AblationRow]) -> dict[tuple[str, str], AblationRow]:
    """Build a lookup keyed by (line_set, mk_class)."""
    return {(r.line_set, r.mk_class): r for r in rows}


PAIRED_FIELDS: tuple[str, ...] = (
    "n_bins_masked",
    "total_width_aa",
    "n_flip_lost",
    "n_flip_gained",
    "mcnemar_exact_p",
    "mean_delta_true_prob",
    "flip_rate_upper95",
    "n_random_controls_succeeded",
    "null_mode",
    "match_on",
    "fill_mode",
)


def _paired_fields(row: AblationRow) -> dict[str, Any]:
    """JSON-serialisable view of the paired statistics and configuration fields."""
    out: dict[str, Any] = {}
    for name in PAIRED_FIELDS:
        v = getattr(row, name)
        if isinstance(v, (np.floating, float)):
            v = float(v)
            v = None if not np.isfinite(v) else v
        elif isinstance(v, (np.integer, int)):
            v = int(v)
        out[name] = v
    return out


def evaluate_headline_gate(rows: list[AblationRow]) -> dict[str, Any]:
    """Evaluate the FGK headline gate on the ablation rows.

    A headline pair passes when its Monte-Carlo p-value is at most
    ``GATE_PVALUE_MAX`` and the upper bootstrap CI bound on delta is below
    ``GATE_DELTA_CI_HIGH_MAX``. The paired statistics (flip counts, exact
    McNemar p, mean change in true-class probability, Clopper-Pearson
    bound) are reported per pair but do not enter the pass/fail decision.

    Returns a dict with:
      - n_pairs_passed: int
      - pairs_evaluated: per-pair detail
      - pivot_confirmed: per-pivot-pair detail with predicted near-zero delta
      - gate_status: PASS | FAIL | INCONCLUSIVE
      - draw_success_rates: filled by the caller from masked_line_ablation
    """
    lookup = _row_lookup(rows)
    pairs_eval: list[dict[str, Any]] = []
    n_passed = 0
    for line_set, mk_class in HEADLINE_PAIRS:
        row = lookup.get((line_set, mk_class))
        if row is None:
            pairs_eval.append({
                "line_set": line_set,
                "mk_class": mk_class,
                "found": False,
                "passed": False,
                "reason": "row not present in ablation output",
            })
            continue
        passed = bool(
            np.isfinite(row.p_value_vs_random)
            and np.isfinite(row.delta_acc_ci_high)
            and row.p_value_vs_random <= GATE_PVALUE_MAX
            and row.delta_acc_ci_high < GATE_DELTA_CI_HIGH_MAX
        )
        if passed:
            n_passed += 1
        pairs_eval.append({
            "line_set": line_set,
            "mk_class": mk_class,
            "found": True,
            "n_test": int(row.n_test),
            "baseline_acc": float(row.baseline_acc),
            "masked_acc_mean": float(row.masked_acc_mean),
            "delta_acc_mean": float(row.delta_acc_mean),
            "delta_acc_ci_low": float(row.delta_acc_ci_low),
            "delta_acc_ci_high": float(row.delta_acc_ci_high),
            "p_value_vs_random": float(row.p_value_vs_random),
            **_paired_fields(row),
            "passed": passed,
        })
    pivot_eval: list[dict[str, Any]] = []
    for line_set, mk_class in PIVOT_PAIRS:
        row = lookup.get((line_set, mk_class))
        if row is None:
            pivot_eval.append({
                "line_set": line_set,
                "mk_class": mk_class,
                "found": False,
                "predicted_near_zero": "",
            })
            continue
        # A near-zero delta is predicted for these pairs. The prediction is
        # flagged as confirmed when the 95 percent CI for delta_acc straddles
        # zero; refuted otherwise.
        ci_lo = float(row.delta_acc_ci_low)
        ci_hi = float(row.delta_acc_ci_high)
        confirms = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= 0.0 <= ci_hi)
        pivot_eval.append({
            "line_set": line_set,
            "mk_class": mk_class,
            "found": True,
            "n_test": int(row.n_test),
            "delta_acc_mean": float(row.delta_acc_mean),
            "delta_acc_ci_low": ci_lo,
            "delta_acc_ci_high": ci_hi,
            "p_value_vs_random": float(row.p_value_vs_random),
            **_paired_fields(row),
            "decision_30_prediction": "near_zero_delta",
            "ci_straddles_zero": confirms,
        })
    gate_status = "PASS" if n_passed >= HEADLINE_PASS_FLOOR else "FAIL"
    return {
        "n_pairs_passed": int(n_passed),
        "n_pairs_required": HEADLINE_PASS_FLOOR,
        "n_pairs_total": len(HEADLINE_PAIRS),
        "pairs_evaluated": pairs_eval,
        "pivot_confirmed": pivot_eval,
        "gate_status": gate_status,
    }


def _sensitivity_pass(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wave_centers: np.ndarray,
    class_labels: list[str],
    gap_mask: np.ndarray,
    headline_only: dict[str, list[tuple[float, float]]],
    seed: int,
    n_bootstrap: int,
    n_random_controls: int,
    continuum_fill: float,
    fill_mode: str = "constant",
    null_mode: str = "class_matched",
    match_on: str = "bins",
) -> dict[tuple[str, str], float]:
    """Re-run masked_line_ablation on the headline line sets only with an
    explicit ``continuum_fill`` value and ``fill_mode``. Returns
    ``{(set, class): delta_acc_mean}``.
    """
    logger.info(
        "sensitivity pass: continuum_fill=%.6f fill_mode=%s over headline line sets %s",
        continuum_fill, fill_mode, list(headline_only.keys()),
    )
    rows = masked_line_ablation(
        model,
        X_test,
        y_test,
        wave_centers,
        headline_only,
        class_labels,
        per_class=True,
        n_bootstrap=n_bootstrap,
        n_random_controls=n_random_controls,
        seed=seed,
        gap_mask=gap_mask,
        continuum_fill=continuum_fill,
        fill_mode=fill_mode,
        null_mode=null_mode,
        match_on=match_on,
    )
    return {(r.line_set, r.mk_class): float(r.delta_acc_mean) for r in rows}


def _line_match_with_gap_mask(
    perm_npz_path: Path,
    gap_mask: np.ndarray,
    csv_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    """Run the line-matching tolerance sweep on the global perm-importance trace
    after zeroing out gap-mask bins, so gap-imputed bins cannot register
    as line matches.
    """
    if not perm_npz_path.exists():
        logger.warning("perm-importance npz not found at %s; skipping line match", perm_npz_path)
        return {"available": False, "path": str(perm_npz_path)}
    payload = np.load(perm_npz_path)
    importance = payload["importance_mean"].astype(np.float64).copy()
    wave_centers_perm = payload["wave_centers"]
    if importance.shape != wave_centers_perm.shape:
        raise ValueError(
            f"perm importance shape {importance.shape} does not match "
            f"wave_centers shape {wave_centers_perm.shape}"
        )
    if gap_mask.shape != importance.shape:
        raise ValueError(
            f"gap_mask shape {gap_mask.shape} does not match "
            f"perm wave_centers shape {importance.shape}"
        )
    importance[gap_mask] = 0.0
    results = sweep_tolerances(importance, wave_centers_perm)
    save_sweep(csv_path, json_path, results)
    summary = {
        "available": True,
        "path": str(perm_npz_path),
        "tolerances_aa": [r.tolerance_aa for r in results],
        "precision": {r.tolerance_aa: r.precision for r in results},
        "recall": {r.tolerance_aa: r.recall for r in results},
        "jaccard": {r.tolerance_aa: r.jaccard for r in results},
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the ablation driver."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="bootstrap resamples for the delta CI (default 2000; legacy runs used 500)",
    )
    p.add_argument(
        "--n-random-controls", type=int, default=5000,
        help=(
            "size of the random-window Monte-Carlo reference set; p-value "
            "floor is 1/(B+1) (default 5000 -> 0.0002; legacy 500 -> 0.002)"
        ),
    )
    p.add_argument(
        "--null-mode", choices=VALID_NULL_MODES, default="class_matched",
        help=(
            "class_matched: compare each class's recall delta against that "
            "class's random-window recall deltas (default); pooled: compare "
            "against the all-class accuracy deltas (legacy)"
        ),
    )
    p.add_argument(
        "--match-on", choices=VALID_MATCH_ON, default="bins",
        help=(
            "bins: random windows reproduce the line set's per-segment bin "
            "counts and do not overlap (default); angstrom: equal-width "
            "segments matching total width in Angstrom (legacy)"
        ),
    )
    p.add_argument(
        "--fill-mode", choices=VALID_FILL_MODES + ("interp",), default="constant",
        help=(
            "production fill convention for masked bins: constant (default), "
            "noise, or gp_interp/interp (linear interpolation)"
        ),
    )
    p.add_argument("--boundary-k", type=float, default=150.0,
                   help="robustness subset: keep only spectra with boundary_distance_k >= this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--perm-importance",
        type=Path,
        default=Path("artifacts/interpret/perm_importance.npz"),
        help="path to global permutation-importance npz for line-matching",
    )
    p.add_argument(
        "--continuum-fill",
        type=str,
        default="1.0",
        help=(
            "continuum-fill value for masked bins; either a float literal or "
            "'empirical-median' to use np.nanmedian(X_test[:, ~gap_mask]). The "
            "sensitivity rerun is unconditional and uses the empirical median "
            "regardless of this setting."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info(
        "ablation CLI: features=%s model=%s out_dir=%s n_bootstrap=%d "
        "n_random_controls=%d seed=%d null_mode=%s match_on=%s fill_mode=%s",
        args.features, args.model, args.out_dir, args.n_bootstrap,
        args.n_random_controls, args.seed, args.null_mode, args.match_on,
        args.fill_mode,
    )
    mode_kwargs: dict[str, Any] = {
        "null_mode": args.null_mode,
        "match_on": args.match_on,
    }

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    boundary = payload["boundary_distance_k"]
    test_idx = payload["test_idx"]
    if "gap_mask" not in payload.files:
        raise KeyError(
            "features.npz is missing 'gap_mask'. Re-run "
            "scripts/build_features.py to add it."
        )
    gap_mask = payload["gap_mask"].astype(bool)

    model = load_model(args.model)
    # Use the full ALLOWED_MK_CLASSES mapping so that integer class ids in y
    # index correctly into class_labels even when not all classes are present.
    class_labels = list(ALLOWED_MK_CLASSES)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    X_test = X[test_idx]
    y_test = y[test_idx]
    empirical_median = float(np.nanmedian(X_test[:, ~gap_mask]))
    if args.continuum_fill == "empirical-median":
        production_fill: float | None = empirical_median
        logger.info("production fill: empirical median %.6f", empirical_median)
    else:
        try:
            production_fill = float(args.continuum_fill)
        except ValueError as exc:
            raise ValueError(
                f"--continuum-fill must be a float or 'empirical-median', got {args.continuum_fill!r}"
            ) from exc
        if production_fill == 1.0:
            production_fill = None  # use module CONTINUUM_FILL=1.0
            logger.info("production fill: locked CONTINUUM_FILL=1.0")
        else:
            logger.info("production fill: %.6f", production_fill)

    logger.info("running full-test ablation (n=%d, gap_bins=%d)",
                len(test_idx), int(gap_mask.sum()))

    draw_stats: dict[str, dict[str, int]] = {}
    per_row: dict[str, np.ndarray] = {}
    rows_full = masked_line_ablation(
        model, X_test, y_test, wc, LINE_SETS, class_labels,
        per_class=True,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        seed=args.seed,
        gap_mask=gap_mask,
        continuum_fill=production_fill,
        draw_stats_out=draw_stats,
        per_row_out=per_row,
        fill_mode=args.fill_mode,
        **mode_kwargs,
    )
    full_csv = args.out_dir / "masked_line_ablation_full.csv"
    _write_csv(full_csv, rows_full)
    logger.info("wrote %s (%d rows)", full_csv, len(rows_full))

    per_row["test_idx"] = np.asarray(test_idx, dtype=np.int64)
    per_row_path = args.out_dir / "per_row_predictions.npz"
    np.savez(per_row_path, **per_row)
    logger.info("wrote %s (%d keys, %d rows per array)",
                per_row_path, len(per_row), len(test_idx))

    boundary_mask = boundary[test_idx] >= args.boundary_k
    kept = test_idx[boundary_mask]
    logger.info(
        "boundary-distance subset (>= %.0f K): %d/%d test rows",
        args.boundary_k, int(boundary_mask.sum()), len(test_idx),
    )
    if len(kept) >= 50:
        rows_sub = masked_line_ablation(
            model, X[kept], y[kept], wc, LINE_SETS, class_labels,
            per_class=True,
            n_bootstrap=args.n_bootstrap,
            n_random_controls=args.n_random_controls,
            seed=args.seed,
            gap_mask=gap_mask,
            continuum_fill=production_fill,
            fill_mode=args.fill_mode,
            **mode_kwargs,
        )
        _write_csv(
            args.out_dir / f"masked_line_ablation_boundary{int(args.boundary_k)}.csv",
            rows_sub,
        )
    else:
        logger.warning("fewer than 50 rows after boundary cut; skipping subset run")

    # Continuum-fill sensitivity pass. The check is invariant to the
    # production fill: always compare a fill=1.0 rerun against a
    # fill=empirical-median rerun on the headline pairs only.
    headline_line_sets = {
        name: LINE_SETS[name]
        for name in {ls for ls, _ in HEADLINE_PAIRS}
        if name in LINE_SETS
    }
    deltas_at_one = _sensitivity_pass(
        model, X_test, y_test, wc, class_labels, gap_mask,
        headline_line_sets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        continuum_fill=1.0,
        fill_mode=args.fill_mode,
        **mode_kwargs,
    )
    deltas_at_median = _sensitivity_pass(
        model, X_test, y_test, wc, class_labels, gap_mask,
        headline_line_sets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        continuum_fill=empirical_median,
        fill_mode=args.fill_mode,
        **mode_kwargs,
    )
    sensitivity_pairs: list[dict[str, Any]] = []
    max_shift = 0.0
    for line_set, mk_class in HEADLINE_PAIRS:
        d_one = deltas_at_one.get((line_set, mk_class))
        d_med = deltas_at_median.get((line_set, mk_class))
        if d_one is None or d_med is None:
            sensitivity_pairs.append({
                "line_set": line_set,
                "mk_class": mk_class,
                "delta_at_fill_1.0": d_one,
                "delta_at_empirical_median": d_med,
                "shift": None,
            })
            continue
        shift = abs(d_one - d_med)
        max_shift = max(max_shift, shift)
        sensitivity_pairs.append({
            "line_set": line_set,
            "mk_class": mk_class,
            "delta_at_fill_1.0": d_one,
            "delta_at_empirical_median": d_med,
            "shift": shift,
        })
    if max_shift > SENSITIVITY_SHIFT_THRESHOLD:
        continuum_fill_status = "REQUIRES_RERUN_WITH_EMPIRICAL_MEDIAN"
        logger.critical(
            "continuum-fill sensitivity max_shift=%.4f exceeds %.4f threshold; "
            "production run REQUIRES rerun with continuum_fill=%.4f",
            max_shift, SENSITIVITY_SHIFT_THRESHOLD, empirical_median,
        )
    else:
        continuum_fill_status = "PASS_AT_1.0"
        logger.info(
            "continuum-fill sensitivity max_shift=%.4f within %.4f threshold; "
            "1.0 fill convention is defensible",
            max_shift, SENSITIVITY_SHIFT_THRESHOLD,
        )
    sensitivity = {
        "continuum_fill_locked": 1.0,
        "continuum_fill_empirical_median": empirical_median,
        "shift_threshold": SENSITIVITY_SHIFT_THRESHOLD,
        "max_shift": max_shift,
        "continuum_fill_status": continuum_fill_status,
        "production_continuum_fill": (
            1.0 if production_fill is None else float(production_fill)
        ),
        "pairs": sensitivity_pairs,
    }

    # Fill-mode sensitivity. The constant-fill sensitivity check exceeded
    # the 0.02 trigger (max_shift = 0.039 in production), so the fill
    # convention matters; the noise-injection and linear-interpolation
    # fills at the production continuum_fill bound the dependence of the
    # headline deltas on the specific fill convention.
    production_continuum_fill = (
        1.0 if production_fill is None else float(production_fill)
    )
    fill_mode_line_sets = {
        name: LINE_SETS[name]
        for name in {ls for ls, _ in (HEADLINE_PAIRS + PIVOT_PAIRS)}
        if name in LINE_SETS
    }
    deltas_constant = _sensitivity_pass(
        model, X_test, y_test, wc, class_labels, gap_mask,
        fill_mode_line_sets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        continuum_fill=production_continuum_fill,
        fill_mode="constant",
        **mode_kwargs,
    )
    deltas_noise = _sensitivity_pass(
        model, X_test, y_test, wc, class_labels, gap_mask,
        fill_mode_line_sets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        continuum_fill=production_continuum_fill,
        fill_mode="noise",
        **mode_kwargs,
    )
    deltas_gp = _sensitivity_pass(
        model, X_test, y_test, wc, class_labels, gap_mask,
        fill_mode_line_sets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_random_controls=args.n_random_controls,
        continuum_fill=production_continuum_fill,
        fill_mode="gp_interp",
        **mode_kwargs,
    )
    fill_mode_pairs: list[dict[str, Any]] = []
    max_fill_mode_shift = 0.0
    for line_set, mk_class in HEADLINE_PAIRS + PIVOT_PAIRS:
        d_c = deltas_constant.get((line_set, mk_class))
        d_n = deltas_noise.get((line_set, mk_class))
        d_g = deltas_gp.get((line_set, mk_class))
        values = [v for v in (d_c, d_n, d_g) if v is not None]
        if len(values) >= 2:
            shift = float(max(values) - min(values))
            max_fill_mode_shift = max(max_fill_mode_shift, shift)
        else:
            shift = None
        fill_mode_pairs.append({
            "line_set": line_set,
            "mk_class": mk_class,
            "delta_at_constant": d_c,
            "delta_at_noise": d_n,
            "delta_at_gp_interp": d_g,
            "shift_max_minus_min": shift,
        })
    fill_mode_sensitivity = {
        "production_fill_mode": args.fill_mode,
        "production_continuum_fill": production_continuum_fill,
        "shift_threshold": SENSITIVITY_SHIFT_THRESHOLD,
        "max_shift": max_fill_mode_shift,
        "status": (
            "ROBUST_ACROSS_FILL_MODES"
            if max_fill_mode_shift <= SENSITIVITY_SHIFT_THRESHOLD
            else "FILL_MODE_DEPENDENT"
        ),
        "pairs": fill_mode_pairs,
    }
    if max_fill_mode_shift > SENSITIVITY_SHIFT_THRESHOLD:
        logger.warning(
            "fill_mode sensitivity max_shift=%.4f exceeds %.4f threshold; "
            "manuscript should report this explicitly",
            max_fill_mode_shift, SENSITIVITY_SHIFT_THRESHOLD,
        )
    else:
        logger.info(
            "fill_mode sensitivity max_shift=%.4f within %.4f threshold; "
            "headline deltas are robust across constant/noise/gp_interp fills",
            max_fill_mode_shift, SENSITIVITY_SHIFT_THRESHOLD,
        )

    # Line-matching with gap-masked perm trace.
    line_match_summary = _line_match_with_gap_mask(
        args.perm_importance,
        gap_mask,
        args.out_dir / "line_match.csv",
        args.out_dir / "line_match.json",
    )
    precision_at_5 = None
    if line_match_summary.get("available"):
        precision_at_5 = float(
            line_match_summary["precision"].get(5.0, float("nan"))
        )
        logger.info("line-match precision at 5 A: %.4f", precision_at_5)

    # Headline gate verdict.
    gate = evaluate_headline_gate(rows_full)
    gate["draw_success_rates"] = {
        set_name: {
            "requested": stats["requested"],
            "succeeded": stats["succeeded"],
            "rate": stats["succeeded"] / stats["requested"] if stats["requested"] else 0.0,
            "above_floor": stats["succeeded"] >= int(0.8 * stats["requested"]),
            "match_on": stats.get("match_on"),
            "target_bin_counts": stats.get("target_bin_counts"),
            "target_total_width_aa": stats.get("target_total_width_aa"),
            "n_draws_bin_matched": stats.get("n_draws_bin_matched"),
            "realised_total_bins_histogram": stats.get("realised_total_bins_histogram"),
        }
        for set_name, stats in draw_stats.items()
    }
    gate["continuum_fill_sensitivity"] = sensitivity
    gate["fill_mode_sensitivity"] = fill_mode_sensitivity
    gate["line_match"] = {
        "precision_at_5_aa": precision_at_5,
        "summary": line_match_summary,
    }
    gate["headline_pairs"] = [{"line_set": ls, "mk_class": c} for ls, c in HEADLINE_PAIRS]
    gate["pivot_pairs"] = [{"line_set": ls, "mk_class": c} for ls, c in PIVOT_PAIRS]
    gate["seed"] = int(args.seed)
    gate["n_bootstrap"] = int(args.n_bootstrap)
    gate["n_random_controls"] = int(args.n_random_controls)
    gate["null_mode"] = args.null_mode
    gate["match_on"] = args.match_on
    gate["fill_mode"] = args.fill_mode
    gate["gate_threshold_per_pair"] = float(GATE_PVALUE_MAX)
    gate["gate_threshold_familywise_alpha"] = float(GATE_FAMILYWISE_ALPHA)
    gate["gate_correction"] = "bonferroni"
    gate["gate_p_value_convention"] = "phipson_smyth_2010_b_plus_one"

    gate_path = args.out_dir / "gate_eval.json"
    with gate_path.open("w") as f:
        json.dump(gate, f, indent=2)
    logger.info("wrote %s", gate_path)
    logger.info(
        "headline gate: %s (%d/%d pairs passed; threshold %d)",
        gate["gate_status"], gate["n_pairs_passed"],
        gate["n_pairs_total"], gate["n_pairs_required"],
    )
    print(f"wrote ablation artefacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
