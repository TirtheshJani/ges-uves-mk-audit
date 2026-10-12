#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Positive control for the K-class insensitivity claim of the masked-line ablation.

The deposited ablation (``scripts/revision/ablation_recalibrated.py``) masks
five catalogued line sets and finds that K recall does not respond to any of
them. This script asks whether the same machinery CAN detect a K-class
dependence when the masked window is chosen to be the one the model actually
uses: the 20, 50 and 100 bins with the largest validation-set mean |SHAP| for
the K class, plus the union of all five catalogued windows masked at once.

Line-set dictionary (in this order; the first five are the deposited sets,
taken verbatim from ``artifacts/revision/ablation_recalibrated.json``):

  H_balmer, Mg_b, Na_D, Ca_I, Fe_Cr      deposited catalogued sets
  Joint_all                               union of the nine windows above
  SHAP_K_top20 / top50 / top100           top-N K-class mean |SHAP| bins
                                          (gap bins excluded), expressed as
                                          windows over maximal runs of
                                          consecutive bins so that
                                          ``_window_mask`` recovers them
                                          exactly

Reference-set settings are those of the deposited run: class-matched null,
bin-matched non-overlapping random windows, 5000 random windows, 2000
bootstrap resamples, seed 42, gap bins forbidden. Two fills are run:
``gp_interp`` (primary, linear interpolation) and ``constant`` (train
off-gap median, about 0.9168).

Reproduction check. ``masked_line_ablation`` builds the forbidden mask of the
random-window sampler from the union of ALL line sets it is given. The SHAP
sets add bins outside the 56 catalogued bins, so in the nine-set call the
random windows for the first five sets are drawn from a smaller region than
in the deposited run and the random-number stream diverges as soon as a draw
is rejected. Deterministic fields (delta, flips) must still match the
deposited rows; the Monte-Carlo p-values (and the bootstrap CIs, which come
later in the stream) may not. The script therefore runs, for each fill,

  (a) the five deposited sets alone (exact deposited configuration), which
      must reproduce the deposited rows field for field, and
  (b) the nine-set call that yields the positive-control rows,

and writes both comparisons into ``reproduction_check``.

Outputs (under ``artifacts/revision/``):

  k_positive_control.json          line sets, reproduction check, both
                                   runs, summary table
  k_positive_control_per_row.npz   per-row predictions / probabilities,
                                   keys prefixed ``<run>__`` (nine-set call)
                                   and ``<run>__repro5__`` (five-set call)

Run from the repo root:  python scripts/revision/k_positive_control.py
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.ablation import (  # noqa: E402
    AblationRow,
    _segment_bin_counts,
    _window_mask,
    masked_line_ablation,
)
from src.interpret.classifier import load_model  # noqa: E402
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS  # noqa: E402

logger = logging.getLogger("k_positive_control")

CLASS_NAMES: dict[int, str] = {1: "F", 2: "G", 3: "K"}
SEED = 42
N_BOOTSTRAP = 2000
N_RANDOM_CONTROLS = 5000
NULL_MODE = "class_matched"
MATCH_ON = "bins"
DEPOSITED_SET_ORDER = ("H_balmer", "Mg_b", "Na_D", "Ca_I", "Fe_Cr")
NEW_SET_ORDER = ("Joint_all", "SHAP_K_top20", "SHAP_K_top50", "SHAP_K_top100")
SHAP_TOP_N = {"SHAP_K_top20": 20, "SHAP_K_top50": 50, "SHAP_K_top100": 100}
SHAP_K_CLASS_INDEX = 2  # mean_abs_per_class rows are (F, G, K)
REPRO_FIELDS = ("delta_acc_mean", "n_flip_lost", "n_flip_gained", "p_value_vs_random")
EXTRA_FIELDS = (
    "delta_acc_ci_low", "delta_acc_ci_high", "mcnemar_exact_p", "mean_delta_true_prob",
    "flip_rate_upper95", "n_random_controls_succeeded", "masked_acc_mean", "baseline_acc",
)
SUMMARY_FIELDS = (
    "delta_acc_mean", "delta_acc_ci_low", "delta_acc_ci_high", "n_flip_lost", "n_flip_gained",
    "mcnemar_exact_p", "flip_rate_upper95", "p_value_vs_random", "mean_delta_true_prob",
)
EXPECTED_CONFUSION = np.array([[68, 13, 0], [3, 222, 5], [1, 8, 136]])
REPRO_TOL = 1e-9


# --------------------------------------------------------------------------- helpers
def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not np.isfinite(v) else v
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def row_to_dict(row: AblationRow) -> dict[str, Any]:
    """Every AblationRow field, JSON-safe (NaN -> null)."""
    return {k: _jsonable(v) for k, v in asdict(row).items()}


def fe_cr_from_source() -> list[tuple[float, float]] | None:
    """Parse ``FE_CR_LINE_SET`` from the appendix script without executing it.

    The appendix script imports ``src.interpret.benchmark`` at module level,
    which is not part of this partial repository copy, so ``ast`` is used.
    """
    path = REPO_ROOT / "scripts" / "appendix_a2_polynomial_diagnostics.py"
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "FE_CR_LINE_SET":
                    val = ast.literal_eval(node.value)
                    return [(float(a), float(b)) for a, b in val]
    return None


def runs_of_consecutive(indices: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs ``(start, end)`` (inclusive) of consecutive sorted indices."""
    idx = np.sort(np.asarray(indices, dtype=int))
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == p + 1:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    runs.append((s, p))
    return runs


def shap_top_n_windows(
    score: np.ndarray, wave_centers: np.ndarray, gap_mask: np.ndarray, n: int,
) -> tuple[list[tuple[float, float]], np.ndarray, dict[str, Any]]:
    """Windows covering the ``n`` bins with the largest ``score`` outside the gap.

    Ranking is a stable descending sort of ``score`` with gap bins set to
    -inf. The selected bin indices are grouped into maximal runs of
    consecutive indices and each run becomes ``(wave_centers[start],
    wave_centers[end])``; ``_window_mask`` (inclusive on bin centres) then
    recovers exactly the selected bins, which is asserted.
    """
    s = np.asarray(score, dtype=np.float64).copy()
    s[gap_mask] = -np.inf
    order = np.argsort(-s, kind="stable")
    sel = np.sort(order[:n])
    assert not gap_mask[sel].any(), "gap bin selected"
    runs = runs_of_consecutive(sel)
    windows = [(float(wave_centers[a]), float(wave_centers[b])) for a, b in runs]
    m = _window_mask(wave_centers, windows)
    assert np.array_equal(np.flatnonzero(m), sel), f"_window_mask does not recover the top-{n} bins"
    nth, nxt = float(s[order[n - 1]]), float(s[order[n]])
    meta = {
        "n_bins": int(n),
        "n_windows": len(windows),
        "bin_indices": [int(i) for i in sel],
        "nth_score": nth,
        "next_score": nxt,
        "tie_at_boundary": bool(nth == nxt),
        "share_of_total_mean_abs_shap": float(np.asarray(score, dtype=np.float64)[sel].sum()
                                             / np.asarray(score, dtype=np.float64).sum()),
    }
    return windows, sel, meta


def lost_row_destinations(
    y_test: np.ndarray, y_pred_base: np.ndarray, y_pred_masked: np.ndarray, cls: int,
) -> dict[str, int]:
    rows = (y_test == cls) & (y_pred_base == cls) & (y_pred_masked != cls)
    dest = y_pred_masked[rows]
    return {CLASS_NAMES[c]: int(np.sum(dest == c)) for c in CLASS_NAMES if c != cls}


def gained_row_origins(
    y_test: np.ndarray, y_pred_base: np.ndarray, y_pred_masked: np.ndarray, cls: int,
) -> dict[str, int]:
    rows = (y_test == cls) & (y_pred_base != cls) & (y_pred_masked == cls)
    origin = y_pred_base[rows]
    return {CLASS_NAMES[c]: int(np.sum(origin == c)) for c in CLASS_NAMES if c != cls}


def compare_rows(
    new_rows: list[AblationRow],
    deposited_rows: list[dict[str, Any]],
    sets: tuple[str, ...],
    fields: tuple[str, ...],
    tol: float,
) -> dict[str, Any]:
    """Field-by-field comparison of ``new_rows`` with the deposited rows for ``sets``."""
    dep = {(r["line_set"], r["mk_class"]): r for r in deposited_rows}
    new = {(r.line_set, r.mk_class): row_to_dict(r) for r in new_rows}
    details: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    max_abs: dict[str, float] = {f: 0.0 for f in fields}
    n_rows = 0
    for s in sets:
        for cls in ("F", "G", "K"):
            key = (s, cls)
            if key not in dep or key not in new:
                mismatches.append({"line_set": s, "mk_class": cls, "field": "<row>",
                                   "deposited": key in dep, "reproduced": key in new})
                continue
            n_rows += 1
            entry: dict[str, Any] = {"line_set": s, "mk_class": cls, "fields": {}}
            for f in fields:
                a, b = dep[key][f], new[key][f]
                if a is None or b is None:
                    diff = 0.0 if (a is None and b is None) else float("inf")
                else:
                    diff = abs(float(a) - float(b))
                ok = diff <= tol
                max_abs[f] = max(max_abs[f], diff if np.isfinite(diff) else float("inf"))
                entry["fields"][f] = {"deposited": a, "reproduced": b, "abs_diff": _jsonable(diff), "match": ok}
                if not ok:
                    mismatches.append({"line_set": s, "mk_class": cls, "field": f,
                                       "deposited": a, "reproduced": b, "abs_diff": _jsonable(diff)})
            details.append(entry)
    overall = max(max_abs.values()) if max_abs else 0.0
    return {
        "passed": len(mismatches) == 0,
        "n_rows_compared": n_rows,
        "n_mismatching_fields": len(mismatches),
        "max_abs_diff": _jsonable(overall),
        "max_abs_diff_per_field": {f: _jsonable(v) for f, v in max_abs.items()},
        "mismatches": mismatches,
        "details": details,
    }


def run_ablation(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, wave_centers: np.ndarray,
    line_sets: dict[str, list[tuple[float, float]]], class_labels: list[str],
    gap_mask: np.ndarray, fill_mode: str, continuum_fill: float | None,
    n_bootstrap: int, n_random_controls: int, seed: int, tag: str,
) -> tuple[list[AblationRow], dict[str, Any], dict[str, np.ndarray], float]:
    t0 = time.perf_counter()
    forbidden = gap_mask.copy()
    for w in line_sets.values():
        forbidden |= _window_mask(wave_centers, w)
    print(f"[{tag}] start: {len(line_sets)} line sets {list(line_sets)}, fill_mode={fill_mode}, "
          f"fill={continuum_fill}, forbidden bins for random sampler={int(forbidden.sum())}", flush=True)
    draw_stats: dict[str, dict[str, Any]] = {}
    per_row: dict[str, np.ndarray] = {}
    rows = masked_line_ablation(
        model, X_test, y_test, wave_centers, line_sets, class_labels,
        per_class=True,
        n_bootstrap=n_bootstrap,
        n_random_controls=n_random_controls,
        seed=seed,
        gap_mask=gap_mask,
        continuum_fill=continuum_fill,
        draw_stats_out=draw_stats,
        per_row_out=per_row,
        fill_mode=fill_mode,
        null_mode=NULL_MODE,
        match_on=MATCH_ON,
    )
    dt = time.perf_counter() - t0
    succ = {s: d["succeeded"] / d["requested"] for s, d in draw_stats.items()}
    print(f"[{tag}] done in {dt:.0f} s; random-draw success rates {succ}", flush=True)
    for s, r in succ.items():
        if r < 0.80:
            print(f"[{tag}] WARNING: sampler struggled on {s}: success rate {r:.3f}", flush=True)
    return rows, draw_stats, per_row, dt


def print_table(rows: list[AblationRow], title: str) -> None:
    print(f"--- {title}")
    print(f"{'set':14s} {'cls':3s} {'n':>3s} {'delta':>8s} {'ci_lo':>8s} {'ci_hi':>8s} {'lost':>4s} "
          f"{'gain':>4s} {'McNemar':>10s} {'CP95':>7s} {'p_MC':>10s} {'mdp':>8s} {'succ':>5s}")
    for r in rows:
        print(f"{r.line_set:14s} {r.mk_class:3s} {r.n_test:3d} {r.delta_acc_mean:+8.3f} "
              f"{r.delta_acc_ci_low:+8.3f} {r.delta_acc_ci_high:+8.3f} {r.n_flip_lost:4d} "
              f"{r.n_flip_gained:4d} {r.mcnemar_exact_p:10.4g} {r.flip_rate_upper95:7.4f} "
              f"{r.p_value_vs_random:10.4g} {r.mean_delta_true_prob:+8.4f} "
              f"{r.n_random_controls_succeeded:5d}")
    sys.stdout.flush()


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--shap", type=Path, default=Path("artifacts/interpret/shap_values.npz"))
    p.add_argument("--deposited", type=Path, default=Path("artifacts/revision/ablation_recalibrated.json"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/revision"))
    p.add_argument("--out-stem", type=str, default="k_positive_control")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--n-random-controls", type=int, default=N_RANDOM_CONTROLS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-repro5", action="store_true",
                   help="skip the five-set reproduction calls (nine-set calls only)")
    p.add_argument("--fills", nargs="*", default=["gp_interp", "constant"],
                   choices=["gp_interp", "constant"])
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    t_script = time.perf_counter()

    # ---- data, model, deposited results ------------------------------------
    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"].astype(bool)
    train_idx, val_idx, test_idx = fp["train_idx"], fp["val_idx"], fp["test_idx"]
    X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load_model(args.model)
    for w in caught:
        logger.info("unpickle warning (expected, harmless): %s", str(w.message).splitlines()[0])
    class_labels = list(ALLOWED_MK_CLASSES)  # index 1 -> F, 2 -> G, 3 -> K
    with args.deposited.open() as f:
        deposited = json.load(f)

    # ---- Step 0: confusion-matrix sanity check -----------------------------
    y_pred_base = np.asarray(model.predict(X_test)).astype(np.int64)
    cm = np.zeros((3, 3), dtype=int)
    for i, ct in enumerate((1, 2, 3)):
        for j, cp in enumerate((1, 2, 3)):
            cm[i, j] = int(np.sum((y_test == ct) & (y_pred_base == cp)))
    step0_ok = bool(np.array_equal(cm, EXPECTED_CONFUSION))
    print(f"[step 0] confusion matrix (rows true F,G,K; cols pred F,G,K): {cm.tolist()} "
          f"expected {EXPECTED_CONFUSION.tolist()} -> {'OK' if step0_ok else 'MISMATCH'}", flush=True)
    if not step0_ok:
        print("[step 0] deposited model does not reproduce the deposited confusion matrix; stopping.",
              flush=True)
        return 2
    baseline_acc = float(np.mean(y_pred_base == y_test))

    # ---- line sets ---------------------------------------------------------
    dep_sets = {k: [tuple(float(x) for x in w) for w in v] for k, v in deposited["line_sets"].items()}
    assert tuple(dep_sets) == DEPOSITED_SET_ORDER, f"deposited set order {tuple(dep_sets)}"
    for name in ("H_balmer", "Mg_b", "Na_D", "Ca_I"):
        src_set = [tuple(float(x) for x in w) for w in LINE_SETS[name]]
        if src_set != dep_sets[name]:
            raise RuntimeError(f"deposited {name} windows {dep_sets[name]} differ from LINE_SETS {src_set}")
    fe_cr_src = fe_cr_from_source()
    if fe_cr_src is not None and fe_cr_src != dep_sets["Fe_Cr"]:
        raise RuntimeError(f"deposited Fe_Cr {dep_sets['Fe_Cr']} differs from FE_CR_LINE_SET {fe_cr_src}")

    line_sets: dict[str, list[tuple[float, float]]] = {k: list(v) for k, v in dep_sets.items()}
    joint = [w for v in dep_sets.values() for w in v]
    line_sets["Joint_all"] = joint
    catalogued_mask = _window_mask(wave_centers, joint)

    sh = np.load(args.shap, allow_pickle=False)
    assert np.array_equal(sh["wave_centers"], wave_centers), "SHAP grid differs from feature grid"
    mean_abs = sh["mean_abs_per_class"]
    assert mean_abs.shape == (3, len(wave_centers)), mean_abs.shape
    k_score = mean_abs[SHAP_K_CLASS_INDEX].astype(np.float64)
    shap_sample = sh["sample_idx"]
    shap_sample_is_val = bool(
        len(shap_sample) == len(val_idx)
        and np.array_equal(shap_sample, np.arange(len(val_idx)))
        and np.array_equal(sh["y_sample"].astype(np.int64), y[val_idx])
    )
    shap_meta: dict[str, Any] = {}
    for name, n in SHAP_TOP_N.items():
        windows, sel, meta = shap_top_n_windows(k_score, wave_centers, gap_mask, n)
        line_sets[name] = windows
        shap_meta[name] = meta

    line_set_info: dict[str, Any] = {}
    for name, windows in line_sets.items():
        m = _window_mask(wave_centers, windows)
        line_set_info[name] = {
            "windows": [[float(lo), float(hi)] for lo, hi in windows],
            "n_windows": len(windows),
            "n_bins": int(m.sum()),
            "segment_bin_counts": _segment_bin_counts(wave_centers, windows),
            "total_width_aa": float(sum(hi - lo for lo, hi in windows)),
            "n_bins_overlapping_catalogued_windows": int((m & catalogued_mask).sum()),
            "n_bins_overlapping_each_catalogued_set": {
                s: int((m & _window_mask(wave_centers, dep_sets[s])).sum()) for s in DEPOSITED_SET_ORDER
            },
            "overlaps_gap": bool((m & gap_mask).any()),
        }
        if name in shap_meta:
            line_set_info[name].update(shap_meta[name])
        print(f"[line set] {name:14s} windows={len(windows):2d} bins={int(m.sum()):3d} "
              f"overlap_catalogued={int((m & catalogued_mask).sum()):3d} "
              f"segments={_segment_bin_counts(wave_centers, windows)}", flush=True)
    n_joint_bins = line_set_info["Joint_all"]["n_bins"]
    print(f"[line set] Joint_all has {line_set_info['Joint_all']['n_windows']} windows and "
          f"{n_joint_bins} bins (sum of the five sets' bins: "
          f"{sum(line_set_info[s]['n_bins'] for s in DEPOSITED_SET_ORDER)})", flush=True)

    fill_train = float(np.nanmedian(X_train[:, ~gap_mask]))
    print(f"[fill] constant fill = train off-gap median = {fill_train:.6f}", flush=True)

    all_nine_mask = gap_mask.copy()
    for w in line_sets.values():
        all_nine_mask |= _window_mask(wave_centers, w)
    five_mask = gap_mask | catalogued_mask

    # ---- runs --------------------------------------------------------------
    run_cfg = {
        "production__gp_interp": dict(fill_mode="gp_interp", fill=None),
        "production__constant": dict(fill_mode="constant", fill=fill_train),
    }
    run_cfg = {k: v for k, v in run_cfg.items() if v["fill_mode"] in args.fills}

    results: dict[str, Any] = {}
    repro: dict[str, Any] = {
        "fields_compared": list(REPRO_FIELDS),
        "tolerance": REPRO_TOL,
        "five_set_rerun_vs_deposited": {},
        "nine_set_run_first_five_sets_vs_deposited": {},
        "nine_set_vs_five_set_deterministic_fields": {},
        "extra_fields_five_set_rerun_vs_deposited": {},
    }
    per_row_all: dict[str, np.ndarray] = {"test_idx": np.asarray(test_idx, dtype=np.int64)}
    timings: dict[str, Any] = {}
    out_json = args.out_dir / f"{args.out_stem}.json"
    out_npz = args.out_dir / f"{args.out_stem}_per_row.npz"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_outputs(stage: str) -> None:
        payload = build_payload(stage)
        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        np.savez(out_npz, **per_row_all)
        logger.info("wrote %s and %s (stage: %s)", out_json, out_npz, stage)

    def build_payload(stage: str) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for s in NEW_SET_ORDER:
            summary[s] = {}
            for cls in ("F", "G", "K"):
                summary[s][cls] = {}
                for run_name, res in results.items():
                    row = next((r for r in res["rows"] if r["line_set"] == s and r["mk_class"] == cls), None)
                    if row is None:
                        continue
                    summary[s][cls][res["fill_mode"]] = {
                        "n_test": row["n_test"],
                        "baseline_recall": row["baseline_acc"],
                        **{k: row[k] for k in SUMMARY_FIELDS},
                        "n_random_controls_succeeded": row["n_random_controls_succeeded"],
                        "p_value_floor": 1.0 / (row["n_random_controls_succeeded"] + 1)
                        if row["n_random_controls_succeeded"] else None,
                        "lost_row_destinations": res["lost_row_destinations"].get(f"{s}__{cls}"),
                        "gained_row_origins": res["gained_row_origins"].get(f"{s}__{cls}"),
                    }
        gp = repro["five_set_rerun_vs_deposited"].get("production__gp_interp")
        gp9 = repro["nine_set_run_first_five_sets_vs_deposited"].get("production__gp_interp")
        details: list[dict[str, Any]] = []
        for block_name, block in (
            ("five_set_rerun_vs_deposited", repro["five_set_rerun_vs_deposited"]),
            ("nine_set_run_first_five_sets_vs_deposited", repro["nine_set_run_first_five_sets_vs_deposited"]),
        ):
            for run_name, cmp in block.items():
                for e in cmp["details"]:
                    details.append({"comparison": block_name, "run": run_name, **e})
        repro_out = {
            "description": (
                "Row-by-row comparison with artifacts/revision/ablation_recalibrated.json. "
                "'passed' and 'max_abs_diff' refer to the five deposited sets re-run alone under "
                "the exact deposited configuration (production__gp_interp). "
                "'passed_literal_nine_set_assertion' is the brief's assertion applied to the first "
                "five sets of the nine-set call; its p-values are not expected to reproduce because "
                "masked_line_ablation forbids the random-window sampler from every bin of every line "
                "set it is given, so the SHAP sets shrink the sampling region and shift the "
                "random-number stream."
            ),
            "fields_compared": list(REPRO_FIELDS),
            "tolerance": REPRO_TOL,
            "passed": None if gp is None else gp["passed"],
            "max_abs_diff": None if gp is None else gp["max_abs_diff"],
            "passed_literal_nine_set_assertion": None if gp9 is None else gp9["passed"],
            "max_abs_diff_literal_nine_set_assertion": None if gp9 is None else gp9["max_abs_diff"],
            "five_set_rerun_vs_deposited": {
                k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in repro["five_set_rerun_vs_deposited"].items()
            },
            "nine_set_run_first_five_sets_vs_deposited": {
                k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in repro["nine_set_run_first_five_sets_vs_deposited"].items()
            },
            "nine_set_vs_five_set_deterministic_fields": repro["nine_set_vs_five_set_deterministic_fields"],
            "extra_fields_five_set_rerun_vs_deposited": {
                k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in repro["extra_fields_five_set_rerun_vs_deposited"].items()
            },
            "details": details,
        }
        return {
            "description": (
                "Positive control for the K-class insensitivity claim: the deposited masked-line "
                "ablation machinery (fixed production LightGBM model, 456 held-out test rows, "
                "class-matched null, bin-matched non-overlapping random windows) applied to four "
                "additional line sets chosen to be maximally K-relevant: the union of all five "
                "catalogued windows (Joint_all) and the 20, 50 and 100 bins with the largest "
                "validation-set mean |SHAP| for the K class. The five deposited sets are run first "
                "in the same call. Two fills: gp_interp (primary, linear interpolation) and "
                "constant (train off-gap median)."
            ),
            "stage": stage,
            "inputs": {
                "features": str(args.features),
                "model": str(args.model),
                "deposited_results": str(args.deposited),
                "n_test": int(len(test_idx)),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_bins": int(X.shape[1]),
                "n_gap_bins": int(gap_mask.sum()),
                "seed": int(args.seed),
                "n_bootstrap": int(args.n_bootstrap),
                "n_random_controls": int(args.n_random_controls),
                "null_mode": NULL_MODE,
                "match_on": MATCH_ON,
                "per_class": True,
                "class_labels": class_labels,
                "class_label_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
                "fills": {
                    "production__gp_interp": {"fill_mode": "gp_interp", "continuum_fill": None,
                                              "note": "linear np.interp across each masked run"},
                    "production__constant": {"fill_mode": "constant", "continuum_fill": fill_train,
                                             "note": "median of the TRAIN matrix outside the gap"},
                },
                "constant_fill_train_median_production": fill_train,
                "step0_confusion_matrix": cm.tolist(),
                "step0_expected": EXPECTED_CONFUSION.tolist(),
                "step0_passed": step0_ok,
                "baseline_accuracy": baseline_acc,
                "shap": {
                    "file": str(args.shap),
                    "key": "mean_abs_per_class",
                    "class_index": SHAP_K_CLASS_INDEX,
                    "class_order": ["F", "G", "K"],
                    "sample": ("all 454 validation rows (sample_idx = positions 0..453 within val_idx; "
                               "y_sample equals y[val_idx])" if shap_sample_is_val
                               else "sample_idx does not match val_idx; see shap file"),
                    "sample_is_validation_set": shap_sample_is_val,
                    "test_rows_used_in_bin_selection": False if shap_sample_is_val else None,
                    "top_n_rule": (
                        "exclude gap bins; rank bins by mean_abs_per_class[2] (K) in stable descending "
                        "order; take the N largest; sort the selected bin indices and group maximal runs "
                        "of consecutive indices; each run becomes the window (wave_centers[start], "
                        "wave_centers[end]) so that _window_mask (inclusive on bin centres) recovers "
                        "exactly the selected bins (asserted)"
                    ),
                    "n_bins_with_positive_k_mean_abs_shap": int((k_score > 0).sum()),
                },
                "joint_all_rule": "concatenation of the windows of the five deposited sets in table order",
                "joint_all_n_bins_verified_with_window_mask": n_joint_bins,
                "joint_all_n_bins_expected_in_brief": 85,
                "joint_all_bin_count_note": (
                    "the brief anticipated 85 bins; _window_mask gives the verified count above "
                    "(29 + 8 + 4 + 2 + 13, the five sets do not overlap)"
                ),
                "forbidden_mask_n_bins": {
                    "five_set_call_(deposited_configuration)": int(five_mask.sum()),
                    "nine_set_call": int(all_nine_mask.sum()),
                },
                "library_versions": _library_versions(),
            },
            "line_sets": line_set_info,
            "reproduction_check": repro_out,
            "runs": results,
            "summary": summary,
            "timings_s": timings,
        }

    dep_runs = deposited["runs"]
    for i, (run_name, cfg) in enumerate(run_cfg.items(), start=1):
        dep_rows = dep_runs[run_name]["rows"]
        fill_mode, fill = cfg["fill_mode"], cfg["fill"]

        # (a) five deposited sets alone: exact deposited configuration.
        rows5: list[AblationRow] | None = None
        if not args.skip_repro5:
            rows5, ds5, pr5, dt5 = run_ablation(
                model, X_test, y_test, wave_centers, dep_sets, class_labels, gap_mask,
                fill_mode, fill, args.n_bootstrap, args.n_random_controls, args.seed,
                tag=f"{i}a/{len(run_cfg)} {run_name} five-set reproduction",
            )
            timings[f"{run_name}__five_set_rerun_s"] = dt5
            for k, v in pr5.items():
                per_row_all[f"{run_name}__repro5__{k}"] = v
            cmp5 = compare_rows(rows5, dep_rows, DEPOSITED_SET_ORDER, REPRO_FIELDS, REPRO_TOL)
            repro["five_set_rerun_vs_deposited"][run_name] = cmp5
            cmp5x = compare_rows(rows5, dep_rows, DEPOSITED_SET_ORDER, EXTRA_FIELDS, REPRO_TOL)
            repro["extra_fields_five_set_rerun_vs_deposited"][run_name] = cmp5x
            print(f"[repro] {run_name} five-set rerun vs deposited: "
                  f"{'EXACT MATCH' if cmp5['passed'] else 'MISMATCH'} on {REPRO_FIELDS} "
                  f"(max |diff| {cmp5['max_abs_diff']:.3g}); extra fields "
                  f"{'match' if cmp5x['passed'] else 'MISMATCH'} (max |diff| {cmp5x['max_abs_diff']:.3g})",
                  flush=True)
            for mm in cmp5["mismatches"] + cmp5x["mismatches"]:
                print(f"[repro]   differs: {mm}", flush=True)
            print_table(rows5, f"{run_name}: five-set reproduction run")
            write_outputs(f"{run_name} five-set rerun done")

        # (b) nine-set call: deposited sets first, then the four new sets.
        rows9, ds9, pr9, dt9 = run_ablation(
            model, X_test, y_test, wave_centers, line_sets, class_labels, gap_mask,
            fill_mode, fill, args.n_bootstrap, args.n_random_controls, args.seed,
            tag=f"{i}b/{len(run_cfg)} {run_name} nine-set positive control",
        )
        timings[f"{run_name}__nine_set_s"] = dt9
        for k, v in pr9.items():
            per_row_all[f"{run_name}__{k}"] = v
        cmp9 = compare_rows(rows9, dep_rows, DEPOSITED_SET_ORDER, REPRO_FIELDS, REPRO_TOL)
        repro["nine_set_run_first_five_sets_vs_deposited"][run_name] = cmp9
        det_fields = ("delta_acc_mean", "n_flip_lost", "n_flip_gained")
        cmp9_det = compare_rows(rows9, dep_rows, DEPOSITED_SET_ORDER, det_fields, REPRO_TOL)
        cmp9["passed_deterministic_fields"] = cmp9_det["passed"]
        cmp9["max_abs_diff_deterministic_fields"] = cmp9_det["max_abs_diff"]
        print(f"[repro] {run_name} nine-set run, first five sets vs deposited: "
              f"{'EXACT MATCH' if cmp9['passed'] else 'MISMATCH'} on {REPRO_FIELDS}; deterministic "
              f"fields {det_fields}: {'match' if cmp9_det['passed'] else 'MISMATCH'}; "
              f"{cmp9['n_mismatching_fields']} mismatching fields, max |diff| {cmp9['max_abs_diff']:.3g}",
              flush=True)
        for mm in cmp9["mismatches"]:
            print(f"[repro]   differs: {mm}", flush=True)
        if rows5 is not None:
            # deterministic fields must agree between the two calls (same mask, same model).
            det_all = ("delta_acc_mean", "n_flip_lost", "n_flip_gained", "masked_acc_mean",
                       "mcnemar_exact_p", "flip_rate_upper95", "mean_delta_true_prob")
            cmp_55 = compare_rows(rows9, [row_to_dict(r) for r in rows5], DEPOSITED_SET_ORDER, det_all, REPRO_TOL)
            repro["nine_set_vs_five_set_deterministic_fields"][run_name] = {
                k: v for k, v in cmp_55.items() if k != "details"
            }
            print(f"[repro] {run_name} nine-set vs five-set deterministic fields: "
                  f"{'identical' if cmp_55['passed'] else 'DIFFER'}", flush=True)
        print_table(rows9, f"{run_name}: nine-set positive-control run")

        yb = pr9["y_pred_base"].astype(np.int64)
        destinations: dict[str, dict[str, int]] = {}
        origins: dict[str, dict[str, int]] = {}
        for s in line_sets:
            ym = pr9[f"y_pred_masked__{s}"].astype(np.int64)
            for c, cname in CLASS_NAMES.items():
                destinations[f"{s}__{cname}"] = lost_row_destinations(y_test, yb, ym, c)
                origins[f"{s}__{cname}"] = gained_row_origins(y_test, yb, ym, c)
        results[run_name] = {
            "continuum": "production",
            "fill_mode": fill_mode,
            "continuum_fill": fill,
            "null_mode": NULL_MODE,
            "match_on": MATCH_ON,
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_random_controls_requested": int(args.n_random_controls),
            "forbidden_mask_n_bins": int(all_nine_mask.sum()),
            "baseline_accuracy": float(np.mean(yb == y_test)),
            "per_class_baseline_recall": {
                CLASS_NAMES[c]: float(np.mean(yb[y_test == c] == c)) for c in CLASS_NAMES
            },
            "draw_stats": ds9,
            "draw_success_rate": {s: d["succeeded"] / d["requested"] for s, d in ds9.items()},
            "p_value_floor_per_line_set": {s: 1.0 / (d["succeeded"] + 1) for s, d in ds9.items()},
            "lost_row_destinations": destinations,
            "gained_row_origins": origins,
            "rows": [row_to_dict(r) for r in rows9],
            "five_set_rerun_rows": None if rows5 is None else [row_to_dict(r) for r in rows5],
            "five_set_rerun_draw_stats": None if rows5 is None else ds5,
            "runtime_s_nine_set": dt9,
            "runtime_s_five_set_rerun": None if rows5 is None else timings[f"{run_name}__five_set_rerun_s"],
        }
        write_outputs(f"{run_name} nine-set run done")

    timings["total_script_s"] = time.perf_counter() - t_script
    write_outputs("complete")

    # ---- final console summary -------------------------------------------
    print("\n=== positive-control summary (new sets) ===")
    print(f"{'set':14s} {'cls':3s} {'fill':9s} {'delta':>8s} {'lost/gained':>11s} {'McNemar p':>11s} "
          f"{'CP upper95':>10s} {'p_MC':>10s}")
    for s in NEW_SET_ORDER:
        for cls in ("F", "G", "K"):
            for run_name, res in results.items():
                row = next(r for r in res["rows"] if r["line_set"] == s and r["mk_class"] == cls)
                print(f"{s:14s} {cls:3s} {res['fill_mode']:9s} {row['delta_acc_mean']:+8.3f} "
                      f"{row['n_flip_lost']:4d}/{row['n_flip_gained']:<6d} {row['mcnemar_exact_p']:11.4g} "
                      f"{row['flip_rate_upper95']:10.4f} {row['p_value_vs_random']:10.4g}")
    gp = repro["five_set_rerun_vs_deposited"].get("production__gp_interp")
    if gp is not None:
        print(f"\nreproduction check (five deposited sets, exact configuration, production__gp_interp): "
              f"{'PASSED' if gp['passed'] else 'FAILED'}, max |diff| = {gp['max_abs_diff']}")
    gp9 = repro["nine_set_run_first_five_sets_vs_deposited"].get("production__gp_interp")
    if gp9 is not None:
        print(f"literal nine-set assertion (first five sets of the nine-set call): "
              f"{'PASSED' if gp9['passed'] else 'FAILED'}, {gp9['n_mismatching_fields']} fields differ, "
              f"max |diff| = {gp9['max_abs_diff']}; deterministic fields "
              f"{'match' if gp9['passed_deterministic_fields'] else 'DIFFER'}")
    print(f"total runtime {timings['total_script_s']:.0f} s")
    return 0


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod in ("numpy", "scipy", "sklearn", "lightgbm", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            out[mod] = "unavailable"
    out["python"] = sys.version.split()[0]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
