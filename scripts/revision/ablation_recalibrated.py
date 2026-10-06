#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Recalibrated masked-line ablation table under two continua and two fills.

The production LightGBM model (``artifacts/lgbm_mk.pkl``) is held fixed and
the masked-line ablation of :func:`src.interpret.ablation.masked_line_ablation`
is run on the 456 held-out test rows for all five line sets (the four MK sets
in ``src.interpret.lines.LINE_SETS`` plus the Fe i / Cr i K-class multiplet
set of ``scripts/appendix_a2_polynomial_diagnostics.py``) and all three
classes, under a 2 x 2 design:

  continuum   production      unchanged pipeline-normalised test features
              polynomial_n5   every test row re-normalised with the
                              sigma-clipped 5th-order Legendre fit
                              (``polynomial_renormalise`` imported from
                              ``scripts/revision/scale_decomposition.py``)
  fill        gp_interp       PRIMARY: linear interpolation across each
                              masked run from the unmasked bins of the row
              constant        SENSITIVITY: masked bins set to the median of
                              the TRAIN matrix outside the inter-chip gap,
                              computed under the same continuum as the run

Reference-set settings are the same for every run: class-matched null,
bin-matched non-overlapping random windows, 5000 random windows, 2000
bootstrap resamples, seed 42, gap bins forbidden to the random sampler. The
Monte-Carlo p-value is the Phipson-Smyth ``(b + 1) / (B + 1)`` estimator, so
its floor is ``1 / (B_succeeded + 1)``.

Outputs (under ``artifacts/revision/``):

  ablation_recalibrated.json      full table, draw statistics, headline
                                  gate, K-class summary, lost-row
                                  destinations
  ablation_recalibrated.csv       one row per (run, line set, class)
  ablation_recalibrated_per_row.npz
                                  per-row predictions and probabilities for
                                  every run, keys prefixed ``<run>__``
  ../figures/ablation_bars.pdf and ../../submission/ablation_bars.pdf
                                  headline/pivot bar chart from the primary
                                  run under the production continuum

Run from the repo root: ``python scripts/revision/ablation_recalibrated.py``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.ablation import AblationRow, masked_line_ablation  # noqa: E402
from src.interpret.classifier import load_model  # noqa: E402
from src.interpret.lines import ALLOWED_MK_CLASSES, LINE_SETS  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_NAMES: dict[int, str] = {1: "F", 2: "G", 3: "K"}
CLASS_INTS: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}
SEED = 42
N_BOOTSTRAP = 2000
N_RANDOM_CONTROLS = 5000
NULL_MODE = "class_matched"
MATCH_ON = "bins"
PRIMARY_RUN = "production__gp_interp"
CONSTANT_RUN = "production__constant"
DEPOSITED_TEST_MEDIAN_FILL = 0.9158  # constant fill used by the deposited ablation

# Display names for the figure (mirrors src.interpret.plotting._LINE_SET_DISPLAY).
LINE_SET_DISPLAY: dict[str, str] = {
    "H_balmer": "H Balmer",
    "Mg_b": "Mg b",
    "Na_D": "Na D",
    "Ca_I": "Ca I",
    "Fe_Cr": "Fe/Cr",
}


def load_script_module(relative_path: str, name: str) -> ModuleType:
    """Import a script that is not part of a package by file path."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_line_sets() -> dict[str, list[tuple[float, float]]]:
    """The four MK line sets plus the Fe i / Cr i multiplet set, in table order."""
    a2 = load_script_module("scripts/appendix_a2_polynomial_diagnostics.py", "appendix_a2")
    sets = {name: [tuple(w) for w in LINE_SETS[name]] for name in ("H_balmer", "Mg_b", "Na_D", "Ca_I")}
    sets["Fe_Cr"] = [tuple(w) for w in a2.FE_CR_LINE_SET]
    return sets


def polynomial_renormalise(X: np.ndarray, wave_centers: np.ndarray, order: int = 5) -> np.ndarray:
    """Sigma-clipped Legendre re-normalisation, delegated to ``scale_decomposition.py``."""
    sd = load_script_module("scripts/revision/scale_decomposition.py", "scale_decomposition")
    return sd.polynomial_renormalise(X, wave_centers, order)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not np.isfinite(v) else v
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def row_to_dict(row: AblationRow) -> dict[str, Any]:
    """Every AblationRow field, JSON-safe."""
    return {k: _jsonable(v) for k, v in asdict(row).items()}


def lost_row_destinations(
    y_test: np.ndarray, y_pred_base: np.ndarray, y_pred_masked: np.ndarray, cls: int,
) -> dict[str, int]:
    """Destination class counts for rows of ``cls`` that flip from correct to incorrect."""
    rows = (y_test == cls) & (y_pred_base == cls) & (y_pred_masked != cls)
    dest = y_pred_masked[rows]
    return {CLASS_NAMES[c]: int(np.sum(dest == c)) for c in CLASS_NAMES if c != cls}


def evaluate_gate(rows: list[AblationRow], gate_module: ModuleType) -> dict[str, Any]:
    """Pre-specified headline gate, evaluated with the production driver's function."""
    gate = gate_module.evaluate_headline_gate(rows)
    gate["criterion"] = (
        f"delta_acc_ci_high < {gate_module.GATE_DELTA_CI_HIGH_MAX} and "
        f"p_value_vs_random <= {gate_module.GATE_PVALUE_MAX:.5f} "
        f"(family-wise alpha {gate_module.GATE_FAMILYWISE_ALPHA}, Bonferroni over 3 pairs)"
    )
    gate["gate_threshold_per_pair"] = float(gate_module.GATE_PVALUE_MAX)
    gate["headline_pairs"] = [{"line_set": a, "mk_class": b} for a, b in gate_module.HEADLINE_PAIRS]
    gate["pivot_pairs"] = [{"line_set": a, "mk_class": b} for a, b in gate_module.PIVOT_PAIRS]
    return gate


def format_p(p: float) -> str:
    """Exact-decimal p-value string (never a '<' floor)."""
    if not np.isfinite(p):
        return "p = n/a"
    return f"p = {p:.3f}" if p >= 0.01 else f"p = {p:.4f}"


def plot_ablation_bars(
    rows: list[dict[str, Any]],
    headline_pairs: list[tuple[str, str]],
    pivot_pairs: list[tuple[str, str]],
    out_paths: list[Path],
    n_random_controls: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> None:
    """Delta-accuracy bar chart for the headline (solid) and pivot (hatched) pairs.

    Bars carry 95 percent percentile-bootstrap CIs; each bar is annotated with
    its delta and its Monte-Carlo p-value written as an exact decimal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, mid, small = 9.0, 8.0, 7.0
    plt.rcParams.update({
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": small, "ytick.labelsize": small, "legend.fontsize": mid,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "font.family": "sans-serif",
    })
    lookup = {(r["line_set"], r["mk_class"]): r for r in rows}
    ordered: list[dict[str, Any]] = []
    for ls, cls in headline_pairs:
        ordered.append({**lookup[(ls, cls)], "is_pivot": False})
    for ls, cls in pivot_pairs:
        ordered.append({**lookup[(ls, cls)], "is_pivot": True})

    labels = [
        f"{LINE_SET_DISPLAY.get(r['line_set'], r['line_set'])}\non {r['mk_class']}\n(n = {r['n_test']})"
        for r in ordered
    ]
    means = np.array([r["delta_acc_mean"] for r in ordered])
    lo = np.array([r["delta_acc_ci_low"] for r in ordered])
    hi = np.array([r["delta_acc_ci_high"] for r in ordered])
    pvals = [r["p_value_vs_random"] for r in ordered]
    is_pivot = [r["is_pivot"] for r in ordered]

    c_head, c_pivot = "#1f5fa8", "#d55e00"  # blue / vermilion, distinct in deuteranopia
    colors = [c_pivot if pv else c_head for pv in is_pivot]
    hatches = ["///" if pv else "" for pv in is_pivot]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(ordered))
    bars = ax.bar(
        x, means, color=colors, alpha=0.9,
        yerr=[np.abs(means - lo), np.abs(hi - means)], capsize=4,
        edgecolor="black", lw=0.6, error_kw={"lw": 0.9},
    )
    for bar, h, m in zip(bars, hatches, means):
        bar.set_hatch(h)
        if abs(m) < 1e-9:  # zero-valued bar: visible stub at the baseline
            ax.plot(bar.get_x() + bar.get_width() / 2, 0.0, marker="o", ms=4,
                    color=bar.get_facecolor(), mec="black", mew=0.6, zorder=3)
    ax.axhline(0.0, color="black", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ recall (masked $-$ baseline)")
    ax.set_title(
        "Masking Balmer and Mg b lines lowers F and G recall; K recall is unchanged",
        loc="left",
    )
    ax.grid(axis="y", color="lightgrey", lw=0.5, alpha=0.6, zorder=-1)

    span = max(hi.max(), 0.0) - min(lo.min(), 0.0)
    span = span if span > 0 else 1.0
    ax.set_ylim(min(lo.min(), 0.0) - 0.26 * span, max(hi.max(), 0.0) + 0.24 * span)
    for xi, m, l, h, pv in zip(x, means, lo, hi, pvals):
        # Annotations sit just outside the error-bar tip on the side the bar
        # points to, so they never overlap the bar itself.
        if m < 0:
            tip = min(m, l)
            ax.text(xi, tip - 0.035 * span, f"$\\Delta$ = {m:+.3f}", ha="center", va="top",
                    fontsize=mid, fontweight="bold")
            ax.text(xi, tip - 0.105 * span, format_p(pv), ha="center", va="top",
                    fontsize=mid, color="dimgrey")
        else:
            tip = max(m, h)
            ax.text(xi, tip + 0.035 * span, f"$\\Delta$ = {m:+.3f}", ha="center", va="bottom",
                    fontsize=mid, fontweight="bold")
            ax.text(xi, tip + 0.105 * span, format_p(pv), ha="center", va="bottom",
                    fontsize=mid, color="dimgrey")
    head_proxy = plt.Rectangle((0, 0), 1, 1, fc=c_head, alpha=0.9, edgecolor="black", lw=0.6)
    pivot_proxy = plt.Rectangle((0, 0), 1, 1, fc=c_pivot, alpha=0.9, hatch="///",
                                edgecolor="black", lw=0.6)
    ax.legend([head_proxy, pivot_proxy],
              ["pre-specified headline pair", "pivot pair (near-zero predicted)"],
              loc="upper left", frameon=False)
    ax.text(
        0.995, 0.02,
        f"95% bootstrap CI ({n_bootstrap} resamples); p vs {n_random_controls} class-matched\n"
        f"bin-matched random windows, floor p = {1.0 / (n_random_controls + 1):.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=small, color="dimgrey",
    )
    ax.tick_params(axis="x", which="major", pad=5)
    fig.tight_layout()
    for out in out_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=300)
        logger.info("wrote %s", out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/revision"))
    p.add_argument("--figure-dirs", type=Path, nargs="*",
                   default=[Path("artifacts/figures"), Path("submission")])
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--n-random-controls", type=int, default=N_RANDOM_CONTROLS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--polynomial-order", type=int, default=5)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--figure-only", action="store_true",
                   help="re-draw the bar chart from the existing JSON without re-running the ablation")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    if args.figure_only:
        with (args.out_dir / "ablation_recalibrated.json").open() as f:
            payload = json.load(f)
        gate = payload["headline_gate"]
        plot_ablation_bars(
            payload["runs"][payload["primary_run"]]["rows"],
            [(d["line_set"], d["mk_class"]) for d in gate["headline_pairs"]],
            [(d["line_set"], d["mk_class"]) for d in gate["pivot_pairs"]],
            [d / "ablation_bars.pdf" for d in args.figure_dirs],
            n_random_controls=int(min(
                d["succeeded"] for d in payload["runs"][payload["primary_run"]]["draw_stats"].values()
            )),
            n_bootstrap=int(payload["inputs"]["n_bootstrap"]),
        )
        return 0

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"].astype(bool)
    train_idx, test_idx = fp["train_idx"], fp["test_idx"]
    X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
    model = load_model(args.model)
    class_labels = list(ALLOWED_MK_CLASSES)  # index 1 -> F, 2 -> G, 3 -> K
    line_sets = all_line_sets()
    gate_module = load_script_module("scripts/ablation.py", "ablation_driver")

    logger.info("polynomial_n%d re-normalisation of %d test and %d train rows",
                args.polynomial_order, len(X_test), len(X_train))
    X_test_poly = polynomial_renormalise(X_test, wave_centers, args.polynomial_order)
    X_train_poly = polynomial_renormalise(X_train, wave_centers, args.polynomial_order)

    fill_train_prod = float(np.nanmedian(X_train[:, ~gap_mask]))
    fill_test_prod = float(np.nanmedian(X_test[:, ~gap_mask]))
    fill_train_poly = float(np.nanmedian(X_train_poly[:, ~gap_mask]))
    fill_test_poly = float(np.nanmedian(X_test_poly[:, ~gap_mask]))

    runs: dict[str, dict[str, Any]] = {
        "production__gp_interp": dict(X=X_test, continuum="production", fill_mode="gp_interp", fill=None),
        "production__constant": dict(X=X_test, continuum="production", fill_mode="constant", fill=fill_train_prod),
        "polynomial_n5__gp_interp": dict(X=X_test_poly, continuum="polynomial_n5", fill_mode="gp_interp", fill=None),
        "polynomial_n5__constant": dict(X=X_test_poly, continuum="polynomial_n5", fill_mode="constant", fill=fill_train_poly),
    }

    results: dict[str, dict[str, Any]] = {}
    rows_by_run: dict[str, list[AblationRow]] = {}
    per_row_all: dict[str, np.ndarray] = {"test_idx": np.asarray(test_idx, dtype=np.int64)}
    csv_rows: list[dict[str, Any]] = []
    for i, (run_name, cfg) in enumerate(runs.items(), start=1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(runs)}] run {run_name}: continuum={cfg['continuum']} "
              f"fill_mode={cfg['fill_mode']} fill={cfg['fill']}", flush=True)
        draw_stats: dict[str, dict[str, Any]] = {}
        per_row: dict[str, np.ndarray] = {}
        rows = masked_line_ablation(
            model, cfg["X"], y_test, wave_centers, line_sets, class_labels,
            per_class=True,
            n_bootstrap=args.n_bootstrap,
            n_random_controls=args.n_random_controls,
            seed=args.seed,
            gap_mask=gap_mask,
            continuum_fill=cfg["fill"],
            draw_stats_out=draw_stats,
            per_row_out=per_row,
            fill_mode=cfg["fill_mode"],
            null_mode=NULL_MODE,
            match_on=MATCH_ON,
        )
        rows_by_run[run_name] = rows
        for k, v in per_row.items():
            per_row_all[f"{run_name}__{k}"] = v
        y_pred_base = per_row["y_pred_base"].astype(np.int64)
        baseline_acc = float(np.mean(y_pred_base == y_test))
        per_class_baseline = {
            CLASS_NAMES[c]: float(np.mean(y_pred_base[y_test == c] == c)) for c in CLASS_NAMES
        }
        destinations: dict[str, dict[str, int]] = {}
        for set_name in line_sets:
            y_pred_masked = per_row[f"y_pred_masked__{set_name}"].astype(np.int64)
            for c, cname in CLASS_NAMES.items():
                destinations[f"{set_name}__{cname}"] = lost_row_destinations(
                    y_test, y_pred_base, y_pred_masked, c,
                )
        b_succ = {s: int(d["succeeded"]) for s, d in draw_stats.items()}
        results[run_name] = {
            "continuum": cfg["continuum"],
            "fill_mode": cfg["fill_mode"],
            "continuum_fill": cfg["fill"],
            "null_mode": NULL_MODE,
            "match_on": MATCH_ON,
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_random_controls_requested": int(args.n_random_controls),
            "baseline_accuracy": baseline_acc,
            "per_class_baseline_recall": per_class_baseline,
            "draw_stats": draw_stats,
            "p_value_floor_per_line_set": {s: 1.0 / (b + 1) for s, b in b_succ.items()},
            "lost_row_destinations": destinations,
            "rows": [row_to_dict(r) for r in rows],
            "runtime_s": float(time.perf_counter() - t0),
        }
        for r in rows:
            csv_rows.append({"run": run_name, "continuum": cfg["continuum"], **row_to_dict(r)})
        print(f"    done in {time.perf_counter() - t0:.0f} s; baseline acc {baseline_acc:.4f}; "
              f"draws succeeded {b_succ}", flush=True)

    # --- headline gate on the primary run ----------------------------------
    gate = evaluate_gate(rows_by_run[PRIMARY_RUN], gate_module)
    gate["run"] = PRIMARY_RUN
    gate["p_value_floor_per_line_set"] = results[PRIMARY_RUN]["p_value_floor_per_line_set"]
    gate["draw_success_rates"] = {
        s: {"requested": d["requested"], "succeeded": d["succeeded"],
            "rate": d["succeeded"] / d["requested"], "n_draws_bin_matched": d["n_draws_bin_matched"]}
        for s, d in results[PRIMARY_RUN]["draw_stats"].items()
    }
    gate["gate_under_other_runs"] = {
        run_name: {
            "gate_status": g["gate_status"],
            "n_pairs_passed": g["n_pairs_passed"],
            "pairs": [{k: p_[k] for k in ("line_set", "mk_class", "delta_acc_mean",
                                           "delta_acc_ci_high", "p_value_vs_random", "passed")}
                      for p_ in g["pairs_evaluated"]],
        }
        for run_name, g in ((rn, evaluate_gate(rows_by_run[rn], gate_module)) for rn in runs if rn != PRIMARY_RUN)
    }

    # --- K-class summary across runs ---------------------------------------
    summary_fields = (
        "delta_acc_mean", "delta_acc_ci_low", "delta_acc_ci_high", "n_flip_lost", "n_flip_gained",
        "mcnemar_exact_p", "mean_delta_true_prob", "flip_rate_upper95", "p_value_vs_random",
        "n_random_controls_succeeded",
    )
    k_class_summary: dict[str, dict[str, Any]] = {}
    for set_name in line_sets:
        k_class_summary[set_name] = {}
        for run_name, rows in rows_by_run.items():
            row = next(r for r in rows if r.line_set == set_name and r.mk_class == "K")
            d = row_to_dict(row)
            k_class_summary[set_name][run_name] = {
                "continuum": results[run_name]["continuum"],
                "fill_mode": results[run_name]["fill_mode"],
                "n_test": d["n_test"],
                "baseline_recall": d["baseline_acc"],
                **{k: d[k] for k in summary_fields},
                "lost_row_destinations": results[run_name]["lost_row_destinations"][f"{set_name}__K"],
            }

    def _pair(run_name: str, set_name: str, cls: str) -> dict[str, Any]:
        row = next(r for r in rows_by_run[run_name] if r.line_set == set_name and r.mk_class == cls)
        d = row_to_dict(row)
        d["lost_row_destinations"] = results[run_name]["lost_row_destinations"][f"{set_name}__{cls}"]
        return d

    focus_destinations = {
        run_name: {
            "Mg_b__G": results[run_name]["lost_row_destinations"]["Mg_b__G"],
            "H_balmer__F": results[run_name]["lost_row_destinations"]["H_balmer__F"],
        }
        for run_name in runs
    }
    ca_i_k_poly = {
        run_name: {
            "p_value_vs_random": _pair(run_name, "Ca_I", "K")["p_value_vs_random"],
            "delta_acc_mean": _pair(run_name, "Ca_I", "K")["delta_acc_mean"],
            "n_flip_lost": _pair(run_name, "Ca_I", "K")["n_flip_lost"],
            "n_flip_gained": _pair(run_name, "Ca_I", "K")["n_flip_gained"],
            "p_value_floor": results[run_name]["p_value_floor_per_line_set"]["Ca_I"],
        }
        for run_name in ("polynomial_n5__gp_interp", "polynomial_n5__constant")
    }

    payload = {
        "description": (
            "Masked-line ablation of the fixed production LightGBM classifier on the "
            "456 held-out test rows: five line sets x three classes under two continua "
            "(production, polynomial_n5) and two fills (gp_interp = linear interpolation, "
            "primary; constant = train off-gap median, sensitivity). Class-matched null, "
            "bin-matched non-overlapping random windows."
        ),
        "inputs": {
            "features": str(args.features),
            "model": str(args.model),
            "n_test": int(len(test_idx)),
            "n_train": int(len(train_idx)),
            "n_bins": int(X.shape[1]),
            "n_gap_bins": int(gap_mask.sum()),
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_random_controls": int(args.n_random_controls),
            "null_mode": NULL_MODE,
            "match_on": MATCH_ON,
            "primary_fill_mode": "gp_interp (linear np.interp across each masked run)",
            "sensitivity_fill_mode": "constant",
            "continuum_fill_train_median_production": fill_train_prod,
            "continuum_fill_test_median_production": fill_test_prod,
            "continuum_fill_test_median_production_deposited_run": DEPOSITED_TEST_MEDIAN_FILL,
            "continuum_fill_train_median_polynomial_n5": fill_train_poly,
            "continuum_fill_test_median_polynomial_n5": fill_test_poly,
            "constant_fill_used": {
                "production__constant": fill_train_prod,
                "polynomial_n5__constant": fill_train_poly,
            },
            "polynomial_order": int(args.polynomial_order),
            "polynomial_renormalise_source": "scripts/revision/scale_decomposition.py::polynomial_renormalise "
                                             "(src.interpret.benchmark.continuum_normalize, method polynomial_n5, "
                                             "sigma_clip 3.0, 5 iterations, fit over all bins including the gap)",
            "fe_cr_line_set_source": "scripts/appendix_a2_polynomial_diagnostics.py::FE_CR_LINE_SET",
            "class_label_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
        },
        "line_sets": {k: [list(w) for w in v] for k, v in line_sets.items()},
        "primary_run": PRIMARY_RUN,
        "runs": results,
        "headline_gate": gate,
        "k_class_summary": k_class_summary,
        "lost_row_destinations_focus_pairs": focus_destinations,
        "ca_i_k_under_polynomial_continuum": ca_i_k_poly,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_recalibrated.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    csv_path = args.out_dir / "ablation_recalibrated.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    npz_path = args.out_dir / "ablation_recalibrated_per_row.npz"
    np.savez(npz_path, **per_row_all)
    logger.info("wrote %s, %s, %s", json_path, csv_path, npz_path)

    if not args.no_figure:
        plot_ablation_bars(
            results[PRIMARY_RUN]["rows"],
            [(a, b) for a, b in gate_module.HEADLINE_PAIRS],
            [(a, b) for a, b in gate_module.PIVOT_PAIRS],
            [d / "ablation_bars.pdf" for d in args.figure_dirs],
            n_random_controls=int(min(
                d["succeeded"] for d in results[PRIMARY_RUN]["draw_stats"].values()
            )),
            n_bootstrap=int(args.n_bootstrap),
        )
    print(f"headline gate ({PRIMARY_RUN}): {gate['gate_status']} "
          f"({gate['n_pairs_passed']}/{gate['n_pairs_total']} pairs passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
