#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Render the production figures from artifact files.

Outputs (all 300 DPI PDFs under ``--out-dir``):

  * importance_main.pdf            (plot_main_figure)
  * confusion_matrix_test.pdf      (plot_confusion_matrix from metrics.json)
  * confusion_matrix_pickles.pdf   (plot_confusion_matrix from benchmark_report.json, FGK-only)
  * ablation_bars.pdf              (plot_ablation_bars from gate_eval.json)

Legacy overlays (``importance_overlay.pdf``,
``importance_overlay_per_class.pdf``) are also rendered for the dossier.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.interpret.lines import ALLOWED_MK_CLASSES
from src.interpret.plotting import (
    plot_ablation_bars,
    plot_confusion_matrix,
    plot_main_figure,
    plot_per_class_importance,
    plot_per_class_overlay,
    plot_summary_overlay,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with Path(path).open("r") as fh:
        return json.load(fh)


def _representative_spectrum(
    X: np.ndarray, y: np.ndarray, present: list[int], class_labels: list[str],
    rep_cls: str,
) -> np.ndarray:
    """Return the per-bin median spectrum for ``rep_cls`` from the labelled set."""
    if rep_cls in class_labels:
        rep_int = present[class_labels.index(rep_cls)]
        rep = np.nanmedian(X[y == rep_int], axis=0)
    else:
        rep = np.nanmedian(X, axis=0)
    return np.asarray(rep)


def _gate_eval_to_pairs(
    gate_eval: dict,
) -> tuple[list[dict], list[tuple[str, str]], list[tuple[str, str]]]:
    """Pull ablation rows + headline + pivot pair lists from gate_eval.json."""
    pairs_evaluated: list[dict] = list(gate_eval.get("pairs_evaluated", []))
    pivots: list[dict] = list(gate_eval.get("pivot_confirmed", []))
    headline_pairs = [
        (p["line_set"], p["mk_class"])
        for p in gate_eval.get("headline_pairs", [])
    ]
    pivot_pairs = [
        (p["line_set"], p["mk_class"])
        for p in gate_eval.get("pivot_pairs", [])
    ]
    rows = pairs_evaluated + pivots
    return rows, headline_pairs, pivot_pairs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--importance", required=True, type=Path)
    p.add_argument("--shap", required=True, type=Path)
    p.add_argument("--ablation", required=True, type=Path,
                   help="Path to artifacts/ablation/gate_eval.json")
    p.add_argument("--metrics", required=True, type=Path,
                   help="Path to artifacts/metrics.json")
    p.add_argument("--benchmark", required=True, type=Path,
                   help="Path to artifacts/benchmark/benchmark_report.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--representative-class", default="G")
    p.add_argument("--top-k-per-class", type=int, default=10)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    feats = np.load(args.features, allow_pickle=False)
    X = feats["X"]
    y = feats["y"].astype(np.int64)
    wc = feats["wave_centers"]
    gap_mask = feats["gap_mask"].astype(bool)

    imp = np.load(args.importance, allow_pickle=False)["importance_mean"]
    shap_npz = np.load(args.shap, allow_pickle=False)
    mean_abs_per_class = shap_npz["mean_abs_per_class"]

    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]

    rep_spectrum = _representative_spectrum(
        X, y, present, class_labels, args.representative_class,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gate_eval = _load_json(args.ablation)
    metrics = _load_json(args.metrics)
    benchmark = _load_json(args.benchmark)

    rows, headline_pairs, pivot_pairs = _gate_eval_to_pairs(gate_eval)

    # 1. Main figure: spectrum + per-class SHAP top-K + MK lines + gap shading
    #    + ablation bars in side panel.
    main_rows: list[dict[str, Any]] = []
    for ls, cls in headline_pairs:
        for row in rows:
            if row.get("line_set") == ls and row.get("mk_class") == cls:
                main_rows.append({**row, "is_pivot": False})
                break
    for ls, cls in pivot_pairs:
        for row in rows:
            if row.get("line_set") == ls and row.get("mk_class") == cls:
                main_rows.append({**row, "is_pivot": True})
                break

    plot_main_figure(
        wave_centers=wc,
        representative_spectrum=rep_spectrum,
        shap_per_class=mean_abs_per_class,
        class_labels=class_labels,
        ablation_rows=main_rows,
        gap_mask=gap_mask,
        out_path=args.out_dir / "importance_main.pdf",
        top_k_per_class=args.top_k_per_class,
    )

    # 2. Confusion matrix on test set (from metrics.json).
    cm_test = np.asarray(metrics.get("confusion_matrix"))
    test_labels = list(metrics.get("class_labels", class_labels))
    plot_confusion_matrix(
        cm=cm_test,
        labels=test_labels,
        title="LightGBM (test set)",
        out_path=args.out_dir / "confusion_matrix_test.pdf",
    )

    # 3. Confusion matrix from Pickles benchmark (FGK-only).
    cm_pickles_full = np.asarray(benchmark.get("confusion_matrix"))
    bench_labels = list(benchmark.get("labels", class_labels + ["OTHER"]))
    fgk_idx = [i for i, lab in enumerate(bench_labels) if lab in {"F", "G", "K"}]
    if cm_pickles_full.size and fgk_idx:
        cm_pickles_fgk = cm_pickles_full[np.ix_(fgk_idx, fgk_idx)]
        fgk_labels = [bench_labels[i] for i in fgk_idx]
    else:
        cm_pickles_fgk = cm_pickles_full
        fgk_labels = bench_labels
    plot_confusion_matrix(
        cm=cm_pickles_fgk,
        labels=fgk_labels,
        title="Pickles vs model (FGK-only)",
        out_path=args.out_dir / "confusion_matrix_pickles.pdf",
    )

    # 4. Ablation bar chart (headline + pivot) from gate_eval.json.
    plot_ablation_bars(
        ablation_rows=rows,
        headline_pairs=headline_pairs,
        pivot_pairs=pivot_pairs,
        out_path=args.out_dir / "ablation_bars.pdf",
        n_random_controls=int(gate_eval.get("n_random_controls", 500)),
    )

    # 5. Legacy overlays for the dossier.
    plot_summary_overlay(
        wc, imp, mean_abs_per_class, class_labels, rep_spectrum,
        args.out_dir / "importance_overlay.pdf",
    )
    plot_per_class_overlay(
        wc, mean_abs_per_class, class_labels,
        args.out_dir / "importance_overlay_per_class.pdf",
    )
    plot_per_class_importance(
        wc, mean_abs_per_class, class_labels, gap_mask,
        args.out_dir / "importance_per_class_gapmask.pdf",
    )

    print(f"wrote figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
