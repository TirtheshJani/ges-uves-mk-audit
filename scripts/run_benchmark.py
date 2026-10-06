#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Run the Pickles template-matching benchmark against the LightGBM classifier.

The model is trained on F/G/K only (the A class is dropped at training
time). All 131 STScI Pickles templates are loaded by
``src.interpret.benchmark.load_pickles_library``; templates outside A/F/G/K
collapse to ``OTHER`` and are removed by the FGK filter below. The FGK-only
agreement floor of 0.55 is applied here and written to
``artifacts/benchmark/benchmark_report.json`` under ``decision_33_gate``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.interpret.benchmark import (
    benchmark_report,
    best_template_per_spectrum,
    load_pickles_library,
    save_report,
)
from src.interpret.classifier import load_model
from src.interpret.lines import ALLOWED_MK_CLASSES

logger = logging.getLogger(__name__)


def _filter_to_fgk(
    y_pickles: np.ndarray, y_model: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop rows whose Pickles label is OTHER.

    Returns ``(y_pickles_fgk, y_model_fgk, n_dropped_other)``. The model
    cannot predict ``OTHER`` since the classifier head only emits F/G/K,
    so the comparison is meaningful only on rows where Pickles itself
    proposed an FGK label.
    """
    y_pickles = np.asarray(y_pickles)
    y_model = np.asarray(y_model)
    if y_pickles.shape != y_model.shape:
        raise ValueError(
            f"shape mismatch: y_pickles {y_pickles.shape} vs y_model {y_model.shape}"
        )
    mask = y_pickles != "OTHER"
    n_dropped = int((~mask).sum())
    return y_pickles[mask], y_model[mask], n_dropped


def _decision_33_gate(
    agreement_rate: float,
    n_compared: int,
    floor: float = 0.55,
    min_n: int = 100,
) -> dict:
    """Apply the FGK-only agreement gate.

    Pass when ``agreement_rate >= floor`` AND ``n_compared > min_n``.
    """
    status = "PASS" if (agreement_rate >= floor and n_compared > min_n) else "FAIL"
    return {
        "required": float(floor),
        "observed": float(agreement_rate),
        "n_compared": int(n_compared),
        "min_n_compared": int(min_n),
        "status": status,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--pickles-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--restrict-fgk",
        dest="restrict_fgk",
        action="store_true",
        default=True,
        help="Drop OTHER rows before computing the FGK-only metrics (default).",
    )
    p.add_argument(
        "--no-restrict-fgk",
        dest="restrict_fgk",
        action="store_false",
        help="Disable the FGK filter (full A/F/G/K/OTHER comparison).",
    )
    p.add_argument(
        "--continuum-method",
        choices=("median_filter_200", "median_filter_50", "polynomial_n5"),
        default="median_filter_200",
        help=(
            "Pickles continuum-normalisation method. median_filter_200 is the production default; "
            "median_filter_50 narrows the window to preserve M-class TiO "
            "bandhead structure; polynomial_n5 is an iteratively sigma-clipped "
            "5th-order Legendre continuum fit."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    payload = np.load(args.features, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    wc = payload["wave_centers"]
    test_idx = payload["test_idx"]

    templates, pickles_stats = load_pickles_library(
        args.pickles_dir, wc, continuum_method=args.continuum_method,
    )
    pickles_stats["continuum_method"] = args.continuum_method
    mk_per_template = np.array([t.mk_class for t in templates])

    best_t = best_template_per_spectrum(X[test_idx], templates)
    y_pickles = mk_per_template[best_t]

    model = load_model(args.model)
    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    int_to_label = {i: ALLOWED_MK_CLASSES[i] for i in present}
    y_model = np.array([int_to_label[int(i)] for i in model.predict(X[test_idx])])

    n_dropped_other = 0
    if args.restrict_fgk:
        y_pickles_fgk, y_model_fgk, n_dropped_other = _filter_to_fgk(y_pickles, y_model)
        logger.info(
            "FGK filter: dropped %d OTHER rows; %d FGK rows remain.",
            n_dropped_other, len(y_pickles_fgk),
        )
    else:
        y_pickles_fgk = y_pickles
        y_model_fgk = y_model

    report = benchmark_report(
        y_pred_model=y_model_fgk,
        y_pickles_mk=y_pickles_fgk,
        class_labels=class_labels,
    )
    report["n_dropped_other"] = int(n_dropped_other)
    report["restrict_fgk"] = bool(args.restrict_fgk)
    report["pickles_loader_stats"] = pickles_stats
    report["decision_33_gate"] = _decision_33_gate(
        agreement_rate=report["agreement_rate"],
        n_compared=report["n_compared"],
        floor=0.55,
        min_n=100,
    )
    save_report(args.out_dir, report)

    out_dir = Path(args.out_dir)
    with (out_dir / "benchmark_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"agreement (FGK-only): {report['agreement_rate']:.3f}")
    print(f"macro-F1 (FGK): {report['macro_f1_fgk']:.3f}")
    print(f"n_compared: {report['n_compared']}")
    print(f"n_dropped_other: {report['n_dropped_other']}")
    print(f"gate: {report['decision_33_gate']['status']}")


if __name__ == "__main__":
    main()
