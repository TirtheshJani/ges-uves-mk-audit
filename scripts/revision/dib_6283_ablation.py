#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Masked-line ablation of the 6283 A diffuse interstellar band / O2 telluric window.

Two of the ten G-class bins with the largest validation-set mean |SHAP| sit at
6283.1 and 6286.2 A, away from every catalogued line and coincident with the
6283.8 A diffuse interstellar band and the O2 telluric region. This script
asks whether the production model depends on that region. The window
6270 to 6295 A (8 bins) is added as a sixth line set, ``DIB_6283``, to the
five deposited sets of ``scripts/revision/ablation_recalibrated.py`` and the
masked-line ablation of :func:`src.interpret.ablation.masked_line_ablation`
is run on the 456 held-out test rows under the two deposited fills:

  gp_interp   PRIMARY: linear interpolation across the masked run
  constant    SENSITIVITY: masked bins set to the median of the TRAIN
              matrix outside the inter-chip gap (the Table 3 / 4 convention)

The five deposited sets are passed in the same call so that the forbidden
mask of the random-window sampler is the deposited one plus the 8 DIB bins;
their rows (in particular Mg b on G: delta -0.257, 59 lost) are written
alongside the DIB rows as a reproduction check. Reference-set settings are
those of the deposited run: class-matched null, bin-matched non-overlapping
random windows, 5000 random windows, 2000 bootstrap resamples, seed 42, gap
bins forbidden.

Output: ``artifacts/revision/dib_6283_ablation.json`` with the window, its
bin count, the rows for every line set and class under both fills, and the
lost-row destinations per class.

Run time: about 11 to 15 min for the two runs on two cores (recorded run:
414 s interpolation, 262 s constant, 878 s wall; ``runs.*.runtime_s``).

Run from the repo root: ``python scripts/revision/dib_6283_ablation.py``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.ablation import _window_mask, masked_line_ablation  # noqa: E402
from src.interpret.classifier import load_model  # noqa: E402
from src.interpret.lines import ALLOWED_MK_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

DIB_SET = "DIB_6283"
DIB_WINDOW: tuple[float, float] = (6270.0, 6295.0)
SEED = 42
N_BOOTSTRAP = 2000
N_RANDOM_CONTROLS = 5000


def load_script_module(relative_path: str, name: str) -> ModuleType:
    """Import a script that is not part of a package by file path."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


abr = load_script_module("scripts/revision/ablation_recalibrated.py", "ablation_recalibrated")
CLASS_NAMES = abr.CLASS_NAMES
row_to_dict = abr.row_to_dict
lost_row_destinations = abr.lost_row_destinations


def dib_line_sets() -> dict[str, list[tuple[float, float]]]:
    """The five deposited line sets followed by the single-window ``DIB_6283`` set."""
    sets = abr.all_line_sets()
    sets[DIB_SET] = [DIB_WINDOW]
    return sets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/dib_6283_ablation.json"))
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--n-random-controls", type=int, default=N_RANDOM_CONTROLS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    wave_centers = fp["wave_centers"]
    gap_mask = fp["gap_mask"].astype(bool)
    train_idx, test_idx = fp["train_idx"], fp["test_idx"]
    X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
    model = load_model(args.model)
    class_labels = list(ALLOWED_MK_CLASSES)  # index 1 -> F, 2 -> G, 3 -> K
    line_sets = dib_line_sets()
    dib_centers = [float(w) for w in wave_centers[_window_mask(wave_centers, [DIB_WINDOW])]]
    n_bins = len(dib_centers)
    fill_train = float(np.nanmedian(X_train[:, ~gap_mask]))
    print(f"{DIB_SET} window {DIB_WINDOW}: {n_bins} bins at {[f'{w:.1f}' for w in dib_centers]} A; "
          f"constant fill = train off-gap median = {fill_train:.6f}", flush=True)

    runs: dict[str, dict[str, Any]] = {
        "production__gp_interp": dict(fill_mode="gp_interp", fill=None),
        "production__constant": dict(fill_mode="constant", fill=fill_train),
    }
    results: dict[str, dict[str, Any]] = {}
    for i, (run_name, cfg) in enumerate(runs.items(), start=1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(runs)}] run {run_name}: fill_mode={cfg['fill_mode']} fill={cfg['fill']}", flush=True)
        draw_stats: dict[str, dict[str, Any]] = {}
        per_row: dict[str, np.ndarray] = {}
        rows = masked_line_ablation(
            model, X_test, y_test, wave_centers, line_sets, class_labels,
            per_class=True,
            n_bootstrap=args.n_bootstrap,
            n_random_controls=args.n_random_controls,
            seed=args.seed,
            gap_mask=gap_mask,
            continuum_fill=cfg["fill"],
            draw_stats_out=draw_stats,
            per_row_out=per_row,
            fill_mode=cfg["fill_mode"],
            null_mode=abr.NULL_MODE,
            match_on=abr.MATCH_ON,
        )
        y_pred_base = per_row["y_pred_base"].astype(np.int64)
        destinations: dict[str, dict[str, int]] = {}
        for set_name in line_sets:
            y_pred_masked = per_row[f"y_pred_masked__{set_name}"].astype(np.int64)
            for c, cname in CLASS_NAMES.items():
                destinations[f"{set_name}__{cname}"] = lost_row_destinations(
                    y_test, y_pred_base, y_pred_masked, c,
                )
        results[run_name] = {
            "continuum": "production",
            "fill_mode": cfg["fill_mode"],
            "continuum_fill": cfg["fill"],
            "null_mode": abr.NULL_MODE,
            "match_on": abr.MATCH_ON,
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_random_controls_requested": int(args.n_random_controls),
            "baseline_accuracy": float(np.mean(y_pred_base == y_test)),
            "draw_stats": draw_stats,
            "p_value_floor_per_line_set": {s: 1.0 / (d["succeeded"] + 1) for s, d in draw_stats.items()},
            "lost_row_destinations": destinations,
            "rows": [row_to_dict(r) for r in rows],
            "runtime_s": float(time.perf_counter() - t0),
        }
        for r in rows:
            if r.line_set in (DIB_SET, "Mg_b"):
                print(f"    {r.line_set:9s} {r.mk_class} delta={r.delta_acc_mean:+.3f} "
                      f"[{r.delta_acc_ci_low:+.3f}, {r.delta_acc_ci_high:+.3f}] lost/gained={r.n_flip_lost}/"
                      f"{r.n_flip_gained} McNemar p={r.mcnemar_exact_p:.4g} p_MC={r.p_value_vs_random:.4g}",
                      flush=True)
        print(f"    done in {results[run_name]['runtime_s']:.0f} s", flush=True)

    payload = {
        "description": (
            f"Masked-line ablation of the fixed production LightGBM classifier on the 456 held-out "
            f"test rows with the {DIB_WINDOW[0]:.0f} to {DIB_WINDOW[1]:.0f} A window (6283.8 A diffuse "
            f"interstellar band and O2 telluric region) added as line set {DIB_SET} to the five "
            f"deposited sets; two fills (gp_interp = linear interpolation, primary; constant = train "
            f"off-gap median, sensitivity). Class-matched null, bin-matched non-overlapping random windows."
        ),
        "inputs": {
            "features": str(args.features),
            "model": str(args.model),
            "n_test": int(len(test_idx)),
            "n_train": int(len(train_idx)),
            "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap),
            "n_random_controls": int(args.n_random_controls),
            "null_mode": abr.NULL_MODE,
            "match_on": abr.MATCH_ON,
            "constant_fill_train_median_production": fill_train,
            "class_label_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
        },
        "window": list(DIB_WINDOW),
        "n_bins": n_bins,
        "bin_centers_aa": dib_centers,
        "line_sets": {k: [list(w) for w in v] for k, v in line_sets.items()},
        "primary_run": "production__gp_interp",
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
