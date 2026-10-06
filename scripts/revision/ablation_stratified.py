#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Stratify the recalibrated masked-line ablation by luminosity class and Teff boundary distance.

Consumes the per-row payload written by ``ablation_recalibrated.py``
(``artifacts/revision/ablation_recalibrated_per_row.npz``) and the GES label
table (``artifacts/ges_mk_labels.parquet``, joined onto the feature rows by
exact (ra_deg, dec_deg) match). For every (line set, class) pair the delta
recall, flip counts and mean change in the true-class probability are
reported overall and split by

  (a) luminosity class: dwarf (log g > 3.5) vs giant (log g <= 3.5);
  (b) distance to the nearest Teff class boundary (5300 K and 6000 K):
      within 150 K vs beyond 150 K.

Every row that flips from correct to incorrect is listed with its destination
class, Teff, log g, luminosity class and boundary distance. The dwarf/giant
split is reported for the primary run (production continuum, linear
interpolation fill) and for the constant-fill sensitivity run so the
dependence of the split on the fill convention can be stated.

Output: artifacts/revision/ablation_stratified.json
Run from the repo root: ``python scripts/revision/ablation_stratified.py``
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

CLASS_NAMES: dict[int, str] = {1: "F", 2: "G", 3: "K"}
TEFF_BOUNDARIES_K: tuple[float, ...] = (5300.0, 6000.0)
BOUNDARY_CUT_K: float = 150.0
LOGG_DWARF_CUT: float = 3.5
RUNS: tuple[str, ...] = ("production__gp_interp", "production__constant")
STRATA: dict[str, tuple[str, str]] = {
    "luminosity": ("dwarf", "giant"),
    "boundary": ("within_150K", "beyond_150K"),
}


def _bootstrap_ci(
    rng: np.random.Generator, base_ok: np.ndarray, mask_ok: np.ndarray, n_boot: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the recall delta over the given rows."""
    n = len(base_ok)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    d = mask_ok[idx].mean(axis=1) - base_ok[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _summarise(
    rows: np.ndarray, cls: int, y: np.ndarray, pred_b: np.ndarray, pred_m: np.ndarray,
    dp_true: np.ndarray, rng: np.random.Generator, n_boot: int,
) -> dict[str, Any]:
    """Recall delta, flips and probability change for the rows of ``cls`` selected by ``rows``."""
    sel = rows & (y == cls)
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    base_ok = (pred_b[sel] == cls)
    mask_ok = (pred_m[sel] == cls)
    lo, hi = _bootstrap_ci(rng, base_ok.astype(float), mask_ok.astype(float), n_boot)
    return {
        "n": n,
        "baseline_recall": float(base_ok.mean()),
        "masked_recall": float(mask_ok.mean()),
        "delta_recall": float(mask_ok.mean() - base_ok.mean()),
        "delta_ci_low": lo,
        "delta_ci_high": hi,
        "n_flip_lost": int(np.sum(base_ok & ~mask_ok)),
        "n_flip_gained": int(np.sum(~base_ok & mask_ok)),
        "mean_delta_true_prob": float(np.mean(dp_true[sel])),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--labels", type=Path, default=Path("artifacts/ges_mk_labels.parquet"))
    p.add_argument("--per-row", type=Path, default=Path("artifacts/revision/ablation_recalibrated_per_row.npz"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/ablation_stratified.json"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    fp = np.load(args.features, allow_pickle=False)
    test_idx = fp["test_idx"]
    feat = pd.DataFrame({
        "ra_deg": fp["ra_deg"], "dec_deg": fp["dec_deg"], "y": fp["y"].astype(int),
        "boundary_distance_k_features": fp["boundary_distance_k"],
        "dwarf_flag_features": fp["dwarf_flag"],
    })
    labels = pd.read_parquet(args.labels)
    joined = feat.merge(labels, on=["ra_deg", "dec_deg"], how="left", validate="1:1")
    if joined["mk_int"].isna().any():
        raise ValueError("label join left unmatched feature rows")
    if not (joined["mk_int"].astype(int) == joined["y"]).all():
        raise ValueError("label join: mk_int disagrees with features y")
    meta = joined.iloc[test_idx].reset_index(drop=True)

    teff = meta["teff_k"].to_numpy(dtype=float)
    logg = meta["logg"].to_numpy(dtype=float)
    # Deposited definition: distance to the nearest edge of the row's own MK
    # Teff bin (F [6000, 7300), G [5300, 6000), K [3900, 5300)), so K rows
    # can be close to the 3900 K edge. The interior variant uses only the
    # two F/G and G/K boundaries shared by the three classes.
    boundary = meta["boundary_distance_k"].to_numpy(dtype=float)
    boundary_feat = meta["boundary_distance_k_features"].to_numpy(dtype=float)
    boundary_interior = np.min(
        np.abs(teff[:, None] - np.asarray(TEFF_BOUNDARIES_K)[None, :]), axis=1,
    )
    dwarf = logg > LOGG_DWARF_CUT
    dwarf_feat = meta["dwarf_flag_features"].to_numpy().astype(bool)
    near = boundary <= BOUNDARY_CUT_K
    near_interior = boundary_interior <= BOUNDARY_CUT_K
    y = meta["y"].to_numpy(dtype=int)

    pr = np.load(args.per_row, allow_pickle=False)
    if not np.array_equal(pr["test_idx"], test_idx):
        raise ValueError("per-row payload test_idx does not match features.npz test_idx")
    rng = np.random.default_rng(args.seed)

    def run_keys(run: str) -> list[str]:
        prefix = f"{run}__y_pred_masked__"
        return [k[len(prefix):] for k in pr.files if k.startswith(prefix)]

    strata_masks = {
        "luminosity": {"dwarf": dwarf, "giant": ~dwarf},
        "boundary": {"within_150K": near, "beyond_150K": ~near},
        "boundary_interior": {"within_150K": near_interior, "beyond_150K": ~near_interior},
    }
    all_rows = np.ones(len(y), dtype=bool)

    results: dict[str, Any] = {}
    for run in RUNS:
        if f"{run}__y_test" not in pr.files:
            logger.warning("run %s absent from per-row payload; skipping", run)
            continue
        y_run = pr[f"{run}__y_test"].astype(int)
        if not np.array_equal(y_run, y):
            raise ValueError(f"{run}: y_test in payload disagrees with labels")
        pred_b = pr[f"{run}__y_pred_base"].astype(int)
        proba_classes = pr[f"{run}__proba_classes"].astype(int).tolist()
        col = np.array([proba_classes.index(c) for c in y])
        pt_base = pr[f"{run}__proba_base"][np.arange(len(y)), col].astype(float)
        run_out: dict[str, Any] = {
            "fill_mode": str(pr[f"{run}__fill_mode"]),
            "continuum_fill": float(pr[f"{run}__continuum_fill"]),
            "pairs": {},
        }
        for set_name in run_keys(run):
            pred_m = pr[f"{run}__y_pred_masked__{set_name}"].astype(int)
            pt_mask = pr[f"{run}__proba_masked__{set_name}"][np.arange(len(y)), col].astype(float)
            dp_true = pt_mask - pt_base
            for cls, cname in CLASS_NAMES.items():
                pair: dict[str, Any] = {
                    "overall": _summarise(all_rows, cls, y, pred_b, pred_m, dp_true, rng, args.n_bootstrap),
                }
                for stratum, parts in strata_masks.items():
                    pair[stratum] = {
                        part: _summarise(m, cls, y, pred_b, pred_m, dp_true, rng, args.n_bootstrap)
                        for part, m in parts.items()
                    }
                lost = np.flatnonzero((y == cls) & (pred_b == cls) & (pred_m != cls))
                pair["lost_rows"] = [
                    {
                        "test_row": int(i),
                        "feature_index": int(test_idx[i]),
                        "destination_class": CLASS_NAMES[int(pred_m[i])],
                        "teff_k": float(teff[i]),
                        "logg": float(logg[i]),
                        "luminosity": "dwarf" if dwarf[i] else "giant",
                        "boundary_distance_k": float(boundary[i]),
                        "within_150K": bool(near[i]),
                        "boundary_distance_interior_k": float(boundary_interior[i]),
                        "within_150K_interior": bool(near_interior[i]),
                        "delta_true_prob": float(dp_true[i]),
                    }
                    for i in lost
                ]
                dest = pred_m[lost]
                pair["lost_destination_counts"] = {
                    CLASS_NAMES[c]: int(np.sum(dest == c)) for c in CLASS_NAMES if c != cls
                }
                pair["lost_median_teff_k"] = float(np.median(teff[lost])) if len(lost) else None
                pair["lost_n_dwarf"] = int(np.sum(dwarf[lost]))
                pair["lost_n_giant"] = int(np.sum(~dwarf[lost]))
                pair["lost_n_within_150K"] = int(np.sum(near[lost]))
                run_out["pairs"][f"{set_name}__{cname}"] = pair
        results[run] = run_out

    class_geometry = {}
    for cls, cname in CLASS_NAMES.items():
        sel = y == cls
        class_geometry[cname] = {
            "n": int(sel.sum()),
            "n_dwarf": int(np.sum(dwarf & sel)),
            "n_giant": int(np.sum(~dwarf & sel)),
            "median_boundary_distance_k": float(np.median(boundary[sel])),
            "n_within_150K": int(np.sum(near & sel)),
            "n_beyond_150K": int(np.sum(~near & sel)),
            "median_boundary_distance_interior_k": float(np.median(boundary_interior[sel])),
            "n_within_150K_interior": int(np.sum(near_interior & sel)),
            "n_beyond_150K_interior": int(np.sum(~near_interior & sel)),
            "median_teff_k": float(np.median(teff[sel])),
            "median_logg": float(np.median(logg[sel])),
        }

    mg_g = results.get(RUNS[0], {}).get("pairs", {}).get("Mg_b__G", {})
    payload = {
        "description": (
            "Luminosity-class and Teff-boundary stratification of the recalibrated masked-line "
            "ablation on the 456 held-out test rows; production model fixed."
        ),
        "inputs": {
            "features": str(args.features),
            "labels": str(args.labels),
            "per_row_payload": str(args.per_row),
            "n_test": int(len(y)),
            "runs": list(results.keys()),
            "primary_run": RUNS[0],
            "boundary_definition": (
                "stratum 'boundary': deposited boundary_distance_k = distance to the nearest "
                "edge of the row's own MK Teff bin (F [6000, 7300), G [5300, 6000), K [3900, 5300) K); "
                "stratum 'boundary_interior': distance to the nearest of the interior boundaries only"
            ),
            "teff_boundaries_interior_k": list(TEFF_BOUNDARIES_K),
            "boundary_cut_k": BOUNDARY_CUT_K,
            "logg_dwarf_cut": LOGG_DWARF_CUT,
            "n_bootstrap_strata_ci": int(args.n_bootstrap),
            "seed": int(args.seed),
            "join": "exact (ra_deg, dec_deg) match, 1:1, mk_int == y verified",
        },
        "consistency_checks": {
            "dwarf_flag_recomputed_matches_features": bool(np.array_equal(dwarf, dwarf_feat)),
            "n_dwarf_flag_mismatch": int(np.sum(dwarf != dwarf_feat)),
            "boundary_distance_labels_vs_features_max_abs_diff_k": float(np.max(np.abs(boundary - boundary_feat))),
            "boundary_distance_interior_vs_deposited_max_abs_diff_k": float(np.max(np.abs(boundary_interior - boundary))),
        },
        "test_set_geometry": {
            "n_dwarf": int(dwarf.sum()),
            "n_giant": int((~dwarf).sum()),
            "n_within_150K": int(near.sum()),
            "n_beyond_150K": int((~near).sum()),
            "n_within_150K_interior": int(near_interior.sum()),
            "n_beyond_150K_interior": int((~near_interior).sum()),
            "per_class": class_geometry,
        },
        "focus": {
            "Mg_b__G_lost_rows_primary": {
                "n_lost": len(mg_g.get("lost_rows", [])),
                "median_teff_k": mg_g.get("lost_median_teff_k"),
                "destination_counts": mg_g.get("lost_destination_counts"),
                "n_dwarf": mg_g.get("lost_n_dwarf"),
                "n_giant": mg_g.get("lost_n_giant"),
                "n_within_150K": mg_g.get("lost_n_within_150K"),
            },
            "K_class": {
                "median_boundary_distance_k": class_geometry["K"]["median_boundary_distance_k"],
                "n_within_150K": class_geometry["K"]["n_within_150K"],
                "median_boundary_distance_interior_k": class_geometry["K"]["median_boundary_distance_interior_k"],
                "n_within_150K_interior": class_geometry["K"]["n_within_150K_interior"],
                "n": class_geometry["K"]["n"],
            },
        },
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", args.out)
    print(f"K class: median boundary distance {class_geometry['K']['median_boundary_distance_k']:.0f} K, "
          f"{class_geometry['K']['n_within_150K']} of {class_geometry['K']['n']} within {BOUNDARY_CUT_K:.0f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
