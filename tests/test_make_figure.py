"""Tests for scripts/make_figure.py.

Verifies that the production make_figure script propagates the ``gap_mask``
key from features.npz into the main-figure rendering pipeline.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

make_figure: Any = importlib.import_module("make_figure")


def _write_synthetic_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Write minimal artifacts at tmp_path; return path dict for CLI args."""
    rng = np.random.default_rng(42)
    n_bins = 256
    n_test = 64
    n_classes = 3

    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    gap_mask = (wave_centers >= 5769.0) & (wave_centers <= 5834.0)
    X = (1.0 + 0.05 * rng.standard_normal((n_test, n_bins))).astype(np.float32)
    y = rng.integers(low=1, high=4, size=n_test).astype(np.int64)  # F=1,G=2,K=3
    test_idx = np.arange(n_test, dtype=np.int64)

    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        X=X,
        y=y,
        wave_centers=wave_centers,
        gap_mask=gap_mask,
        train_idx=test_idx,
        val_idx=test_idx,
        test_idx=test_idx,
        boundary_distance_k=np.zeros(n_test, dtype=np.float32),
        dwarf_flag=np.ones(n_test, dtype=bool),
        groups=np.zeros(n_test, dtype=np.int64),
        median_imputer=np.ones(n_bins, dtype=np.float32),
        ra_deg=np.zeros(n_test, dtype=np.float32),
        dec_deg=np.zeros(n_test, dtype=np.float32),
    )

    importance_path = tmp_path / "perm_importance.npz"
    np.savez_compressed(
        importance_path,
        importance_mean=rng.random(n_bins).astype(np.float32),
        wave_centers=wave_centers,
    )

    shap_path = tmp_path / "shap_values.npz"
    np.savez_compressed(
        shap_path,
        shap_values=rng.random((n_classes, n_test, n_bins)).astype(np.float32),
        mean_abs_per_class=rng.random((n_classes, n_bins)).astype(np.float32),
        wave_centers=wave_centers,
        sample_idx=test_idx,
        y_sample=y,
    )

    gate_eval = {
        "headline_pairs": [
            {"line_set": "H_balmer", "mk_class": "F"},
            {"line_set": "Mg_b", "mk_class": "G"},
            {"line_set": "Mg_b", "mk_class": "K"},
        ],
        "pivot_pairs": [
            {"line_set": "Na_D", "mk_class": "K"},
            {"line_set": "Ca_I", "mk_class": "K"},
        ],
        "pairs_evaluated": [
            {"line_set": "H_balmer", "mk_class": "F",
             "delta_acc_mean": -0.84, "delta_acc_ci_low": -0.91,
             "delta_acc_ci_high": -0.77, "p_value_vs_random": 0.0},
            {"line_set": "Mg_b", "mk_class": "G",
             "delta_acc_mean": -0.21, "delta_acc_ci_low": -0.26,
             "delta_acc_ci_high": -0.16, "p_value_vs_random": 0.0},
            {"line_set": "Mg_b", "mk_class": "K",
             "delta_acc_mean": 0.0, "delta_acc_ci_low": -0.02,
             "delta_acc_ci_high": 0.02, "p_value_vs_random": 0.866},
        ],
        "pivot_confirmed": [
            {"line_set": "Na_D", "mk_class": "K",
             "delta_acc_mean": -0.007, "delta_acc_ci_low": -0.022,
             "delta_acc_ci_high": 0.0, "p_value_vs_random": 0.006},
            {"line_set": "Ca_I", "mk_class": "K",
             "delta_acc_mean": 0.0, "delta_acc_ci_low": 0.0,
             "delta_acc_ci_high": 0.0, "p_value_vs_random": 0.878},
        ],
    }
    ablation_path = tmp_path / "gate_eval.json"
    ablation_path.write_text(json.dumps(gate_eval))

    metrics = {
        "confusion_matrix": [[68, 13, 0], [3, 222, 5], [1, 8, 136]],
        "class_labels": ["F", "G", "K"],
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    benchmark = {
        "labels": ["F", "G", "K", "OTHER"],
        "confusion_matrix": [
            [10, 5, 0, 0],
            [3, 30, 2, 0],
            [0, 4, 25, 0],
            [0, 0, 0, 0],
        ],
        "agreement_rate": 0.65,
        "macro_f1_fgk": 0.66,
        "n_compared": 79,
    }
    benchmark_path = tmp_path / "benchmark_report.json"
    benchmark_path.write_text(json.dumps(benchmark))

    return {
        "features": features_path,
        "importance": importance_path,
        "shap": shap_path,
        "ablation": ablation_path,
        "metrics": metrics_path,
        "benchmark": benchmark_path,
    }


def test_main_figure_consumes_gap_mask_from_features_npz(tmp_path: Path) -> None:
    """Run make_figure end-to-end on synthetic artifacts; assert that the
    plot_main_figure call received the gap_mask array sourced from features.npz
    (not a default zero mask)."""
    paths = _write_synthetic_artifacts(tmp_path)
    out_dir = tmp_path / "figures"

    captured: dict[str, Any] = {}

    def _spy_plot_main_figure(**kwargs: Any) -> None:
        captured["gap_mask"] = np.asarray(kwargs["gap_mask"]).copy()
        captured["wave_centers"] = np.asarray(kwargs["wave_centers"]).copy()
        captured["out_path"] = kwargs["out_path"]
        # Still emit a tiny valid PDF so downstream assertions on file
        # existence in the script do not fire (none currently, but defensive).
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"%PDF-1.4\n%fake\n%%EOF\n")

    with patch.object(make_figure, "plot_main_figure", side_effect=_spy_plot_main_figure):
        argv = [
            "--features", str(paths["features"]),
            "--importance", str(paths["importance"]),
            "--shap", str(paths["shap"]),
            "--ablation", str(paths["ablation"]),
            "--metrics", str(paths["metrics"]),
            "--benchmark", str(paths["benchmark"]),
            "--out-dir", str(out_dir),
        ]
        make_figure.main(argv)

    assert "gap_mask" in captured, "plot_main_figure was not invoked"
    gap_mask = captured["gap_mask"]
    wave_centers = captured["wave_centers"]
    assert gap_mask.shape == wave_centers.shape
    assert gap_mask.dtype == bool
    # expectation: True bins land in [5769, 5834] A.
    assert gap_mask.sum() > 0
    assert wave_centers[gap_mask].min() >= 5769.0
    assert wave_centers[gap_mask].max() <= 5834.0
    # Output PDF path under the requested out_dir.
    assert Path(captured["out_path"]).parent == out_dir
    assert Path(captured["out_path"]).name == "importance_main.pdf"
