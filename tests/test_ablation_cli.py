"""End-to-end smoke test for ``scripts/ablation.py``.

Builds a tiny synthetic features.npz and a 3-class LightGBM model, then runs
the ablation CLI and asserts the expected output artefacts.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from scripts.ablation import (
    HEADLINE_PAIRS,
    PIVOT_PAIRS,
    evaluate_headline_gate,
    main,
)
from src.interpret.occlusion import AblationRow

EXPECTED_CSV_HEADER = (
    "line_set,mk_class,n_test,baseline_acc,masked_acc_mean,delta_acc_mean,"
    "delta_acc_ci_low,delta_acc_ci_high,p_value_vs_random"
)


def _build_synthetic_features(out: Path, n_rows: int = 60, n_bins: int = 200) -> None:
    rng = np.random.default_rng(123)
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    # LightGBM emits 0..K-1 labels; build_features.py for the production
    # pipeline writes y in {1, 2, 3}. The
    # smoke fixture uses {0, 1, 2} because we are not retraining a labeller,
    # only exercising the CLI plumbing.
    y = np.array([0, 1, 2] * (n_rows // 3), dtype=np.int8)[:n_rows]
    X = rng.normal(loc=1.0, scale=0.02, size=(n_rows, n_bins)).astype(np.float32)
    halpha = (wave_centers >= 6555) & (wave_centers <= 6570)
    mg_b = (wave_centers >= 5165) & (wave_centers <= 5186)
    na_d = (wave_centers >= 5887) & (wave_centers <= 5898)
    X[y == 0][:, halpha] -= 0.4  # class 0 ~ A in ALLOWED_MK_CLASSES
    X[y == 1][:, mg_b] -= 0.3
    X[y == 2][:, na_d] -= 0.5
    gap_mask = (wave_centers >= 5769.0) & (wave_centers <= 5834.0)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    train_idx = indices[: int(n_rows * 0.5)]
    val_idx = indices[int(n_rows * 0.5) : int(n_rows * 0.75)]
    test_idx = indices[int(n_rows * 0.75) :]
    np.savez(
        out,
        X=X,
        y=y,
        wave_centers=wave_centers,
        boundary_distance_k=np.full(n_rows, 500.0, dtype=np.float32),
        dwarf_flag=np.ones(n_rows, dtype=bool),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        groups=np.zeros(n_rows, dtype=np.int64),
        median_imputer=np.ones(n_bins, dtype=np.float32),
        gap_mask=gap_mask,
        ra_deg=np.zeros(n_rows, dtype=np.float64),
        dec_deg=np.zeros(n_rows, dtype=np.float64),
    )


def _build_synthetic_perm(out: Path, n_bins: int = 200) -> None:
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    rng = np.random.default_rng(7)
    importance = rng.random(n_bins).astype(np.float32) * 0.01
    # Plant peaks near canonical MK lines.
    for centre in (4861.33, 5172.68, 5895.92, 6562.80):
        idx = int(np.argmin(np.abs(wave_centers - centre)))
        importance[idx] = 1.0
    np.savez(out, importance_mean=importance, wave_centers=wave_centers)


def _train_tiny_model(features_npz: Path) -> Path:
    import lightgbm as lgb

    payload = np.load(features_npz)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    train_idx = payload["train_idx"]
    val_idx = payload["val_idx"]
    model = lgb.LGBMClassifier(
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.1,
        objective="multiclass",
        num_class=3,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])])
    out_path = features_npz.parent / "tiny_model.pkl"
    with out_path.open("wb") as f:
        pickle.dump(model, f)
    return out_path


def test_evaluate_headline_gate_fgk() -> None:
    """Synthetic rows: two FGK headline pairs PASS at p=0.002 < per-pair
    Bonferroni alpha = 0.01/3 = 0.0033 (Step 9b); the third pair fails."""
    rows: list[AblationRow] = []
    # Two passing headline pairs, one failing.
    for line_set, mk_class, p_val, ci_hi in [
        ("H_balmer", "F", 0.002, -0.01),
        ("Mg_b", "G", 0.002, -0.02),
        ("Mg_b", "K", 0.5, 0.001), # fails
    ]:
        rows.append(AblationRow(
            line_set=line_set,
            mk_class=mk_class,
            n_test=100,
            baseline_acc=0.9,
            masked_acc_mean=0.85,
            delta_acc_mean=ci_hi - 0.005,
            delta_acc_ci_low=ci_hi - 0.02,
            delta_acc_ci_high=ci_hi,
            p_value_vs_random=p_val,
        ))
    # Pivot rows (predicted near zero per ).
    rows.append(AblationRow(
        line_set="Na_D", mk_class="K", n_test=50,
        baseline_acc=0.9, masked_acc_mean=0.9,
        delta_acc_mean=0.0, delta_acc_ci_low=-0.01,
        delta_acc_ci_high=0.01, p_value_vs_random=0.5,
    ))
    rows.append(AblationRow(
        line_set="Ca_I", mk_class="K", n_test=50,
        baseline_acc=0.9, masked_acc_mean=0.9,
        delta_acc_mean=0.0, delta_acc_ci_low=-0.01,
        delta_acc_ci_high=0.01, p_value_vs_random=0.5,
    ))
    gate = evaluate_headline_gate(rows)
    assert gate["gate_status"] == "PASS"
    assert gate["n_pairs_passed"] >= 2
    assert {(p["line_set"], p["mk_class"]) for p in gate["pairs_evaluated"]} == set(
        HEADLINE_PAIRS
    )
    assert {(p["line_set"], p["mk_class"]) for p in gate["pivot_confirmed"]} == set(
        PIVOT_PAIRS
    )


def test_evaluate_headline_gate_fail_when_one_passes() -> None:
    """Only one pair passes -> FAIL."""
    rows = [
        AblationRow("H_balmer", "F", 100, 0.9, 0.85,
                    -0.05, -0.07, -0.03, 0.001),
        AblationRow("Mg_b", "G", 100, 0.9, 0.9,
                    0.0, -0.01, 0.01, 0.5),
        AblationRow("Mg_b", "K", 100, 0.9, 0.9,
                    0.0, -0.01, 0.01, 0.5),
    ]
    gate = evaluate_headline_gate(rows)
    assert gate["gate_status"] == "FAIL"
    assert gate["n_pairs_passed"] == 1


def test_ablation_cli_smoke(tmp_path: Path) -> None:
    """Run the CLI end-to-end on a tiny synthetic fixture."""
    pytest.importorskip("lightgbm")
    features = tmp_path / "features.npz"
    perm = tmp_path / "perm_importance.npz"
    out_dir = tmp_path / "ablation_out"
    _build_synthetic_features(features, n_rows=60, n_bins=200)
    _build_synthetic_perm(perm, n_bins=200)
    model_path = _train_tiny_model(features)

    main([
        "--features", str(features),
        "--model", str(model_path),
        "--out-dir", str(out_dir),
        "--n-bootstrap", "10",
        "--n-random-controls", "12",
        "--perm-importance", str(perm),
        "--seed", "42",
    ])

    full_csv = out_dir / "masked_line_ablation_full.csv"
    line_match_csv = out_dir / "line_match.csv"
    line_match_json = out_dir / "line_match.json"
    gate_eval = out_dir / "gate_eval.json"
    per_row_npz = out_dir / "per_row_predictions.npz"
    assert full_csv.exists()
    assert line_match_csv.exists()
    assert line_match_json.exists()
    assert gate_eval.exists()
    assert per_row_npz.exists(), " CLI must emit per_row_predictions.npz"

    # per_row_predictions.npz has the keys downstream paired bootstrap / flip
    # rate / TOST will read. The fixture features.npz has the same wave grid
    # as LINE_SETS, so all five MK_LINES sets should be present.
    pr = np.load(per_row_npz)
    assert "y_test" in pr.files
    assert "y_pred_base" in pr.files
    assert "wave_centers" in pr.files
    assert "test_idx" in pr.files
    assert "continuum_fill" in pr.files
    assert "seed" in pr.files
    # At least one line set should have produced a masked-prediction column.
    masked_keys = [k for k in pr.files if k.startswith("y_pred_masked__")]
    line_mask_keys = [k for k in pr.files if k.startswith("line_mask__")]
    assert masked_keys, "no masked-prediction columns emitted"
    assert len(masked_keys) == len(line_mask_keys)
    n_test = len(pr["test_idx"])
    assert pr["y_test"].shape == (n_test,)
    assert pr["y_pred_base"].shape == (n_test,)
    for k in masked_keys:
        assert pr[k].shape == (n_test,), f"{k} shape mismatch"

    # Header check on the ablation CSV.
    header = full_csv.read_text().splitlines()[0]
    assert header == EXPECTED_CSV_HEADER

    # Header dataclass field count matches.
    assert len(fields(AblationRow)) == 9
    assert len(EXPECTED_CSV_HEADER.split(",")) == 9

    # gate_eval.json contains the keys evaluate_headline_gate guarantees plus
    # the CLI-side enrichments.
    with gate_eval.open() as f:
        gate = json.load(f)
    for key in (
        "gate_status",
        "n_pairs_passed",
        "pairs_evaluated",
        "pivot_confirmed",
        "draw_success_rates",
        "continuum_fill_sensitivity",
        "line_match",
        "headline_pairs",
        "pivot_pairs",
        "seed",
        "n_bootstrap",
        "n_random_controls",
    ):
        assert key in gate, f"gate_eval.json missing key {key!r}"
    assert gate["seed"] == 42
    assert gate["n_random_controls"] == 12
    assert gate["continuum_fill_sensitivity"]["continuum_fill_locked"] == 1.0
