# SPDX-License-Identifier: MIT
"""Unit tests for scripts.ablation_paired (Steps 2a, 2b, 2c).

Covers:
  - paired_bootstrap_delta_difference returns sensible CI on synthetic
    data where the two classes have known different masking effects
  - flip_rate decomposition sums to n_total and matches hand-computed
    counts on a tiny deterministic example
  - tost_equivalence returns EQUIVALENT when delta is small and the
    bootstrap CI sits well inside the indifference zone, and returns
    INEQUIVALENT or INCONCLUSIVE when delta moves outside it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablation_paired import (
    TOST_INDIFFERENCE_ZONE,
    flip_rate,
    paired_bootstrap_delta_difference,
    tost_equivalence,
)


def test_flip_rate_decomposition_sums_to_n_total() -> None:
    # 10 K-class rows, baseline: 8 correct (predict 3), 2 incorrect (predict 1).
    # Under masking: 6 correct (predict 3), 4 incorrect (predict 1).
    # Of the 8 originally correct, 5 stay correct + 3 flip to incorrect.
    # Of the 2 originally incorrect, 1 flips to correct + 1 stays incorrect.
    y_test = np.array([3] * 10 + [1] * 5, dtype=np.int64)  # 10 K + 5 F decoys
    y_pred_base = np.array([3, 3, 3, 3, 3, 3, 3, 3, 1, 1] + [1, 1, 1, 1, 1], dtype=np.int64)
    y_pred_masked = np.array([3, 3, 3, 3, 3, 1, 1, 1, 3, 1] + [1, 1, 1, 1, 1], dtype=np.int64)
    r = flip_rate(y_test, y_pred_base, y_pred_masked, c=3)
    assert r["n_total"] == 10
    assert (
        r["n_correct_to_incorrect"]
        + r["n_incorrect_to_correct"]
        + r["n_unchanged_correct"]
        + r["n_unchanged_incorrect"]
        == r["n_total"]
    )
    assert r["n_correct_to_incorrect"] == 3
    assert r["n_incorrect_to_correct"] == 1
    assert r["n_unchanged_correct"] == 5
    assert r["n_unchanged_incorrect"] == 1
    assert r["n_flipped_total"] == 4
    assert abs(r["flip_rate_total"] - 0.4) < 1e-9


def test_flip_rate_empty_class_returns_nan() -> None:
    y_test = np.array([1, 2, 1, 2], dtype=np.int64)
    y_pred_base = np.array([1, 2, 1, 2], dtype=np.int64)
    y_pred_masked = np.array([1, 2, 1, 2], dtype=np.int64)
    r = flip_rate(y_test, y_pred_base, y_pred_masked, c=99)  # no support
    assert r["n_total"] == 0
    assert np.isnan(r["flip_rate_total"])


def test_paired_bootstrap_recovers_negative_diff_when_class_a_drops_more() -> None:
    """Synthetic: class 2 ("G") loses 30% under masking, class 3 ("K") gains 0%.

    Constructed so delta_G is strongly negative and delta_K is zero. The
    paired bootstrap should return a negative diff with p << 0.05.
    """
    rng = np.random.default_rng(0)
    n_g, n_k = 200, 200
    y_test = np.array([2] * n_g + [3] * n_k, dtype=np.int64)
    # Baseline: 95% correct on both classes.
    y_pred_base = np.array([2] * n_g + [3] * n_k, dtype=np.int64)
    base_wrong_g = rng.choice(n_g, size=int(0.05 * n_g), replace=False)
    base_wrong_k = rng.choice(n_k, size=int(0.05 * n_k), replace=False)
    y_pred_base[base_wrong_g] = 1
    y_pred_base[n_g + base_wrong_k] = 1
    # Masked: G drops 30 percentage points to 65 percent correct, K unchanged.
    y_pred_masked = y_pred_base.copy()
    extra_wrong_g = rng.choice(
        np.setdiff1d(np.arange(n_g), base_wrong_g), size=int(0.30 * n_g), replace=False
    )
    y_pred_masked[extra_wrong_g] = 1

    res = paired_bootstrap_delta_difference(
        y_test, y_pred_base, y_pred_masked,
        class_a=2, class_b=3,
        n_bootstrap=200, seed=42,
    )
    assert res["n_a"] == n_g
    assert res["n_b"] == n_k
    assert res["delta_a_observed"] < -0.2  # G drops by ~30%
    assert abs(res["delta_b_observed"]) < 0.05  # K barely moves
    assert res["diff_observed"] < -0.2  # diff = delta_G - delta_K is strongly negative
    assert res["diff_ci_high_95"] < 0  # CI upper bound below zero
    # One-sided test: H0 is diff >= 0; observed diff is strongly negative so p tiny.
    assert res["p_one_sided_diff_ge_zero"] < 0.05


def test_tost_equivalent_when_delta_small_and_ci_inside() -> None:
    """When masking does nothing, delta=0 and bootstrap CI is tight at zero;
    TOST should reject inequivalence with verdict EQUIVALENT.
    """
    rng = np.random.default_rng(1)
    n = 200
    y_test = np.full(n, 3, dtype=np.int64)
    # 95% correct baseline; masked predictions identical to baseline.
    y_pred_base = np.full(n, 3, dtype=np.int64)
    wrong = rng.choice(n, size=int(0.05 * n), replace=False)
    y_pred_base[wrong] = 1
    y_pred_masked = y_pred_base.copy()
    res = tost_equivalence(
        y_test, y_pred_base, y_pred_masked, c=3,
        indifference_zone=0.025, n_bootstrap=200, seed=42,
    )
    assert res["delta_observed"] == 0.0
    assert res["verdict"] == "EQUIVALENT"
    assert res["ci_inside_zone"] is True
    assert res["p_tost"] < 0.05


def test_tost_inequivalent_when_delta_far_outside_zone() -> None:
    """When masking drops accuracy by 30 percentage points, the CI lies
    entirely below -0.025, neither bound is inside the zone -> INEQUIVALENT.
    """
    rng = np.random.default_rng(2)
    n = 200
    y_test = np.full(n, 3, dtype=np.int64)
    y_pred_base = np.full(n, 3, dtype=np.int64)
    base_wrong = rng.choice(n, size=int(0.05 * n), replace=False)
    y_pred_base[base_wrong] = 1
    y_pred_masked = y_pred_base.copy()
    extra_wrong = rng.choice(
        np.setdiff1d(np.arange(n), base_wrong), size=int(0.30 * n), replace=False
    )
    y_pred_masked[extra_wrong] = 1
    res = tost_equivalence(
        y_test, y_pred_base, y_pred_masked, c=3,
        indifference_zone=0.025, n_bootstrap=200, seed=42,
    )
    assert res["delta_observed"] < -0.2
    assert res["verdict"] == "INEQUIVALENT"
    assert res["ci_inside_zone"] is False
    # p_lower_bound = fraction of bootstrap deltas <= -zone. With delta well
    # below -0.025, essentially every bootstrap resample lands below -zone, so
    # p_lower_bound ~ 1.0 (cannot reject H0_low: effect IS below -zone).
    # p_upper_bound (fraction above +zone) should be at the Phipson-Smyth floor.
    assert res["p_lower_bound"] > 0.95
    assert res["p_upper_bound"] < 0.05
    # max(p_low, p_high) is dominated by p_low ~ 1.0, so equivalence is NOT
    # rejected and p_tost_significant is False.
    assert res["p_tost_significant"] is False


def test_decision_39_indifference_zone_default() -> None:
    """The module-level constant must equal the locked value."""
    assert TOST_INDIFFERENCE_ZONE == 0.025


def test_paired_bootstrap_empty_class_raises() -> None:
    y_test = np.array([1, 1, 1], dtype=np.int64)
    y_pred_base = y_test.copy()
    y_pred_masked = y_test.copy()
    with pytest.raises(ValueError):
        paired_bootstrap_delta_difference(
            y_test, y_pred_base, y_pred_masked,
            class_a=1, class_b=99,
            n_bootstrap=10, seed=0,
        )
