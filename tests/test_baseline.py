# SPDX-License-Identifier: MIT
"""Unit tests for src.interpret.baseline ()."""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.baseline import _correlation_distance, knn_predict, macro_f1


def test_correlation_distance_zero_on_identical_rows() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 50)).astype(np.float32)
    d = _correlation_distance(A, A)
    # Distance from a row to itself is 1 - 1 = 0.
    diag = np.diag(d)
    np.testing.assert_allclose(diag, 0.0, atol=1e-5)


def test_correlation_distance_two_on_anticorrelated_rows() -> None:
    """Two rows with -1 correlation should have distance 1 - (-1) = 2."""
    a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    b = -a + 6.0  # exact negative of a
    d = _correlation_distance(a, b)
    np.testing.assert_allclose(d[0, 0], 2.0, atol=1e-5)


def test_knn_predict_recovers_perfect_separation() -> None:
    """Two classes with disjoint feature support; kNN should be perfect."""
    rng = np.random.default_rng(0)
    X_train = np.concatenate([
        rng.normal(0, 1, size=(20, 10)) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
        rng.normal(0, 1, size=(20, 10)) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 5]),
    ])
    y_train = np.array([1] * 20 + [2] * 20)
    X_test = np.concatenate([
        rng.normal(0, 1, size=(10, 10)) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
        rng.normal(0, 1, size=(10, 10)) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 5]),
    ])
    y_test = np.array([1] * 10 + [2] * 10)
    preds = knn_predict(X_train, y_train, X_test, k_values=(1, 5))
    # Both k values should be at least 90 percent accurate on this easy case.
    for k, p in preds.items():
        acc = float(np.mean(p == y_test))
        assert acc >= 0.9, f"k={k} got accuracy {acc}"


def test_macro_f1_perfect_and_chance() -> None:
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_perfect = y_true.copy()
    assert macro_f1(y_true, y_perfect, [1, 2, 3]) == 1.0
    # All predictions = class 1 -> F1 for class 1 is 2*1*0.33/(1+0.33) = 0.5,
    # F1 for 2 and 3 is 0, macro mean ~ 0.167.
    y_one = np.ones_like(y_true)
    assert abs(macro_f1(y_true, y_one, [1, 2, 3]) - 0.5 / 3) < 1e-9


def test_macro_f1_handles_missing_class_in_predictions() -> None:
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 1])  # never predicts 3
    f1 = macro_f1(y_true, y_pred, [1, 2, 3])
    # Class 1 F1: tp=1, fp=1, fn=0, prec=0.5, rec=1.0, F1 = 2/3
    # Class 2 F1: tp=1, fp=0, fn=0, F1 = 1
    # Class 3 F1: tp=0, fp=0, fn=1, F1 = 0
    expected = (2 / 3 + 1.0 + 0.0) / 3
    assert abs(f1 - expected) < 1e-9
