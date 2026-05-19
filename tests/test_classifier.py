"""Unit tests for src.interpret.classifier ()."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.interpret.classifier import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_HPARAMS,
    ClassifierMetrics,
    boundary_filtered_accuracy,
    derive_spatial_groups,
    evaluate,
    load_model,
    metrics_to_dict,
    save_model,
    singleton_group_stats,
    train,
    train_cv,
    write_metrics,
)


# expected hyperparameters.
EXPECTED_HPARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "max_depth": 8,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _make_synthetic(
    n_per_class: int = 60,
    n_features: int = 12,
    n_groups: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a 3-class synthetic dataset with class-correlated means and a
    populated ``groups`` array that supports group-aware CV."""
    rng = np.random.default_rng(seed)
    classes = np.array([1, 2, 3], dtype=np.int64)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for c in classes:
        mean = (c - 2) * np.linspace(-1.0, 1.0, n_features)
        Xs.append(rng.normal(loc=mean, scale=0.5, size=(n_per_class, n_features)))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    # Assign each row a group; groups are independent of class so stratified
    # group splitting is well-defined.
    groups = rng.integers(0, n_groups, size=len(y))
    perm = rng.permutation(len(y))
    return X[perm], y[perm], groups[perm]


def test_default_hparams_locked() -> None:
    """ hyperparameters are frozen for reproducibility."""
    assert dict(DEFAULT_HPARAMS) == EXPECTED_HPARAMS
    assert DEFAULT_EARLY_STOPPING == 50


def test_train_runs_on_synthetic() -> None:
    X, y, _ = _make_synthetic(n_per_class=80, n_features=8, seed=1)
    n = len(y)
    cut = int(n * 0.7)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    model = train(X_tr, y_tr, X_va, y_va, num_class=3)
    preds = model.predict(X_va)
    assert preds.shape == y_va.shape
    assert set(np.unique(preds).tolist()).issubset({1, 2, 3})


def test_evaluate_schema() -> None:
    X, y, _ = _make_synthetic(n_per_class=60, n_features=8, seed=2)
    n = len(y)
    cut = int(n * 0.7)
    model = train(X[:cut], y[:cut], X[cut:], y[cut:], num_class=3)
    labels = ["F", "G", "K"]
    m = evaluate(model, X[cut:], y[cut:], class_labels=labels,
                 n_train=cut, n_val=n - cut)
    assert isinstance(m, ClassifierMetrics)
    assert m.class_labels == labels
    assert set(m.per_class_recall.keys()) == set(labels)
    assert set(m.per_class_precision.keys()) == set(labels)
    assert set(m.per_class_f1.keys()) == set(labels)
    cm = np.array(m.confusion_matrix)
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == n - cut
    assert m.n_test == n - cut


def test_train_cv_runs_3fold() -> None:
    X, y, g = _make_synthetic(n_per_class=60, n_features=8, n_groups=12, seed=3)
    out = train_cv(
        X=X, y=y, groups=g,
        num_class=3, n_splits=3, random_state=42,
        class_labels=["F", "G", "K"],
    )
    assert out["n_splits"] == 3
    assert len(out["per_fold"]) == 3
    assert not out["cv_fallback_triggered"]
    assert out["splitter"] == "StratifiedGroupKFold"
    assert np.isfinite(out["cv_macro_f1_mean"])
    assert out["cv_macro_f1_std"] >= 0.0
    for f in out["per_fold"]:
        assert {"fold", "n_train", "n_val", "macro_f1", "accuracy",
                "per_class_recall", "confusion_matrix",
                "runtime_s", "best_iteration"} <= set(f.keys())


def test_train_cv_respects_groups() -> None:
    """No group should appear in both train and val of any fold."""
    X, y, g = _make_synthetic(n_per_class=60, n_features=8, n_groups=12, seed=4)
    # Replicate the splitter the function uses to inspect membership.
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for tr, va in splitter.split(X, y, groups=g):
        tr_groups = set(np.unique(g[tr]).tolist())
        va_groups = set(np.unique(g[va]).tolist())
        assert not (tr_groups & va_groups)
    # And confirm train_cv runs without raising the internal assertion.
    out = train_cv(X=X, y=y, groups=g, num_class=3,
                   n_splits=3, random_state=42, class_labels=["F", "G", "K"])
    assert not out["cv_fallback_triggered"]


def test_train_cv_fallback_on_timeout() -> None:
    """Setting per_fold_timeout_s to ~0 trips the timeout fallback after
    the first fold completes; cv_fallback_triggered must be True and the
    reason string must mention the per-fold budget."""
    X, y, g = _make_synthetic(n_per_class=40, n_features=6, n_groups=10, seed=5)
    out = train_cv(
        X=X, y=y, groups=g,
        num_class=3, n_splits=3, random_state=42,
        per_fold_timeout_s=1e-6,
        class_labels=["F", "G", "K"],
    )
    assert out["cv_fallback_triggered"] is True
    assert "per-fold" in out["cv_fallback_reason"]


def test_train_cv_degrades_when_groups_collapse() -> None:
    """/ features.npz reality: when groups carry a single value,
    StratifiedGroupKFold cannot split; degrade to StratifiedKFold without
    re-deriving groups."""
    X, y, _ = _make_synthetic(n_per_class=60, n_features=8, seed=6)
    g_collapsed = np.zeros(len(y), dtype=np.int64)
    out = train_cv(
        X=X, y=y, groups=g_collapsed,
        num_class=3, n_splits=3, random_state=42,
        class_labels=["F", "G", "K"],
    )
    assert out["splitter"] == "StratifiedKFold"
    assert "n_unique_groups" in out["cv_fallback_reason"]
    assert len(out["per_fold"]) == 3
    assert np.isfinite(out["cv_macro_f1_mean"])


def test_boundary_filtered_accuracy() -> None:
    X, y, _ = _make_synthetic(n_per_class=80, n_features=8, seed=7)
    n = len(y)
    cut = int(n * 0.7)
    model = train(X[:cut], y[:cut], X[cut:], y[cut:], num_class=3)
    rng = np.random.default_rng(7)
    bd = rng.uniform(-500.0, 500.0, size=n - cut).astype(np.float32)
    out = boundary_filtered_accuracy(
        model=model,
        X_test=X[cut:], y_test=y[cut:],
        boundary_distance_k=bd, threshold_k=200.0,
    )
    expected_keys = {
        "full_test_acc", "boundary_filtered_acc", "delta_acc",
        "delta_acc_ci_low", "delta_acc_ci_high",
        "threshold_k", "n_filtered", "n_test",
    }
    assert expected_keys <= set(out.keys())
    assert 0.0 <= out["full_test_acc"] <= 1.0
    if out["n_filtered"] > 0:
        assert 0.0 <= out["boundary_filtered_acc"] <= 1.0
        assert out["delta_acc_ci_low"] <= out["delta_acc_ci_high"]
    assert out["threshold_k"] == 200.0
    assert out["n_test"] == float(n - cut)


def test_metrics_to_dict_flat_json(tmp_path: Path) -> None:
    """metrics_to_dict produces a flat dict that round-trips through json."""
    metrics = ClassifierMetrics(
        accuracy=0.81,
        macro_f1=0.79,
        per_class_precision={"F": 0.8, "G": 0.85, "K": 0.78},
        per_class_recall={"F": 0.7, "G": 0.9, "K": 0.75},
        per_class_f1={"F": 0.75, "G": 0.875, "K": 0.765},
        confusion_matrix=[[10, 2, 0], [1, 18, 1], [0, 3, 12]],
        class_labels=["F", "G", "K"],
        n_train=100, n_val=30, n_test=47,
    )
    cv_results = {
        "cv_macro_f1_mean": 0.78,
        "cv_macro_f1_std": 0.02,
        "per_fold": [
            {"fold": 0, "n_train": 80, "n_val": 20, "macro_f1": 0.77,
             "accuracy": 0.8, "per_class_recall": {"F": 0.7, "G": 0.9, "K": 0.7},
             "confusion_matrix": [[5, 1, 0], [0, 9, 1], [0, 1, 3]],
             "runtime_s": 1.5, "best_iteration": 137},
        ],
        "cv_fallback_triggered": False,
        "cv_fallback_reason": "",
        "splitter": "StratifiedGroupKFold",
        "n_splits": 3,
        "total_runtime_s": 4.5,
    }
    boundary = {
        "full_test_acc": 0.81, "boundary_filtered_acc": 0.86,
        "delta_acc": 0.05, "delta_acc_ci_low": 0.01, "delta_acc_ci_high": 0.09,
        "threshold_k": 200.0, "n_filtered": 35.0, "n_test": 47.0,
    }
    payload = metrics_to_dict(metrics, cv_results, boundary)

    expected_top_level = {
        "accuracy", "macro_f1", "per_class_precision", "per_class_recall",
        "per_class_f1", "confusion_matrix", "class_labels",
        "n_train", "n_val", "n_test",
        "cv_macro_f1_mean", "cv_macro_f1_std", "cv_fallback_triggered",
        "cv_fallback_reason", "cv_splitter", "cv_n_splits",
        "cv_total_runtime_s", "cv_per_fold",
        "boundary_filtered_accuracy",
    }
    assert expected_top_level <= set(payload.keys())
    assert payload["cv_macro_f1_mean"] == pytest.approx(0.78)
    assert payload["cv_splitter"] == "StratifiedGroupKFold"
    assert payload["cv_per_fold"][0]["fold"] == 0
    assert payload["boundary_filtered_accuracy"]["delta_acc"] == pytest.approx(0.05)

    out_path = tmp_path / "metrics.json"
    write_metrics(payload, out_path)
    reloaded = json.loads(out_path.read_text())
    assert reloaded["macro_f1"] == pytest.approx(0.79)
    assert reloaded["cv_per_fold"][0]["best_iteration"] == 137


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    X, y, _ = _make_synthetic(n_per_class=40, n_features=6, seed=8)
    n = len(y)
    cut = int(n * 0.7)
    model = train(X[:cut], y[:cut], X[cut:], y[cut:], num_class=3)
    p = tmp_path / "model.pkl"
    save_model(model, p)
    assert p.exists()
    loaded = load_model(p)
    np.testing.assert_array_equal(model.predict(X[cut:]), loaded.predict(X[cut:]))


def _two_well_separated_clusters(
    n_per_cluster: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixture with two on-sky clusters separated by ~30 degrees.

    Cluster A is centered at (RA, Dec) = (10, -30); cluster B at (200, 40).
    Each cluster has ``n_per_cluster`` members within 0.05 degrees of its
    center (well inside the 0.1 deg DBSCAN epsilon used by ).
    """
    rng = np.random.default_rng(seed)
    ra_a = 10.0 + 0.02 * rng.standard_normal(n_per_cluster)
    dec_a = -30.0 + 0.02 * rng.standard_normal(n_per_cluster)
    ra_b = 200.0 + 0.02 * rng.standard_normal(n_per_cluster)
    dec_b = 40.0 + 0.02 * rng.standard_normal(n_per_cluster)
    ra = np.concatenate([ra_a, ra_b])
    dec = np.concatenate([dec_a, dec_b])
    # ensure ra is in [0, 360)
    ra = np.mod(ra, 360.0)
    return ra, dec


def test_derive_spatial_groups_deterministic() -> None:
    """ same input must produce the same output across calls,
    and a fixture with two well-separated clusters must yield more than one
    unique group ID."""
    ra, dec = _two_well_separated_clusters(n_per_cluster=30, seed=42)
    g1 = derive_spatial_groups(ra, dec, eps_deg=0.1, min_samples=5)
    g2 = derive_spatial_groups(ra, dec, eps_deg=0.1, min_samples=5)
    np.testing.assert_array_equal(g1, g2)
    assert g1.dtype == np.int64
    assert g1.shape == (60,)
    n_unique = int(len(np.unique(g1)))
    assert n_unique > 1, (
        f"two separated clusters must produce >1 group, got {n_unique}"
    )
    # The two well-separated clusters should each form a single DBSCAN
    # cluster (label >= 0); the helper preserves cluster IDs and assigns
    # unique negative IDs to noise points only.
    cluster_labels = g1[g1 >= 0]
    assert int(len(np.unique(cluster_labels))) >= 1


def test_derive_spatial_groups_haversine_units() -> None:
    """ the helper must convert (ra, dec) to radians before
    invoking DBSCAN with metric='haversine'. We verify this indirectly: two
    points that are 0.05 degrees apart must end up in the same cluster
    (eps=0.1 deg), but if eps were misinterpreted as radians (~0.1 rad =
    ~5.7 deg), so would two points 4 degrees apart -- which we use as a
    negative control."""
    # Tight pair: 0.05 deg apart, should be neighbours under eps_deg=0.1.
    ra_tight = np.full(10, 30.0)
    dec_tight = 0.0 + 0.005 * np.arange(10)  # 0.045 deg max separation
    g_tight = derive_spatial_groups(
        ra_tight, dec_tight, eps_deg=0.1, min_samples=5
    )
    # All ten points should land in one DBSCAN cluster (>= 0), since the
    # span is 0.045 deg, well inside 0.1 deg.
    cluster_ids = set(g_tight[g_tight >= 0].tolist())
    assert len(cluster_ids) == 1, (
        f"tight 0.045-deg span must form one cluster, got {cluster_ids}"
    )
    # Far pair: 4 deg apart -- if the helper failed to convert to radians
    # and DBSCAN treated eps=0.1 as 0.1 rad (~5.7 deg), all points would
    # collapse into one cluster. With proper conversion (eps_rad = 0.001745),
    # they must NOT all be in the same cluster.
    ra_far = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0])
    dec_far = np.zeros_like(ra_far)
    g_far = derive_spatial_groups(
        ra_far, dec_far, eps_deg=0.1, min_samples=5
    )
    n_unique_far = int(len(np.unique(g_far)))
    assert n_unique_far > 1, (
        f"4-deg-spaced points must NOT collapse to one cluster under "
        f"eps_deg=0.1; got {n_unique_far} unique groups (radian-conversion "
        "likely missing)"
    )


def test_metrics_to_dict_handles_missing_cv_and_boundary() -> None:
    metrics = ClassifierMetrics(
        accuracy=0.5, macro_f1=0.5,
        per_class_precision={"F": 0.5}, per_class_recall={"F": 0.5},
        per_class_f1={"F": 0.5}, confusion_matrix=[[1]],
        class_labels=["F"], n_train=1, n_val=1, n_test=1,
    )
    payload = metrics_to_dict(metrics, None, None)
    assert payload["cv_fallback_triggered"] is True
    assert payload["cv_per_fold"] == []
    assert payload["boundary_filtered_accuracy"] == {}


def test_singleton_group_stats_basic() -> None:
    """Hand-counted: 3 singletons (groups 1,2,4), 2 clusters with size 2 and 3."""
    groups = np.array([1, 2, 3, 3, 4, 5, 5, 5], dtype=np.int64)
    s = singleton_group_stats(groups)
    assert s["n_total_rows"] == 8
    assert s["n_unique_groups"] == 5
    assert s["n_singleton_groups"] == 3
    assert s["n_rows_in_singleton_groups"] == 3
    assert s["n_non_singleton_clusters"] == 2
    assert s["non_singleton_cluster_size_min"] == 2
    assert s["non_singleton_cluster_size_max"] == 3
    assert abs(s["singleton_group_fraction"] - 3.0 / 5.0) < 1e-9
    assert abs(s["singleton_row_fraction"] - 3.0 / 8.0) < 1e-9
    # No negative IDs in this fixture, so DBSCAN-noise count is zero.
    assert s["n_dbscan_noise_rows"] == 0


def test_singleton_group_stats_dbscan_noise_negative_ids() -> None:
    """Negative-ID rows are flagged as DBSCAN-noise singletons separately."""
    groups = np.array([0, 0, 0, -1, -2, -3, -4], dtype=np.int64)
    s = singleton_group_stats(groups)
    # group 0 has 3 rows; -1, -2, -3, -4 are 4 singletons
    assert s["n_singleton_groups"] == 4
    assert s["n_rows_in_singleton_groups"] == 4
    assert s["n_dbscan_noise_rows"] == 4
    assert s["n_non_singleton_clusters"] == 1
    # 4 singleton groups / 5 unique = 0.8
    assert abs(s["singleton_group_fraction"] - 0.8) < 1e-9
    # 4 singleton rows / 7 total = 0.5714...
    assert abs(s["singleton_row_fraction"] - 4.0 / 7.0) < 1e-9
