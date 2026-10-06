# SPDX-License-Identifier: MIT
"""LightGBM multiclass wrapper for the MK interpretability pipeline.

Hyper-parameters are fixed before training: balanced class weights, shallow
enough trees to keep SHAP tractable, early stopping on validation loss.

The wrapper exposes the sklearn API (``fit``/``predict``) so that
``sklearn.inspection.permutation_importance`` and ``shap.TreeExplainer``
both work without glue code.

adds group-aware cross-validation per and a
boundary-distance sensitivity check per.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HPARAMS: Final[dict[str, Any]] = {
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

DEFAULT_EARLY_STOPPING: Final[int] = 50

CV_TOTAL_TIMEOUT_S: Final[float] = 14400.0


@dataclass
class ClassifierMetrics:
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    n_train: int
    n_val: int
    n_test: int


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_class: int,
    hparams: dict[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
) -> Any:
    """Train a ``lightgbm.LGBMClassifier`` with validation-based early stopping.

    Returns the fitted estimator (already has ``best_iteration_``).
    """
    import lightgbm as lgb

    params = dict(DEFAULT_HPARAMS)
    params["num_class"] = int(num_class)
    if hparams:
        params.update(hparams)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    logger.info(
        "training complete; best_iteration=%s",
        getattr(model, "best_iteration_", None),
    )
    return model


def evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str],
    n_train: int = 0,
    n_val: int = 0,
) -> ClassifierMetrics:
    """Compute accuracy, macro-F1, per-class precision/recall/F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    # Use the model's own classes_ array as the canonical label set so the
    # confusion matrix and per-class metrics align with the encoded y values
    # (which are [1, 2, 3] for F/G/K after dropped A=0).
    model_classes = getattr(model, "classes_", None)
    if model_classes is not None and len(model_classes) == len(class_labels):
        present = np.asarray(model_classes)
    else:
        present = np.arange(len(class_labels))
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=present, average=None, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=present).tolist()
    return ClassifierMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class_precision={class_labels[i]: float(p[i]) for i in range(len(class_labels))},
        per_class_recall={class_labels[i]: float(r[i]) for i in range(len(class_labels))},
        per_class_f1={class_labels[i]: float(f[i]) for i in range(len(class_labels))},
        confusion_matrix=cm,
        class_labels=list(class_labels),
        n_train=int(n_train),
        n_val=int(n_val),
        n_test=int(len(y_test)),
    )


def derive_spatial_groups(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    eps_deg: float = 0.1,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster (ra, dec) into spatial groups for leakage-aware CV.

    amendment to the original ``groups`` array in
    features.npz was zero-filled, so ``StratifiedGroupKFold`` degenerated
    to ``StratifiedKFold`` at. This helper derives groups from the
    on-sky coordinates so can run a second sensitivity check that
    holds out cluster-mate spectra together (Blanco-Cuaresma 2019,
    MNRAS 486, 2075; bibcode 2019MNRAS.486.2075B).

    Uses ``sklearn.cluster.DBSCAN`` with ``metric='haversine'``. Haversine
    expects radians and the (latitude, longitude) tuple convention, so we
    convert ``(ra_deg, dec_deg)`` to radians and stack as ``(dec, ra)``.
    DBSCAN is deterministic for a fixed input (no ``random_state`` is used).
    Noise points (DBSCAN label ``-1``) are each assigned a unique negative
    group ID so they cannot be collapsed into a single giant pseudo-cluster
    that would defeat group-aware splitting.

    Parameters
    ----------
    ra_deg: np.ndarray
        Right ascension in degrees, shape ``(n)``, range ``[0, 360)``.
    dec_deg: np.ndarray
        Declination in degrees, shape ``(n)``, range ``[-90, 90]``.
    eps_deg: float
        DBSCAN neighbourhood radius in degrees. default 0.1 deg
        captures ~30 percent of GES open-cluster fields per the DBSCAN probe. Internally converted to radians.
    min_samples: int
        DBSCAN minimum samples per core point. Default 5 follows scikit-learn
        convention.

    Returns
    -------
    np.ndarray
        ``int64`` array of shape ``(n)``. Cluster members carry their
        DBSCAN cluster index (>= 0); noise points carry distinct negative
        IDs (``-1 - i`` for the ``i``-th noise point).
    """
    from sklearn.cluster import DBSCAN

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    if ra_deg.shape != dec_deg.shape:
        raise ValueError(
            f"ra_deg and dec_deg shape mismatch: "
            f"{ra_deg.shape} vs {dec_deg.shape}"
        )
    if ra_deg.ndim != 1:
        raise ValueError(f"ra_deg must be 1-D, got shape {ra_deg.shape}")

    # Haversine expects (lat, lon) in radians.
    coords_rad = np.deg2rad(np.column_stack([dec_deg, ra_deg]))
    eps_rad = float(np.deg2rad(eps_deg))

    db = DBSCAN(
        eps=eps_rad,
        min_samples=int(min_samples),
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = db.fit_predict(coords_rad).astype(np.int64)

    # Replace each DBSCAN noise label (-1) with a unique negative ID so the
    # noise points are not collapsed into a single group when used by
    # StratifiedGroupKFold.
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise > 0:
        noise_idx = np.flatnonzero(noise_mask)
        labels[noise_idx] = (-1 - np.arange(n_noise, dtype=np.int64))

    cluster_mask = labels >= 0
    if cluster_mask.any():
        n_clusters = int(len(np.unique(labels[cluster_mask])))
    else:
        n_clusters = 0
    n_unique = int(len(np.unique(labels)))
    logger.info(
        "derive_spatial_groups: n=%d eps_deg=%.4f min_samples=%d "
        "-> n_clusters=%d n_noise=%d n_unique_groups=%d",
        len(labels), eps_deg, min_samples, n_clusters, n_noise, n_unique,
    )
    return labels


def singleton_group_stats(groups: np.ndarray) -> dict[str, int | float]:
    """Compute singleton-group transparency stats per.

    when DBSCAN noise points get unique negative IDs, each
    noise point is its own singleton group; StratifiedGroupKFold then
    effectively does per-row stratification for those rows rather than
    group stratification. A standard sanity check is that this masks
    cluster leakage that the spatial-CV macro-F1 gap does not bound.

    Returns counts and fractions:
      - n_total_rows
      - n_unique_groups
      - n_singleton_groups (groups containing exactly one row)
      - n_rows_in_singleton_groups (numerically equal to n_singleton_groups)
      - singleton_group_fraction (singletons / unique groups)
      - singleton_row_fraction (singleton rows / total rows)
      - n_dbscan_noise_rows (rows with group_id < 0, i.e. negative-ID
        singletons assigned by derive_spatial_groups)
      - n_non_singleton_clusters (non-singleton clusters with size >= 2)
      - non_singleton_cluster_size_{min, median, max}
    """
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    singleton_mask = counts == 1
    n_singleton_groups = int(singleton_mask.sum())
    n_unique = int(len(unique))
    singleton_ids = set(unique[singleton_mask].tolist())
    n_rows_in_singletons = int(np.isin(groups, list(singleton_ids)).sum())
    n_total = int(len(groups))
    n_noise = int((groups < 0).sum())
    non_singleton_counts = counts[~singleton_mask]
    if len(non_singleton_counts):
        cluster_min = int(non_singleton_counts.min())
        cluster_max = int(non_singleton_counts.max())
        cluster_med = int(np.median(non_singleton_counts))
    else:
        cluster_min = cluster_max = cluster_med = 0
    return {
        "n_total_rows": n_total,
        "n_unique_groups": n_unique,
        "n_singleton_groups": n_singleton_groups,
        "n_rows_in_singleton_groups": n_rows_in_singletons,
        "singleton_group_fraction": (
            float(n_singleton_groups / n_unique) if n_unique else float("nan")
        ),
        "singleton_row_fraction": (
            float(n_rows_in_singletons / n_total) if n_total else float("nan")
        ),
        "n_dbscan_noise_rows": n_noise,
        "n_non_singleton_clusters": int(len(non_singleton_counts)),
        "non_singleton_cluster_size_min": cluster_min,
        "non_singleton_cluster_size_median": cluster_med,
        "non_singleton_cluster_size_max": cluster_max,
    }


def train_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_class: int,
    n_splits: int = 5,
    random_state: int = 42,
    hparams: dict[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
    per_fold_timeout_s: float = 3600.0,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Run group-aware stratified k-fold CV and return per-fold metrics.

    Uses ``sklearn.model_selection.StratifiedGroupKFold`` so that spatially
    co-located stars never straddle a fold boundary (Blanco-Cuaresma 2019,
    MNRAS 486, 2075). When the supplied ``groups``
    array carries fewer unique values than ``n_splits``, the splitter cannot
    form valid folds, and the function logs the situation and degrades to
    ``StratifiedKFold`` without re-deriving groups.

    Returns a dict with keys:
      - cv_macro_f1_mean, cv_macro_f1_std
      - per_fold: list of {fold, n_train, n_val, macro_f1, accuracy,
        per_class_recall, confusion_matrix, runtime_s, best_iteration}
      - cv_fallback_triggered: bool
      - cv_fallback_reason: str (empty if not triggered)
      - splitter: name of splitter used
      - total_runtime_s: float
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    n_unique_groups = int(len(np.unique(groups)))
    logger.info(
        "train_cv: n=%d, n_splits=%d, n_unique_groups=%d, num_class=%d",
        len(y), n_splits, n_unique_groups, num_class,
    )

    splitter_name = "StratifiedGroupKFold"
    fallback_reason = ""
    if n_unique_groups < n_splits:
        splitter_name = "StratifiedKFold"
        fallback_reason = (
            f"n_unique_groups={n_unique_groups} < n_splits={n_splits}; "
            "group-aware CV degenerates to single fold; using StratifiedKFold"
        )
        logger.warning("train_cv: %s", fallback_reason)
        splitter: Any = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_iter = splitter.split(X, y, groups=groups)

    per_fold: list[dict[str, Any]] = []
    fold_macro_f1s: list[float] = []
    cv_fallback_triggered = False
    cumulative_runtime = 0.0

    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
        if cumulative_runtime > CV_TOTAL_TIMEOUT_S:
            cv_fallback_triggered = True
            fallback_reason = (
                f"cumulative CV runtime {cumulative_runtime:.1f}s exceeded "
                f"{CV_TOTAL_TIMEOUT_S:.0f}s budget"
            )
            logger.warning("train_cv: %s", fallback_reason)
            break

        # group-disjoint sanity check
        if n_unique_groups >= n_splits:
            tr_groups = set(np.unique(groups[tr_idx]).tolist())
            va_groups = set(np.unique(groups[va_idx]).tolist())
            assert not (tr_groups & va_groups), (
                f"fold {fold_idx}: groups appear in both train and val"
            )

        logger.info(
            "fold %d/%d: n_train=%d n_val=%d",
            fold_idx, n_splits, len(tr_idx), len(va_idx),
        )

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        t0 = time.perf_counter()
        try:
            model = train(
                X_train=X_tr, y_train=y_tr,
                X_val=X_va, y_val=y_va,
                num_class=num_class,
                hparams=hparams,
                early_stopping_rounds=early_stopping_rounds,
            )
        except Exception as exc:
            cv_fallback_triggered = True
            fallback_reason = f"fold {fold_idx} raised {type(exc).__name__}: {exc}"
            logger.warning("train_cv: %s", fallback_reason)
            break
        runtime_s = float(time.perf_counter() - t0)
        cumulative_runtime += runtime_s

        if runtime_s > per_fold_timeout_s:
            cv_fallback_triggered = True
            fallback_reason = (
                f"fold {fold_idx} runtime {runtime_s:.1f}s exceeded "
                f"per-fold budget {per_fold_timeout_s:.0f}s"
            )
            logger.warning("train_cv: %s", fallback_reason)
            break

        labels_for_eval = class_labels or [str(i) for i in range(num_class)]
        m = evaluate(
            model, X_va, y_va,
            class_labels=labels_for_eval,
            n_train=int(len(tr_idx)),
            n_val=int(len(va_idx)),
        )
        fold_record = {
            "fold": int(fold_idx),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "macro_f1": float(m.macro_f1),
            "accuracy": float(m.accuracy),
            "per_class_recall": dict(m.per_class_recall),
            "confusion_matrix": list(m.confusion_matrix),
            "runtime_s": runtime_s,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }
        per_fold.append(fold_record)
        fold_macro_f1s.append(float(m.macro_f1))
        logger.info(
            "fold %d done: macro_f1=%.4f acc=%.4f runtime=%.1fs",
            fold_idx, m.macro_f1, m.accuracy, runtime_s,
        )

    if fold_macro_f1s:
        cv_mean = float(np.mean(fold_macro_f1s))
        cv_std = float(np.std(fold_macro_f1s, ddof=0))
    else:
        cv_mean = float("nan")
        cv_std = float("nan")

    result: dict[str, Any] = {
        "cv_macro_f1_mean": cv_mean,
        "cv_macro_f1_std": cv_std,
        "per_fold": per_fold,
        "cv_fallback_triggered": bool(cv_fallback_triggered),
        "cv_fallback_reason": fallback_reason,
        "splitter": splitter_name,
        "n_splits": int(n_splits),
        "total_runtime_s": float(cumulative_runtime),
    }
    logger.info(
        "train_cv summary: splitter=%s mean=%.4f std=%.4f folds=%d fallback=%s",
        splitter_name, cv_mean, cv_std, len(per_fold), cv_fallback_triggered,
    )
    return result


def boundary_filtered_accuracy(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    boundary_distance_k: np.ndarray,
    threshold_k: float = 200.0,
) -> dict[str, float]:
    """Sensitivity check per.

    Computes test accuracy on the subset where ``|boundary_distance_k| > threshold_k``
    and compares it to full-test accuracy. A physically-aware classifier should
    score higher on the boundary-filtered subset because temperature-edge cases
    are removed. Bootstrap 95 percent CI on delta_acc with 1000 resamples,
    seed=42. The filter is applied to the test split only.
    """
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    bd = np.asarray(boundary_distance_k)
    if len(bd) != len(y_test):
        raise ValueError(
            f"boundary_distance_k length {len(bd)} != y_test length {len(y_test)}"
        )

    y_pred = model.predict(X_test)
    correct = (y_pred == y_test).astype(np.int64)

    full_acc = float(correct.mean())
    mask = np.abs(bd) > threshold_k
    n_filtered = int(mask.sum())
    if n_filtered == 0:
        logger.warning(
            "boundary_filtered_accuracy: no test rows with |bd|>%s; returning NaN delta",
            threshold_k,
        )
        filt_acc = float("nan")
        delta = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        filt_acc = float(correct[mask].mean())
        delta = filt_acc - full_acc

        rng = np.random.default_rng(seed=42)
        n_resample = 1000
        n_test = len(y_test)
        deltas = np.empty(n_resample, dtype=np.float64)
        for i in range(n_resample):
            idx = rng.integers(0, n_test, size=n_test)
            c_boot = correct[idx]
            m_boot = mask[idx]
            full_boot = float(c_boot.mean())
            if m_boot.sum() == 0:
                deltas[i] = 0.0
            else:
                filt_boot = float(c_boot[m_boot].mean())
                deltas[i] = filt_boot - full_boot
        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))

    out = {
        "full_test_acc": full_acc,
        "boundary_filtered_acc": filt_acc,
        "delta_acc": float(delta),
        "delta_acc_ci_low": ci_low,
        "delta_acc_ci_high": ci_high,
        "threshold_k": float(threshold_k),
        "n_filtered": float(n_filtered),
        "n_test": float(len(y_test)),
    }
    logger.info(
        "boundary_filtered_accuracy: full=%.4f filtered=%.4f delta=%.4f "
        "ci=[%.4f, %.4f] n_filtered=%d/%d threshold_k=%.0f",
        full_acc, filt_acc, delta, ci_low, ci_high,
        n_filtered, len(y_test), threshold_k,
    )
    return out


def metrics_to_dict(
    metrics: ClassifierMetrics,
    cv_results: dict[str, Any] | None,
    boundary: dict[str, float] | None,
) -> dict[str, Any]:
    """Compose the flat metrics.json payload for.

    Test metrics from ``ClassifierMetrics`` are flattened. CV summary keys are
    promoted to the top level; per-fold records are nested under
    ``cv_per_fold``. Boundary-filter results are nested under
    ``boundary_filtered_accuracy``.
    """
    payload: dict[str, Any] = dict(asdict(metrics))

    if cv_results is None:
        payload["cv_macro_f1_mean"] = float("nan")
        payload["cv_macro_f1_std"] = 0.0
        payload["cv_fallback_triggered"] = True
        payload["cv_fallback_reason"] = "cv not run"
        payload["cv_splitter"] = ""
        payload["cv_n_splits"] = 0
        payload["cv_total_runtime_s"] = 0.0
        payload["cv_per_fold"] = []
    else:
        payload["cv_macro_f1_mean"] = float(cv_results.get("cv_macro_f1_mean", float("nan")))
        payload["cv_macro_f1_std"] = float(cv_results.get("cv_macro_f1_std", 0.0))
        payload["cv_fallback_triggered"] = bool(cv_results.get("cv_fallback_triggered", False))
        payload["cv_fallback_reason"] = str(cv_results.get("cv_fallback_reason", ""))
        payload["cv_splitter"] = str(cv_results.get("splitter", ""))
        payload["cv_n_splits"] = int(cv_results.get("n_splits", 0))
        payload["cv_total_runtime_s"] = float(cv_results.get("total_runtime_s", 0.0))
        payload["cv_per_fold"] = list(cv_results.get("per_fold", []))

    if boundary is None:
        payload["boundary_filtered_accuracy"] = {}
    else:
        payload["boundary_filtered_accuracy"] = dict(boundary)
    return payload


def save_model(model: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("saved model -> %s", path)


def load_model(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_metrics(metrics: ClassifierMetrics | dict[str, Any], path: Path) -> None:
    """Write metrics to JSON. Accepts either a ``ClassifierMetrics`` (legacy)
    or a flat dict (schema produced by ``metrics_to_dict``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics) if isinstance(metrics, ClassifierMetrics) else dict(metrics)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote metrics -> %s", path)
