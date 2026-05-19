# SPDX-License-Identifier: MIT
"""Top-K and Jaccard helpers for the interpretability triangulation.

The pipeline triangulates three independent feature-importance
estimators on the validation split: permutation importance, mean
absolute TreeSHAP, and sliding-window occlusion. Top-K bin sets are
compared via pairwise and three-way Jaccard.

(gap_mask): the UVES inter-chip gap [5769, 5834] A is
imputed at feature-build time and bins inside that band must NEVER
appear in a reported top-K. This module enforces that as a hard
contract via ``topk_excluding_mask`` and ``per_class_topk``.

References:
  - Sacco et al. 2014 A&A 565, A113 (UVES inter-chip gap)
  - Ness et al. 2015 ApJ 808, 16 (interpretability triangulation)
  - Li, Lin and Qiu 2024 (perm + SHAP for stellar tree models)
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


def topk_excluding_mask(
    values: np.ndarray, k: int, exclude_mask: np.ndarray
) -> set[int]:
    """Return the indices of the top-``k`` largest entries of ``values``,
    excluding any index where ``exclude_mask`` is True.

    Parameters
    ----------
    values: np.ndarray
        1-D array of importance scores (higher is more important).
    k: int
        Number of indices to return. Must be >= 1.
    exclude_mask: np.ndarray
        1-D boolean array, same length as ``values``. Indices where
        ``exclude_mask`` is True are ineligible for the top-K.

    Returns
    -------
    set[int]
        Up to ``k`` indices into ``values``. May be smaller than ``k``
        if the eligible pool is smaller than ``k``.
    """
    values = np.asarray(values)
    exclude_mask = np.asarray(exclude_mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if exclude_mask.shape != values.shape:
        raise ValueError(
            f"exclude_mask shape {exclude_mask.shape} != values shape {values.shape}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    eligible = np.flatnonzero(~exclude_mask)
    if eligible.size == 0:
        return set()

    eligible_vals = values[eligible]
    take = min(k, eligible.size)
    # argsort ascending; pick last `take` (largest)
    top_local = np.argsort(eligible_vals)[-take:]
    top_global = eligible[top_local]
    return set(int(i) for i in top_global)


def jaccard(a: set[int], b: set[int]) -> float:
    """Return the Jaccard index of two sets.

    By convention, Jaccard of two empty sets is 1.0 (perfect agreement
    on the empty support).
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def pairwise_jaccard(top_sets: dict[str, set[int]]) -> dict[str, float]:
    """Return Jaccard for each unordered pair of method top-K sets.

    Output keys are ``"{m1}_vs_{m2}"`` with method names sorted by their
    insertion order in ``top_sets``.
    """
    names = list(top_sets.keys())
    out: dict[str, float] = {}
    for a, b in combinations(names, 2):
        out[f"{a}_vs_{b}"] = jaccard(top_sets[a], top_sets[b])
    return out


def three_way_intersection(
    a: set[int], b: set[int], c: set[int]
) -> set[int]:
    """Return ``a & b & c``."""
    return a & b & c


def per_class_topk(
    scores_per_class: np.ndarray,
    k: int,
    exclude_mask: np.ndarray,
    class_labels: list[str],
) -> dict[str, list[int]]:
    """Return per-class top-K bin indices, gap-mask-excluded.

    Parameters
    ----------
    scores_per_class: np.ndarray
        Shape ``(n_classes, n_features)``; row c is the score vector
        for class ``class_labels[c]``.
    k: int
        Top-K size.
    exclude_mask: np.ndarray
        1-D boolean array of length ``n_features``.
    class_labels: list[str]
        Class names, length ``n_classes``.

    Returns
    -------
    dict[str, list[int]]
        Mapping from class label to a list of top-K bin indices, sorted
        ascending. No index in any list lies inside ``exclude_mask``.
    """
    scores_per_class = np.asarray(scores_per_class)
    if scores_per_class.ndim != 2:
        raise ValueError(
            f"scores_per_class must be 2-D, got shape {scores_per_class.shape}"
        )
    n_classes = scores_per_class.shape[0]
    if len(class_labels) != n_classes:
        raise ValueError(
            f"len(class_labels)={len(class_labels)} != n_classes={n_classes}"
        )
    out: dict[str, list[int]] = {}
    for c, label in enumerate(class_labels):
        s = topk_excluding_mask(scores_per_class[c], k, exclude_mask)
        out[label] = sorted(s)
    return out
