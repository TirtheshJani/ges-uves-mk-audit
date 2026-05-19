"""Unit tests for src.interpret.triangulation.

Covers:
  - topk_excluding_mask drops gap-mask bins (HARD acceptance gate)
  - jaccard math (basic, empty)
  - pairwise_jaccard key shape
  - three_way_intersection
  - per_class_topk leaves no gap_mask bin in any class top-K (HARD gate)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.triangulation import (
    jaccard,
    pairwise_jaccard,
    per_class_topk,
    three_way_intersection,
    topk_excluding_mask,
)


def test_topk_excluding_mask_drops_masked_bins() -> None:
    # Largest values are at indices 9, 8, 7, but 8 and 9 are masked.
    values = np.arange(10, dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    mask[8] = True
    mask[9] = True

    top = topk_excluding_mask(values, k=3, exclude_mask=mask)

    assert top == {7, 6, 5}
    # Hard contract: no masked index allowed
    for idx in top:
        assert not mask[idx]


def test_topk_excluding_mask_returns_smaller_when_pool_small() -> None:
    values = np.arange(5, dtype=np.float32)
    mask = np.array([False, True, True, True, True], dtype=bool)
    top = topk_excluding_mask(values, k=3, exclude_mask=mask)
    assert top == {0}


def test_topk_excluding_mask_validates_shapes() -> None:
    values = np.arange(5, dtype=np.float32)
    bad_mask = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError):
        topk_excluding_mask(values, k=2, exclude_mask=bad_mask)


def test_topk_excluding_mask_validates_k() -> None:
    values = np.arange(5, dtype=np.float32)
    mask = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError):
        topk_excluding_mask(values, k=0, exclude_mask=mask)


def test_jaccard_basic() -> None:
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


def test_jaccard_empty_sets_return_one() -> None:
    assert jaccard(set(), set()) == 1.0


def test_jaccard_identical() -> None:
    assert jaccard({1, 2}, {1, 2}) == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_pairwise_jaccard_keys() -> None:
    sets = {
        "perm": {1, 2, 3},
        "shap": {2, 3, 4},
        "occlusion": {3, 4, 5},
    }
    pj = pairwise_jaccard(sets)
    assert set(pj.keys()) == {
        "perm_vs_shap",
        "perm_vs_occlusion",
        "shap_vs_occlusion",
    }
    assert pj["perm_vs_shap"] == pytest.approx(0.5)
    assert pj["perm_vs_occlusion"] == pytest.approx(1 / 5)
    assert pj["shap_vs_occlusion"] == pytest.approx(0.5)


def test_three_way_intersection() -> None:
    a = {1, 2, 3, 4}
    b = {2, 3, 4, 5}
    c = {3, 4, 5, 6}
    assert three_way_intersection(a, b, c) == {3, 4}


def test_three_way_intersection_empty() -> None:
    assert three_way_intersection({1, 2}, {3, 4}, {5, 6}) == set()


def test_per_class_topk_no_gap_bins() -> None:
    """HARD acceptance gate.

    No bin index inside the gap_mask may appear in any class top-K,
    even if it has the largest score in that class's score vector.
    """
    n_features = 20
    n_classes = 3
    rng = np.random.default_rng(42)
    scores = rng.random((n_classes, n_features)).astype(np.float32)
    # Force the gap bins to be the top-scoring features in every class
    gap_mask = np.zeros(n_features, dtype=bool)
    gap_mask[5:9] = True
    scores[:, 5:9] = 100.0  # would dominate a naive top-K
    labels = ["F", "G", "K"]

    out = per_class_topk(scores, k=5, exclude_mask=gap_mask, class_labels=labels)

    assert set(out.keys()) == {"F", "G", "K"}
    for label, idxs in out.items():
        assert len(idxs) == 5, f"{label} returned {len(idxs)} top-K entries"
        for idx in idxs:
            assert not gap_mask[idx], (
                f"gap-mask bin {idx} leaked into top-K for class {label}"
            )


def test_per_class_topk_validates_shapes() -> None:
    scores = np.zeros((3, 10), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError):
        per_class_topk(scores, k=3, exclude_mask=mask, class_labels=["F", "G"])


def test_per_class_topk_indices_sorted_ascending() -> None:
    n_features = 10
    scores = np.array(
        [
            [0.1, 0.2, 0.9, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.zeros(n_features, dtype=bool)
    out = per_class_topk(scores, k=3, exclude_mask=mask, class_labels=["A", "B"])
    assert out["A"] == sorted(out["A"])
    assert out["B"] == sorted(out["B"])
    assert set(out["A"]) == {2, 4, 3}
    assert set(out["B"]) == {5, 6, 7}
