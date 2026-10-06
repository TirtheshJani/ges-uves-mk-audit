# SPDX-License-Identifier: MIT
"""Sliding-window occlusion trace for the MK classifier.

Masking replaces flux inside a wavelength window with 1.0 (the
continuum-normalized level), not with zero. This avoids injecting a
synthetic absorption feature where the line used to be.

The masked-line ablation with its random-window reference set lives in
:mod:`src.interpret.ablation`; its public names are re-exported here so
that existing imports from this module keep working.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.interpret.ablation import (  # noqa: F401  (re-exported for backwards compatibility)
    CONTINUUM_FILL,
    VALID_FILL_MODES,
    AblationRow,
    _apply_mask,
    _per_class_recall,
    _sample_random_windows,
    _sample_random_windows_bins,
    _window_mask,
    masked_line_ablation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONTINUUM_FILL",
    "VALID_FILL_MODES",
    "AblationRow",
    "masked_line_ablation",
    "sliding_window_occlusion",
]


def sliding_window_occlusion(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    wave_centers: np.ndarray,
    window_aa: float = 50.0,
    stride_aa: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a single-window mask across ``wave_centers`` and report delta_acc.

    Returns (window_centers, delta_acc). delta_acc is negative when masking
    the window hurts accuracy.
    """
    baseline_acc = float(np.mean(model.predict(X) == y))
    wmin = float(wave_centers.min())
    wmax = float(wave_centers.max())
    centers = np.arange(wmin + window_aa / 2, wmax - window_aa / 2 + 1e-6, stride_aa)
    delta = np.empty(len(centers), dtype=np.float32)
    for i, c in enumerate(centers):
        lo = c - window_aa / 2
        hi = c + window_aa / 2
        mask = (wave_centers >= lo) & (wave_centers <= hi)
        X_masked = _apply_mask(X, mask)
        masked_acc = float(np.mean(model.predict(X_masked) == y))
        delta[i] = masked_acc - baseline_acc
    logger.info(
        "sliding occlusion: %d windows, baseline_acc=%.4f, min_delta=%.4f",
        len(centers), baseline_acc, float(delta.min()),
    )
    return centers.astype(np.float32), delta
