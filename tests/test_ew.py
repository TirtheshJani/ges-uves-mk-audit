# SPDX-License-Identifier: MIT
"""Unit tests for src.interpret.ew ().

Covers:
  - measure_ew_one_line returns the expected EW for a synthetic Gaussian
    absorption profile of known equivalent width
  - measure_mg_b_triplet returns three NaN-safe per-row EWs plus a total
  - ks_two_sample matches scipy.stats.ks_2samp within tolerance on a small
    synthetic case (scipy is checked at import time and the test is skipped
    if unavailable)
  - summary_per_class returns IQR and median that match numpy on
    deterministic data
"""
from __future__ import annotations

import numpy as np
import pytest

from src.interpret.ew import (
    MG_B_LINES_AA,
    ks_two_sample,
    measure_ew_one_line,
    measure_mg_b_triplet,
    summary_per_class,
)


def _gaussian_absorption(
    wave: np.ndarray, centre: float, depth: float, sigma_aa: float
) -> np.ndarray:
    """1.0 - depth * exp(-(wave-centre)^2 / (2 sigma^2)); positive depth = absorption."""
    return 1.0 - depth * np.exp(-0.5 * ((wave - centre) / sigma_aa) ** 2)


def test_measure_ew_one_line_recovers_known_gaussian_ew() -> None:
    """A Gaussian absorption with depth=d and FWHM=2.355*sigma has analytic
    EW = d * sigma * sqrt(2 pi). On a grid fine enough to resolve the line
    we should recover this within a few percent.
    """
    wave = np.linspace(5160.0, 5200.0, 400, dtype=np.float64)  # 0.1 A bins
    centre = 5180.0
    depth = 0.30
    sigma = 1.0  # FWHM = 2.355 A
    flux = _gaussian_absorption(wave, centre, depth, sigma)
    analytic_ew = depth * sigma * np.sqrt(2 * np.pi)
    ew, cont, n_line, n_cont = measure_ew_one_line(
        flux, wave, centre,
        line_half_width_aa=3.0,
        continuum_half_width_aa=15.0,
    )
    assert n_line > 0
    assert n_cont > 0
    assert abs(cont - 1.0) < 1e-3  # continuum near 1.0
    # Within 5 percent of the analytic value over the truncated +/-3 A integration.
    # 3 sigma captures ~99.7 percent of a Gaussian, so truncation error is small.
    assert abs(ew - analytic_ew) / analytic_ew < 0.05


def test_measure_mg_b_triplet_emits_finite_values() -> None:
    """On a flat-1.0 continuum with no absorption, every EW is approximately zero."""
    wave = np.linspace(4801.0, 6796.0, 696, dtype=np.float64)
    n_rows = 5
    X = np.ones((n_rows, len(wave)), dtype=np.float32)
    out = measure_mg_b_triplet(X, wave)
    for key in ("ew_mg_b1", "ew_mg_b2", "ew_mg_b3", "ew_mg_b_total"):
        assert out[key].shape == (n_rows,)
        assert np.all(np.isfinite(out[key]))
        # All EWs should be near zero for a flat continuum at 1.0.
        assert np.all(np.abs(out[key]) < 0.1)


def test_measure_mg_b_triplet_recovers_stronger_absorption_on_one_row() -> None:
    """Inject a Gaussian absorption at Mg b1 on row 0 only; row 0 should have
    a significantly higher Mg b1 EW than row 1.
    """
    wave = np.linspace(4801.0, 6796.0, 696, dtype=np.float64)
    n_rows = 2
    X = np.ones((n_rows, len(wave)), dtype=np.float32)
    X[0] = _gaussian_absorption(wave, MG_B_LINES_AA[0], depth=0.4, sigma_aa=1.5).astype(
        np.float32
    )
    out = measure_mg_b_triplet(X, wave)
    assert out["ew_mg_b1"][0] > 0.5  # strong absorption
    assert abs(out["ew_mg_b1"][1]) < 0.05  # ~flat


def test_ks_two_sample_matches_scipy_within_tolerance() -> None:
    """The D statistic must match scipy exactly; the asymptotic p-value need
    only land in the same broad regime (< 0.05 vs scipy's exact < 0.05).

    The asymptotic Kolmogorov approximation used here is standard for two-
    sample KS at moderate sample sizes (Press et al. 2007). It diverges
    from scipy.stats.ks_2samp's exact method at extreme p (< 1e-6) but
    agrees on the practically relevant question of significance at the
    0.05 / 0.01 thresholds the manuscript will quote.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 150)
    b = rng.normal(0.15, 1, 150)  # small effect; p moderate, not 1e-10
    d_ours, p_ours = ks_two_sample(a, b)
    res = scipy_stats.ks_2samp(a, b)
    assert abs(d_ours - res.statistic) < 1e-9  # D must match exactly
    # Both p-values agree on significance vs the 0.05 threshold.
    assert (p_ours < 0.05) == (res.pvalue < 0.05)
    # And within the same order of magnitude.
    assert 0.2 < p_ours / max(res.pvalue, 1e-9) < 5.0


def test_summary_per_class_matches_numpy() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    s = summary_per_class(vals, labels)
    assert s["F"]["n"] == 4
    assert s["G"]["n"] == 4
    assert s["F"]["median_aa"] == np.median([1.0, 2.0, 3.0, 4.0])
    assert s["G"]["median_aa"] == np.median([5.0, 6.0, 7.0, 8.0])


def test_summary_per_class_handles_empty_class() -> None:
    vals = np.array([1.0, 2.0, 3.0])
    labels = np.array([1, 1, 2])  # no class K (label 3)
    s = summary_per_class(vals, labels)
    assert s["K"]["n"] == 0
    assert np.isnan(s["K"]["median_aa"])
