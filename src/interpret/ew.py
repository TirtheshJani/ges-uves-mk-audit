# SPDX-License-Identifier: MIT
"""Equivalent-width measurement on continuum-normalised regridded spectra.

The Mg b saturation argument should rest on the K-class test spectra
themselves rather than on Pickles K templates. This module measures equivalent widths directly on the audit's own
145 K-class and 230 G-class test rows so the Section 5 saturation argument
can rest on the dataset under audit, not on a surrogate library.

Equivalent width definition (positive for absorption):
    EW = sum_{i in line bins} (1 - F_i / F_cont) * delta_lambda_i

F_cont is estimated per row as the median of "off-line" bins in a +/-15 A
window around each line, excluding bins within +/-half_width_aa of the line
centre. This local-continuum convention is robust to small departures from
the global continuum normalisation level (which is 0.916 here per empirical median, not 1.0).

Rest wavelengths are NIST AIR values per the canonical MK list in
 Mg b1=5167.32, Mg b2=5172.68, Mg b3=5183.60.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# NIST AIR values per canonical MK list.
MG_B_LINES_AA: tuple[float, ...] = (5167.32, 5172.68, 5183.60)
DEFAULT_LINE_HALF_WIDTH_AA: float = 3.0
DEFAULT_CONTINUUM_HALF_WIDTH_AA: float = 15.0


@dataclass
class EquivalentWidthRow:
    """Per-line EW result for one spectrum row."""
    row_index: int
    line_centre_aa: float
    n_line_bins: int
    n_continuum_bins: int
    continuum_level: float
    ew_aa: float


def _bins_in_window(
    wave_centers: np.ndarray, centre_aa: float, half_width_aa: float
) -> np.ndarray:
    lo = centre_aa - half_width_aa
    hi = centre_aa + half_width_aa
    return (wave_centers >= lo) & (wave_centers <= hi)


def measure_ew_one_line(
    flux: np.ndarray,
    wave_centers: np.ndarray,
    line_centre_aa: float,
    line_half_width_aa: float = DEFAULT_LINE_HALF_WIDTH_AA,
    continuum_half_width_aa: float = DEFAULT_CONTINUUM_HALF_WIDTH_AA,
) -> tuple[float, float, int, int]:
    """Measure equivalent width of one absorption line on one spectrum row.

    Returns (ew_aa, continuum_level, n_line_bins, n_continuum_bins).

    Local continuum is the median of bins in [centre - continuum_half_width,
    centre + continuum_half_width] that fall OUTSIDE the line window. EW is
    a Riemann sum over the line bins of (1 - F/F_cont) weighted by the
    centre-to-centre bin spacing.
    """
    line_mask = _bins_in_window(wave_centers, line_centre_aa, line_half_width_aa)
    cont_window = _bins_in_window(wave_centers, line_centre_aa, continuum_half_width_aa)
    cont_mask = cont_window & ~line_mask
    if not line_mask.any() or not cont_mask.any():
        return float("nan"), float("nan"), int(line_mask.sum()), int(cont_mask.sum())
    cont_level = float(np.median(flux[cont_mask]))
    if not np.isfinite(cont_level) or cont_level <= 0:
        return float("nan"), float(cont_level), int(line_mask.sum()), int(cont_mask.sum())
    diffs = np.diff(wave_centers)
    mean_bin_aa = float(np.mean(diffs))
    depth = 1.0 - flux[line_mask] / cont_level
    ew = float(np.sum(depth) * mean_bin_aa)
    return ew, cont_level, int(line_mask.sum()), int(cont_mask.sum())


def measure_mg_b_triplet(
    X: np.ndarray,
    wave_centers: np.ndarray,
    line_half_width_aa: float = DEFAULT_LINE_HALF_WIDTH_AA,
    continuum_half_width_aa: float = DEFAULT_CONTINUUM_HALF_WIDTH_AA,
) -> dict[str, np.ndarray]:
    """Measure Mg b1 + b2 + b3 equivalent widths on each row of X.

    Returns a dict with keys:
      - ew_mg_b1, ew_mg_b2, ew_mg_b3: per-row EW for each triplet member
      - ew_mg_b_total: sum of the three (positive for absorption)
      - continuum_mg_b1, continuum_mg_b2, continuum_mg_b3: per-row local
        continuum level used per line (sanity check; should be near the
        global normalisation level, ~0.916 in production data)
    """
    n_rows = X.shape[0]
    ew = np.full((n_rows, len(MG_B_LINES_AA)), np.nan, dtype=np.float64)
    cont = np.full_like(ew, np.nan)
    for j, line in enumerate(MG_B_LINES_AA):
        for r in range(n_rows):
            ew_val, cont_val, _, _ = measure_ew_one_line(
                X[r], wave_centers, line,
                line_half_width_aa=line_half_width_aa,
                continuum_half_width_aa=continuum_half_width_aa,
            )
            ew[r, j] = ew_val
            cont[r, j] = cont_val
    return {
        "ew_mg_b1": ew[:, 0],
        "ew_mg_b2": ew[:, 1],
        "ew_mg_b3": ew[:, 2],
        "ew_mg_b_total": np.nansum(ew, axis=1),
        "continuum_mg_b1": cont[:, 0],
        "continuum_mg_b2": cont[:, 1],
        "continuum_mg_b3": cont[:, 2],
    }


def ks_two_sample(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value.

    Implemented locally rather than importing scipy.stats to keep this
    module's dependency surface minimal (the rest of the repo already uses
    numpy directly; scipy is only pulled in by sklearn). The asymptotic
    p-value uses the Kolmogorov-Smirnov limiting distribution which is
    accurate for combined n*m / (n+m) > 4 (Press et al. 2007).
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, combined, side="right") / na
    cdf_b = np.searchsorted(b, combined, side="right") / nb
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(na * nb / (na + nb))
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov distribution: Q(lam) = 2 * sum_{j>=1} (-1)^(j-1) * exp(-2 j^2 lam^2)
    j = np.arange(1, 101, dtype=np.float64)
    q = 2.0 * float(np.sum(((-1) ** (j - 1)) * np.exp(-2.0 * j ** 2 * lam ** 2)))
    p = max(0.0, min(1.0, q))
    return d, p


def summary_per_class(
    values: np.ndarray,
    labels: np.ndarray,
    class_labels: tuple[int, ...] = (1, 2, 3),
    class_names: tuple[str, ...] = ("F", "G", "K"),
) -> dict[str, dict[str, float]]:
    """Compute median, IQR, mean, std, n for ``values`` grouped by ``labels``."""
    out: dict[str, dict[str, float]] = {}
    for lab, name in zip(class_labels, class_names):
        v = values[labels == lab]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[name] = {
                "n": 0,
                "median_aa": float("nan"),
                "iqr_aa": float("nan"),
                "mean_aa": float("nan"),
                "std_aa": float("nan"),
            }
            continue
        q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        out[name] = {
            "n": int(len(v)),
            "median_aa": float(np.median(v)),
            "iqr_aa": q3 - q1,
            "q1_aa": q1,
            "q3_aa": q3,
            "mean_aa": float(np.mean(v)),
            "std_aa": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }
    return out
