# SPDX-License-Identifier: MIT
"""Sliding-window occlusion trace and masked-line ablation with null controls.

Masking replaces flux inside a wavelength window with 1.0 (the
continuum-normalized level), not with zero. This avoids injecting a
synthetic absorption feature where the line used to be.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONTINUUM_FILL: float = 1.0


def _window_mask(wave_centers: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(wave_centers.shape, dtype=bool)
    for lo, hi in windows:
        m |= (wave_centers >= lo) & (wave_centers <= hi)
    return m


VALID_FILL_MODES = ("constant", "noise", "gp_interp")


def _apply_mask(
    X: np.ndarray,
    mask: np.ndarray,
    fill: float = CONTINUUM_FILL,
    fill_mode: str = "constant",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace masked bins under one of three fill conventions.

    ``fill_mode="constant"`` (default): set every masked bin to ``fill``. This
    matches production behaviour.

    ``fill_mode="noise"``: set each masked bin to a Gaussian draw centered on
    ``fill`` with row-wise std equal to the off-mask continuum std of that
    same row. Preserves the heteroscedastic noise floor the classifier
    actually trained on instead of installing a perfectly flat patch.
    Requires ``rng``.

    ``fill_mode="gp_interp"``: linearly interpolate across each masked run
    using the two nearest unmasked bin values per row (np.interp). Boundary
    runs fall back to nearest-neighbor extrapolation.

    of the revision contract.
    """
    if fill_mode not in VALID_FILL_MODES:
        raise ValueError(f"unknown fill_mode {fill_mode!r}, must be one of {VALID_FILL_MODES}")
    X_out = X.copy()
    if not mask.any():
        return X_out
    if fill_mode == "constant":
        X_out[:, mask] = fill
        return X_out
    if fill_mode == "noise":
        if rng is None:
            raise ValueError("fill_mode='noise' requires a numpy Generator via rng=")
        off_mask = ~mask
        if not off_mask.any():
            X_out[:, mask] = fill
            return X_out
        sigma = np.std(X[:, off_mask], axis=1, ddof=1)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0.0).astype(np.float64)
        n_masked = int(mask.sum())
        noise = rng.standard_normal(size=(X.shape[0], n_masked)).astype(X.dtype)
        X_out[:, mask] = fill + noise * sigma[:, None].astype(X.dtype)
        return X_out
    # fill_mode == "gp_interp"
    n_bins = X.shape[1]
    idx_all = np.arange(n_bins)
    off_idx = idx_all[~mask]
    on_idx = idx_all[mask]
    if len(off_idx) < 2:
        X_out[:, mask] = fill
        return X_out
    for r in range(X.shape[0]):
        X_out[r, mask] = np.interp(on_idx, off_idx, X[r, ~mask]).astype(X.dtype)
    return X_out


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


@dataclass
class AblationRow:
    line_set: str
    mk_class: str
    n_test: int
    baseline_acc: float
    masked_acc_mean: float
    delta_acc_mean: float
    delta_acc_ci_low: float
    delta_acc_ci_high: float
    p_value_vs_random: float


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    mask = y_true == c
    if not mask.any():
        return np.nan
    return float(np.mean(y_pred[mask] == c))


def _sample_random_windows(
    rng: np.random.Generator,
    wave_centers: np.ndarray,
    total_width_aa: float,
    n_segments: int,
    forbidden_mask: np.ndarray,
    max_attempts_per_segment: int = 200,
) -> list[tuple[float, float]]:
    """Draw ``n_segments`` random windows whose total width matches ``total_width_aa``
    and avoid any bin flagged in ``forbidden_mask``.

    All-or-nothing: returns ``[]`` if the full ``n_segments`` cannot be drawn,
    so a partial sample never enters the null distribution with mismatched
    geometry (convention: fail-closed on geometry mismatch).

    ``forbidden_mask`` is the union of any bins that must be excluded from
    the null draw (typically all MK_LINES bins plus the UVES inter-chip gap
    bins per ).
    """
    seg_width = total_width_aa / n_segments
    wmin = float(wave_centers.min())
    wmax = float(wave_centers.max())
    picks: list[tuple[float, float]] = []
    for _ in range(n_segments * max_attempts_per_segment):
        lo = rng.uniform(wmin, wmax - seg_width)
        hi = lo + seg_width
        m = (wave_centers >= lo) & (wave_centers <= hi)
        if m.any() and not forbidden_mask[m].any():
            picks.append((lo, hi))
            if len(picks) == n_segments:
                return picks
    return []


def masked_line_ablation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wave_centers: np.ndarray,
    line_sets: dict[str, list[tuple[float, float]]],
    class_labels: list[str],
    per_class: bool = True,
    n_bootstrap: int = 500,
    n_random_controls: int = 100,
    seed: int = 42,
    gap_mask: np.ndarray | None = None,
    continuum_fill: float | None = None,
    draw_stats_out: dict[str, dict[str, int]] | None = None,
    per_row_out: dict[str, np.ndarray] | None = None,
    fill_mode: str = "constant",
) -> list[AblationRow]:
    """Bootstrap the accuracy drop from masking each ``line_set``.

    ``p_value_vs_random`` is the fraction of random-window controls whose
    delta is at least as negative as the observed line-set delta
    (one-sided: smaller p -> line set is more damaging than random).

    ``gap_mask``: optional boolean array of shape ``wave_centers.shape``. When
    provided, the forbidden mask passed to the random-window sampler becomes
    the union of all MK line bins AND the gap mask, so null draws cannot land
    in the UVES inter-chip gap.

    ``continuum_fill``: optional override for the value placed into masked
    bins. When ``None`` the module-level ``CONTINUUM_FILL`` (1.0) is used.

    ``draw_stats_out``: optional dict that, when supplied, will be populated
    with ``{set_name: {"requested": int, "succeeded": int}}`` so the caller
    can audit the random-draw success rate per line set.

    ``per_row_out``: optional dict that, when supplied, will be populated with
    flat keys for ``np.savez`` round-trip. Keys: ``y_test``, ``y_pred_base``,
    ``wave_centers``, ``y_pred_masked__<set_name>`` and ``line_mask__<set_name>``
    per line set. Downstream consumers (paired bootstrap, flip rate, TOST)
    reconstruct per-class subsets by indexing ``y_test`` against the class
    label.

    ``fill_mode``: one of ``"constant"`` (default), ``"noise"`` (per-row
    Gaussian draw with row-wise off-mask std), or ``"gp_interp"`` (linear
    interpolation across the masked window). sensitivity
    probe. The noise mode uses a separate seeded Generator (seed + 1000)
    so switching modes does not perturb the random-window or bootstrap
    draws and headline p-values remain comparable across modes.
    """
    fill_value = CONTINUUM_FILL if continuum_fill is None else float(continuum_fill)
    rng = np.random.default_rng(seed)
    rng_fill = np.random.default_rng(seed + 1000) if fill_mode == "noise" else None
    y_pred_base = model.predict(X_test)
    baseline_acc = float(np.mean(y_pred_base == y_test))
    rows: list[AblationRow] = []

    if per_row_out is not None:
        per_row_out["y_test"] = np.asarray(y_test).astype(np.int16, copy=False)
        per_row_out["y_pred_base"] = np.asarray(y_pred_base).astype(np.int16, copy=False)
        per_row_out["wave_centers"] = np.asarray(wave_centers, dtype=np.float32)
        per_row_out["continuum_fill"] = np.asarray(fill_value, dtype=np.float64)
        per_row_out["seed"] = np.asarray(seed, dtype=np.int64)
        per_row_out["fill_mode"] = np.asarray(fill_mode, dtype="<U16")

    all_line_bins = np.zeros(wave_centers.shape, dtype=bool)
    for windows in line_sets.values():
        all_line_bins |= _window_mask(wave_centers, windows)

    if gap_mask is not None:
        gap_mask = np.asarray(gap_mask, dtype=bool)
        if gap_mask.shape != wave_centers.shape:
            raise ValueError(
                f"gap_mask shape {gap_mask.shape} does not match "
                f"wave_centers shape {wave_centers.shape}"
            )
        forbidden_mask = all_line_bins | gap_mask
    else:
        forbidden_mask = all_line_bins

    logger.info(
        "ablation forbidden-mask: %d line bins, %d gap bins, %d total forbidden",
        int(all_line_bins.sum()),
        int(gap_mask.sum()) if gap_mask is not None else 0,
        int(forbidden_mask.sum()),
    )

    for set_name, windows in line_sets.items():
        line_mask = _window_mask(wave_centers, windows)
        if not line_mask.any():
            logger.warning("line set %s has no overlap with wave_centers - skipping", set_name)
            continue
        total_width = sum(hi - lo for lo, hi in windows)
        n_segments = len(windows)

        X_masked = _apply_mask(
            X_test, line_mask, fill=fill_value, fill_mode=fill_mode, rng=rng_fill,
        )
        y_pred_masked = model.predict(X_masked)

        if per_row_out is not None:
            per_row_out[f"y_pred_masked__{set_name}"] = np.asarray(
                y_pred_masked
            ).astype(np.int16, copy=False)
            per_row_out[f"line_mask__{set_name}"] = np.asarray(line_mask, dtype=bool)

        # Random-null distribution: draw n_random_controls non-line windows
        # with matched width/segment-count, outside any MK line window AND
        # outside the optional gap_mask.
        null_deltas: list[float] = []
        n_succeeded = 0
        for _ in range(n_random_controls):
            picks = _sample_random_windows(
                rng, wave_centers, total_width, n_segments, forbidden_mask
            )
            if not picks:
                continue
            m = _window_mask(wave_centers, picks)
            y_pred_null = model.predict(
                _apply_mask(
                    X_test, m, fill=fill_value, fill_mode=fill_mode, rng=rng_fill,
                )
            )
            null_deltas.append(
                float(np.mean(y_pred_null == y_test)) - baseline_acc
            )
            n_succeeded += 1
        null_deltas_arr = np.array(null_deltas, dtype=np.float32)
        success_rate = n_succeeded / n_random_controls if n_random_controls else 0.0
        if success_rate < 0.8:
            logger.warning(
                "line set %s: random-control success rate %.2f below 0.80 floor (%d/%d)",
                set_name, success_rate, n_succeeded, n_random_controls,
            )
        else:
            logger.info(
                "line set %s: random-control success rate %.2f (%d/%d)",
                set_name, success_rate, n_succeeded, n_random_controls,
            )
        if draw_stats_out is not None:
            draw_stats_out[set_name] = {
                "requested": int(n_random_controls),
                "succeeded": int(n_succeeded),
            }

        iter_classes: list[int] = (
            sorted(np.unique(y_test).tolist()) if per_class else [-1]
        )
        for c in iter_classes:
            if c == -1:
                base_metric = baseline_acc
                masked_metric = float(np.mean(y_pred_masked == y_test))
                label = "ALL"
                n = len(y_test)
            else:
                base_metric = _per_class_recall(y_test, y_pred_base, c)
                masked_metric = _per_class_recall(y_test, y_pred_masked, c)
                label = class_labels[c] if c < len(class_labels) else str(c)
                n = int(np.sum(y_test == c))
            delta = masked_metric - base_metric

            # Bootstrap CI on delta (resample test rows with replacement).
            boot_deltas = np.empty(n_bootstrap, dtype=np.float32)
            for b in range(n_bootstrap):
                idx = rng.integers(0, len(y_test), size=len(y_test))
                if c == -1:
                    bb = float(np.mean(y_pred_base[idx] == y_test[idx]))
                    bm = float(np.mean(y_pred_masked[idx] == y_test[idx]))
                else:
                    yi = y_test[idx]
                    bb = _per_class_recall(yi, y_pred_base[idx], c)
                    bm = _per_class_recall(yi, y_pred_masked[idx], c)
                boot_deltas[b] = (bm - bb) if np.isfinite(bm) and np.isfinite(bb) else np.nan
            boot_valid = boot_deltas[np.isfinite(boot_deltas)]
            if len(boot_valid) == 0:
                ci_lo = ci_hi = np.nan
            else:
                ci_lo = float(np.percentile(boot_valid, 2.5))
                ci_hi = float(np.percentile(boot_valid, 97.5))

            # Phipson and Smyth (2010), Stat Appl Genet Mol Biol 9:39:
            # p = (n_at_least_as_extreme + 1) / (n_random_controls + 1)
            # adds 1 to numerator and denominator so the floor at
            # n_random_controls=500 is 1/501 ~ 0.002 rather than 0/500=0
            # (the unbiased estimator for permutation p-values; the
            # uncorrected fraction is biased toward zero when the
            # observed value is more extreme than all permutations).
            n_succ = len(null_deltas_arr)
            p_val = (
                float((int(np.sum(null_deltas_arr <= delta)) + 1) / (n_succ + 1))
                if n_succ else np.nan
            )
            rows.append(AblationRow(
                line_set=set_name,
                mk_class=label,
                n_test=n,
                baseline_acc=float(base_metric),
                masked_acc_mean=float(masked_metric),
                delta_acc_mean=float(delta),
                delta_acc_ci_low=ci_lo,
                delta_acc_ci_high=ci_hi,
                p_value_vs_random=p_val,
            ))
    return rows
