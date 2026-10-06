# SPDX-License-Identifier: MIT
"""Masked-line ablation with a Monte-Carlo random-window reference set.

The ablation masks the bins of a named diagnostic line set, re-predicts the
test spectra, and reports the change in accuracy (or per-class recall)
relative to the unmasked baseline. Significance is assessed against a
reference set of random, non-line windows with matched geometry: the
reported p-value is the fraction of random windows whose delta is at least
as negative as the observed line-set delta.

Masking replaces flux inside a wavelength window with the continuum level
(1.0 after normalisation), not with zero, so no synthetic absorption
feature is injected where the line used to be. Alternative fills (row-wise
Gaussian noise around the continuum, or linear interpolation across the
window) are available for sensitivity checks.

Two choices control how the reference set is built and compared:

``null_mode``
    ``"class_matched"`` (default): for every random window the per-class
    recall delta is computed for each class, and each class's observed delta
    is compared against that class's own reference distribution. ``"pooled"``
    compares every per-class observed delta against the all-class accuracy
    reference distribution (the legacy convention).

``match_on``
    ``"bins"`` (default): random windows reproduce the line set's per-segment
    bin counts exactly; segments do not overlap each other or the forbidden
    mask. ``"angstrom"``: random windows reproduce the line set's total width
    in Angstrom split into equal-width segments (legacy convention; segments
    may overlap one another).

Paired statistics are reported alongside the Monte-Carlo p-value: the
number of test rows that flip from correct to incorrect and vice versa, an
exact McNemar test on those discordant pairs, the mean change in the
predicted probability of the true class, and a one-sided exact
Clopper-Pearson upper bound on the lost-flip rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONTINUUM_FILL: float = 1.0

VALID_FILL_MODES = ("constant", "noise", "gp_interp")
FILL_MODE_ALIASES: dict[str, str] = {"interp": "gp_interp"}
VALID_NULL_MODES = ("class_matched", "pooled")
VALID_MATCH_ON = ("bins", "angstrom")


def _canonical_fill_mode(fill_mode: str) -> str:
    """Resolve ``fill_mode`` aliases and validate against ``VALID_FILL_MODES``."""
    mode = FILL_MODE_ALIASES.get(fill_mode, fill_mode)
    if mode not in VALID_FILL_MODES:
        raise ValueError(
            f"unknown fill_mode {fill_mode!r}, must be one of "
            f"{VALID_FILL_MODES + tuple(FILL_MODE_ALIASES)}"
        )
    return mode


def _window_mask(wave_centers: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of bins whose centre lies inside any ``(lo, hi)`` window (inclusive)."""
    m = np.zeros(wave_centers.shape, dtype=bool)
    for lo, hi in windows:
        m |= (wave_centers >= lo) & (wave_centers <= hi)
    return m


def _segment_bin_counts(
    wave_centers: np.ndarray, windows: list[tuple[float, float]]
) -> list[int]:
    """Number of grid bins covered by each ``(lo, hi)`` window, in window order."""
    return [int(_window_mask(wave_centers, [(lo, hi)]).sum()) for lo, hi in windows]


def _apply_mask(
    X: np.ndarray,
    mask: np.ndarray,
    fill: float = CONTINUUM_FILL,
    fill_mode: str = "constant",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace masked bins under one of three fill conventions.

    ``fill_mode="constant"`` (default): set every masked bin to ``fill``. This
    is the production convention.

    ``fill_mode="noise"``: set each masked bin to a Gaussian draw centred on
    ``fill`` with row-wise standard deviation equal to the off-mask standard
    deviation of that same row. This preserves the heteroscedastic noise
    floor the classifier was trained on instead of installing a perfectly
    flat patch. Requires ``rng``.

    ``fill_mode="gp_interp"`` (alias ``"interp"``): plain linear interpolation
    across each masked run using the unmasked bin values of the same row
    (``np.interp``). Runs touching the grid edge fall back to the nearest
    unmasked value. The historical name is kept for backwards compatibility;
    no Gaussian-process model is involved.
    """
    fill_mode = _canonical_fill_mode(fill_mode)
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
    # fill_mode == "gp_interp": linear interpolation across the masked bins.
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


@dataclass
class AblationRow:
    """One (line set, class) result of :func:`masked_line_ablation`.

    The first nine fields are the legacy columns; the remaining fields carry
    the paired statistics and the configuration under which the row was
    computed. Defaults on the new fields keep positional construction of the
    nine legacy columns valid.

    ``p_value_vs_random`` is the Monte-Carlo p-value against the random-window
    reference set: ``(b + 1) / (B + 1)`` where ``b`` is the number of random
    windows whose delta is at least as negative as the observed delta and
    ``B`` is the number of successfully drawn windows.

    ``n_flip_lost`` / ``n_flip_gained`` count rows of the class that move from
    correct to incorrect / incorrect to correct under masking.
    ``mcnemar_exact_p`` is the two-sided exact binomial test on those
    discordant pairs (1.0 when there are none). ``mean_delta_true_prob`` is
    the mean over the class's rows of P(true class | masked) minus
    P(true class | baseline); NaN when the model has no ``predict_proba``.
    ``flip_rate_upper95`` is the one-sided exact Clopper-Pearson 95 percent
    upper bound on ``n_flip_lost / n_test``.
    """

    line_set: str
    mk_class: str
    n_test: int
    baseline_acc: float
    masked_acc_mean: float
    delta_acc_mean: float
    delta_acc_ci_low: float
    delta_acc_ci_high: float
    p_value_vs_random: float
    n_bins_masked: int = 0
    total_width_aa: float = float("nan")
    n_flip_lost: int = 0
    n_flip_gained: int = 0
    mcnemar_exact_p: float = float("nan")
    mean_delta_true_prob: float = float("nan")
    flip_rate_upper95: float = float("nan")
    n_random_controls_succeeded: int = 0
    null_mode: str = "class_matched"
    match_on: str = "bins"
    fill_mode: str = "constant"


LEGACY_ABLATION_FIELDS: tuple[str, ...] = (
    "line_set",
    "mk_class",
    "n_test",
    "baseline_acc",
    "masked_acc_mean",
    "delta_acc_mean",
    "delta_acc_ci_low",
    "delta_acc_ci_high",
    "p_value_vs_random",
)


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    mask = y_true == c
    if not mask.any():
        return np.nan
    return float(np.mean(y_pred[mask] == c))


def mcnemar_exact_p(n_lost: int, n_gained: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs.

    Under the null hypothesis that masking neither helps nor hurts, the
    ``n_lost`` correct-to-incorrect flips among ``n_lost + n_gained``
    discordant pairs follow Binomial(n, 0.5). Returns 1.0 when there are no
    discordant pairs.
    """
    from scipy.stats import binomtest

    n_disc = int(n_lost) + int(n_gained)
    if n_disc == 0:
        return 1.0
    return float(binomtest(int(n_lost), n_disc, p=0.5, alternative="two-sided").pvalue)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound on a proportion.

    Returns the ``confidence``-level upper bound on ``p`` given ``k``
    successes in ``n`` Bernoulli trials; NaN when ``n == 0`` and 1.0 when
    ``k == n``. For ``k == 0`` this reduces to ``1 - (1 - confidence)**(1/n)``.
    """
    from scipy.stats import beta

    k = int(k)
    n = int(n)
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def _sample_random_windows(
    rng: np.random.Generator,
    wave_centers: np.ndarray,
    total_width_aa: float,
    n_segments: int,
    forbidden_mask: np.ndarray,
    max_attempts_per_segment: int = 200,
) -> list[tuple[float, float]]:
    """Draw ``n_segments`` equal-width random windows totalling ``total_width_aa``.

    Width is matched in Angstrom (``match_on="angstrom"``). Each segment must
    contain at least one bin and must not touch any bin flagged in
    ``forbidden_mask``; segments are not prevented from overlapping each
    other.

    All-or-nothing: returns ``[]`` if the full ``n_segments`` cannot be drawn,
    so a partial sample never enters the reference set with mismatched
    geometry (fail-closed on geometry mismatch).

    ``forbidden_mask`` is the union of every bin that must be excluded from
    the random draw: all MK line-set bins plus the UVES inter-chip gap bins
    when a gap mask is supplied.
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


def _sample_random_windows_bins(
    rng: np.random.Generator,
    wave_centers: np.ndarray,
    segment_bin_counts: list[int],
    forbidden_mask: np.ndarray,
    max_attempts_per_segment: int = 200,
) -> list[tuple[float, float]]:
    """Draw random windows covering exactly ``segment_bin_counts[i]`` bins each.

    Width is matched in grid bins (``match_on="bins"``). Each segment is a
    run of contiguous bins chosen uniformly among all start positions; it
    must avoid every bin flagged in ``forbidden_mask`` and every bin already
    claimed by a previously drawn segment of the same draw, so the returned
    segments are mutually non-overlapping. Segments with a zero bin count are
    skipped. Returned windows are ``(wave_centers[start], wave_centers[end])``
    so that :func:`_window_mask` recovers exactly the drawn bins.

    All-or-nothing: returns ``[]`` if any segment cannot be placed.
    """
    n_bins = len(wave_centers)
    claimed = forbidden_mask.copy()
    picks: list[tuple[float, float]] = []
    for k in segment_bin_counts:
        k = int(k)
        if k <= 0:
            continue
        if k > n_bins:
            return []
        placed = False
        for _ in range(max_attempts_per_segment):
            start = int(rng.integers(0, n_bins - k + 1))
            stop = start + k
            if claimed[start:stop].any():
                continue
            claimed[start:stop] = True
            picks.append((float(wave_centers[start]), float(wave_centers[stop - 1])))
            placed = True
            break
        if not placed:
            return []
    return picks


def _proba_columns(model: Any, classes: list[int]) -> dict[int, int] | None:
    """Map integer class labels to ``predict_proba`` column indices, or None."""
    if not hasattr(model, "predict_proba"):
        return None
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        model_classes = np.asarray(sorted(classes))
    model_classes = np.asarray(model_classes)
    lookup: dict[int, int] = {}
    for c in classes:
        hit = np.flatnonzero(model_classes == c)
        if len(hit) == 0:
            return None
        lookup[int(c)] = int(hit[0])
    return lookup


def masked_line_ablation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wave_centers: np.ndarray,
    line_sets: dict[str, list[tuple[float, float]]],
    class_labels: list[str],
    per_class: bool = True,
    n_bootstrap: int = 2000,
    n_random_controls: int = 5000,
    seed: int = 42,
    gap_mask: np.ndarray | None = None,
    continuum_fill: float | None = None,
    draw_stats_out: dict[str, dict[str, Any]] | None = None,
    per_row_out: dict[str, np.ndarray] | None = None,
    fill_mode: str = "constant",
    null_mode: str = "class_matched",
    match_on: str = "bins",
) -> list[AblationRow]:
    """Mask each ``line_set`` and measure the accuracy drop against a random-window reference set.

    For every line set the function masks the line bins, re-predicts
    ``X_test`` and reports, per class (and for all classes pooled when
    ``per_class`` is False), the observed delta in recall/accuracy, a
    percentile bootstrap CI on that delta (test rows resampled with
    replacement), a Monte-Carlo p-value against random windows, and paired
    flip statistics.

    p-value convention
        ``n_random_controls`` random, non-line windows with geometry matched
        to the line set are drawn and the same masking is applied. The
        p-value is the fraction of random windows whose delta is at least as
        negative as the observed line-set delta (one-sided: a small p means
        the line set is more damaging than a typical random window of the
        same size). It is computed as ``(b + 1) / (B + 1)`` with ``b`` the
        count of random windows at least as negative and ``B`` the number of
        successful draws. This is the valid, slightly conservative
        Monte-Carlo p-value of Phipson and Smyth (2010, Stat Appl Genet Mol
        Biol 9:39); the uncorrected ``b / B`` is the unbiased estimator of
        the same quantity but can be exactly zero. The random windows form a
        Monte-Carlo reference set, not a permutation distribution: they
        sample the distribution of deltas under "a window of this size placed
        at random outside the known lines", which is the comparison the
        headline question asks for.

    ``null_mode``
        ``"class_matched"`` (default): the per-class recall delta is computed
        for each random window and each class's observed delta is compared
        against its own class's reference deltas. ``"pooled"``: every class's
        observed delta is compared against the all-class accuracy reference
        deltas (legacy convention). The ``ALL`` row uses the accuracy
        reference in both modes.

    ``match_on``
        ``"bins"`` (default): random windows reproduce the line set's
        per-segment bin counts exactly and are mutually non-overlapping.
        ``"angstrom"``: random windows reproduce the line set's total width in
        Angstrom as equal-width segments (legacy convention).

    ``gap_mask``: optional boolean array of shape ``wave_centers.shape``. When
    provided, the forbidden mask passed to the random-window sampler becomes
    the union of all MK line bins and the gap mask, so random draws cannot
    land in the UVES inter-chip gap.

    ``continuum_fill``: optional override for the value placed into masked
    bins. When ``None`` the module-level ``CONTINUUM_FILL`` (1.0) is used.

    ``fill_mode``: one of ``"constant"`` (default), ``"noise"`` (row-wise
    Gaussian draw with row-wise off-mask std) or ``"gp_interp"`` / ``"interp"``
    (linear interpolation across the masked window). The noise mode uses a
    separate seeded Generator (``seed + 1000``) so switching modes does not
    perturb the random-window or bootstrap draws.

    ``draw_stats_out``: optional dict populated per line set with
    ``requested``, ``succeeded``, ``match_on``, ``target_bin_counts``,
    ``target_total_width_aa``, ``n_draws_bin_matched`` (draws whose realised
    per-segment bin counts equal the target) and
    ``realised_total_bins_histogram`` (``{total_bins: n_draws}``).

    ``per_row_out``: optional dict populated with flat keys for an
    ``np.savez`` round-trip: ``y_test``, ``y_pred_base``, ``wave_centers``,
    ``continuum_fill``, ``seed``, ``fill_mode``, ``null_mode``, ``match_on``,
    ``proba_classes`` and ``proba_base`` (when the model exposes
    ``predict_proba``), and per line set ``y_pred_masked__<set>``,
    ``line_mask__<set>`` and ``proba_masked__<set>``.

    Random-number consumption order (window draws for a line set, then the
    bootstrap for each class in sorted order) is identical across modes, so
    ``null_mode="pooled", match_on="angstrom"`` reproduces the legacy
    implementation exactly for the same ``seed`` and draw counts.
    """
    if null_mode not in VALID_NULL_MODES:
        raise ValueError(f"unknown null_mode {null_mode!r}, must be one of {VALID_NULL_MODES}")
    if match_on not in VALID_MATCH_ON:
        raise ValueError(f"unknown match_on {match_on!r}, must be one of {VALID_MATCH_ON}")
    fill_mode = _canonical_fill_mode(fill_mode)

    fill_value = CONTINUUM_FILL if continuum_fill is None else float(continuum_fill)
    rng = np.random.default_rng(seed)
    rng_fill = np.random.default_rng(seed + 1000) if fill_mode == "noise" else None
    y_test = np.asarray(y_test)
    y_pred_base = np.asarray(model.predict(X_test))
    baseline_acc = float(np.mean(y_pred_base == y_test))
    base_correct = y_pred_base == y_test
    rows: list[AblationRow] = []

    classes_present: list[int] = sorted(int(c) for c in np.unique(y_test).tolist())
    proba_cols = _proba_columns(model, classes_present)
    proba_base: np.ndarray | None = None
    if proba_cols is not None:
        proba_base = np.asarray(model.predict_proba(X_test), dtype=np.float64)
        if proba_base.ndim != 2 or proba_base.shape[0] != len(y_test):
            proba_cols = None
            proba_base = None

    def _true_prob(P: np.ndarray) -> np.ndarray:
        cols = np.array([proba_cols[int(c)] for c in y_test])  # type: ignore[index]
        return P[np.arange(len(y_test)), cols]

    true_prob_base = _true_prob(proba_base) if proba_base is not None else None

    if per_row_out is not None:
        per_row_out["y_test"] = y_test.astype(np.int16, copy=False)
        per_row_out["y_pred_base"] = y_pred_base.astype(np.int16, copy=False)
        per_row_out["wave_centers"] = np.asarray(wave_centers, dtype=np.float32)
        per_row_out["continuum_fill"] = np.asarray(fill_value, dtype=np.float64)
        per_row_out["seed"] = np.asarray(seed, dtype=np.int64)
        per_row_out["fill_mode"] = np.asarray(fill_mode, dtype="<U16")
        per_row_out["null_mode"] = np.asarray(null_mode, dtype="<U16")
        per_row_out["match_on"] = np.asarray(match_on, dtype="<U16")
        if proba_base is not None:
            per_row_out["proba_base"] = proba_base.astype(np.float32)
            model_classes = getattr(model, "classes_", None)
            if model_classes is None or len(model_classes) != proba_base.shape[1]:
                model_classes = [
                    c for c, _ in sorted(proba_cols.items(), key=lambda kv: kv[1])  # type: ignore[union-attr]
                ]
            per_row_out["proba_classes"] = np.asarray(model_classes, dtype=np.int16)

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
        "ablation forbidden-mask: %d line bins, %d gap bins, %d total forbidden "
        "(null_mode=%s, match_on=%s, fill_mode=%s)",
        int(all_line_bins.sum()),
        int(gap_mask.sum()) if gap_mask is not None else 0,
        int(forbidden_mask.sum()),
        null_mode, match_on, fill_mode,
    )

    for set_name, windows in line_sets.items():
        line_mask = _window_mask(wave_centers, windows)
        if not line_mask.any():
            logger.warning("line set %s has no overlap with wave_centers - skipping", set_name)
            continue
        total_width = float(sum(hi - lo for lo, hi in windows))
        n_segments = len(windows)
        target_bin_counts = _segment_bin_counts(wave_centers, windows)
        n_bins_masked = int(line_mask.sum())

        X_masked = _apply_mask(
            X_test, line_mask, fill=fill_value, fill_mode=fill_mode, rng=rng_fill,
        )
        y_pred_masked = np.asarray(model.predict(X_masked))
        masked_correct = y_pred_masked == y_test
        true_prob_masked: np.ndarray | None = None
        if proba_base is not None:
            P_masked = np.asarray(model.predict_proba(X_masked), dtype=np.float64)
            true_prob_masked = _true_prob(P_masked)
            if per_row_out is not None:
                per_row_out[f"proba_masked__{set_name}"] = P_masked.astype(np.float32)

        if per_row_out is not None:
            per_row_out[f"y_pred_masked__{set_name}"] = y_pred_masked.astype(np.int16, copy=False)
            per_row_out[f"line_mask__{set_name}"] = np.asarray(line_mask, dtype=bool)

        # Monte-Carlo reference set: n_random_controls non-line windows with
        # matched geometry, outside every MK line window and outside the
        # optional gap mask. Predictions are stored so that both the pooled
        # (accuracy) and the class-matched (per-class recall) reference
        # deltas can be derived from the same draws.
        null_preds: list[np.ndarray] = []
        n_succeeded = 0
        n_bin_matched = 0
        realised_hist: dict[int, int] = {}
        for _ in range(n_random_controls):
            if match_on == "bins":
                picks = _sample_random_windows_bins(
                    rng, wave_centers, target_bin_counts, forbidden_mask
                )
            else:
                picks = _sample_random_windows(
                    rng, wave_centers, total_width, n_segments, forbidden_mask
                )
            if not picks:
                continue
            m = _window_mask(wave_centers, picks)
            realised = _segment_bin_counts(wave_centers, picks)
            realised_total = int(m.sum())
            realised_hist[realised_total] = realised_hist.get(realised_total, 0) + 1
            if realised == [k for k in target_bin_counts if k > 0]:
                n_bin_matched += 1
            y_pred_null = np.asarray(model.predict(
                _apply_mask(
                    X_test, m, fill=fill_value, fill_mode=fill_mode, rng=rng_fill,
                )
            ))
            null_preds.append(y_pred_null.astype(np.int16, copy=False))
            n_succeeded += 1
        null_pred_arr = (
            np.stack(null_preds, axis=0) if null_preds
            else np.empty((0, len(y_test)), dtype=np.int16)
        )
        null_correct = null_pred_arr == y_test[None, :]
        # All-class accuracy reference deltas (float32 as in the legacy code).
        null_acc_deltas = (
            null_correct.mean(axis=1).astype(np.float64) - baseline_acc
        ).astype(np.float32)

        success_rate = n_succeeded / n_random_controls if n_random_controls else 0.0
        if success_rate < 0.8:
            logger.warning(
                "line set %s: random-control success rate %.2f below 0.80 floor (%d/%d)",
                set_name, success_rate, n_succeeded, n_random_controls,
            )
        else:
            logger.info(
                "line set %s: random-control success rate %.2f (%d/%d); "
                "%d/%d draws reproduce the target bin counts %s",
                set_name, success_rate, n_succeeded, n_random_controls,
                n_bin_matched, n_succeeded, target_bin_counts,
            )
        if draw_stats_out is not None:
            draw_stats_out[set_name] = {
                "requested": int(n_random_controls),
                "succeeded": int(n_succeeded),
                "match_on": match_on,
                "target_bin_counts": [int(k) for k in target_bin_counts],
                "target_total_width_aa": float(total_width),
                "n_draws_bin_matched": int(n_bin_matched),
                "realised_total_bins_histogram": {
                    str(k): int(v) for k, v in sorted(realised_hist.items())
                },
            }

        iter_classes: list[int] = classes_present if per_class else [-1]
        for c in iter_classes:
            if c == -1:
                base_metric = baseline_acc
                masked_metric = float(np.mean(masked_correct))
                label = "ALL"
                cls_rows = np.ones(len(y_test), dtype=bool)
                null_deltas_arr = null_acc_deltas
            else:
                base_metric = _per_class_recall(y_test, y_pred_base, c)
                masked_metric = _per_class_recall(y_test, y_pred_masked, c)
                label = class_labels[c] if c < len(class_labels) else str(c)
                cls_rows = y_test == c
                if null_mode == "class_matched" and len(null_pred_arr):
                    null_recall = (null_pred_arr[:, cls_rows] == c).mean(axis=1)
                    null_deltas_arr = (
                        null_recall.astype(np.float64) - base_metric
                    ).astype(np.float32)
                else:
                    null_deltas_arr = null_acc_deltas
            n = int(cls_rows.sum())
            delta = masked_metric - base_metric

            # Percentile bootstrap CI on delta (test rows resampled with
            # replacement; per-class recall recomputed on each resample).
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

            # Monte-Carlo p-value, Phipson and Smyth (2010): (b + 1) / (B + 1)
            # with b = number of random windows whose delta is at least as
            # negative as the observed delta and B = successful draws. The
            # +1 terms make the estimator valid (never exactly zero; floor
            # 1/(B+1)) at the cost of a slight conservative bias; b / B would
            # be the unbiased estimator of the same tail probability.
            n_succ = len(null_deltas_arr)
            p_val = (
                float((int(np.sum(null_deltas_arr <= delta)) + 1) / (n_succ + 1))
                if n_succ else np.nan
            )

            # Paired flip statistics on the class's rows.
            n_lost = int(np.sum(base_correct[cls_rows] & ~masked_correct[cls_rows]))
            n_gained = int(np.sum(~base_correct[cls_rows] & masked_correct[cls_rows]))
            p_mcnemar = mcnemar_exact_p(n_lost, n_gained)
            upper95 = clopper_pearson_upper(n_lost, n)
            if true_prob_base is not None and true_prob_masked is not None and n > 0:
                mean_dp = float(np.mean(
                    true_prob_masked[cls_rows] - true_prob_base[cls_rows]
                ))
            else:
                mean_dp = float("nan")

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
                n_bins_masked=n_bins_masked,
                total_width_aa=total_width,
                n_flip_lost=n_lost,
                n_flip_gained=n_gained,
                mcnemar_exact_p=p_mcnemar,
                mean_delta_true_prob=mean_dp,
                flip_rate_upper95=upper95,
                n_random_controls_succeeded=int(n_succeeded),
                null_mode=null_mode,
                match_on=match_on,
                fill_mode=fill_mode,
            ))
    return rows
