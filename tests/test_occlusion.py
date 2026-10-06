"""Unit tests for the masked-line ablation as re-exported by src.interpret.occlusion.

Covers:
  - random-control sampler honours an extended forbidden mask that includes
    the UVES inter-chip gap plus all MK_LINES bins
  - masked_line_ablation forwards the gap-augmented forbidden mask to the
    sampler (gap_mask propagation contract) in both matching modes
  - AblationRow keeps the nine legacy fields first, followed by the paired
    statistics and configuration fields
  - random-window picks fail closed when no draw is feasible
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from src.interpret import ablation as abl
from src.interpret import occlusion as occ
from src.interpret.ablation import LEGACY_ABLATION_FIELDS
from src.interpret.lines import LINE_SETS
from src.interpret.occlusion import (
    AblationRow,
    _sample_random_windows,
    masked_line_ablation,
)


def _toy_wave_centers(n: int = 200, lo: float = 4800.0, hi: float = 6800.0) -> np.ndarray:
    return np.linspace(lo, hi, n).astype(np.float32)


def test_random_controls_avoid_gap_mask() -> None:
    """Synthetic wave_centers + gap band + lines: 100 draws, zero overlap with gap."""
    wc = _toy_wave_centers(n=400)
    # Gap covers 5769-5834 A like the UVES inter-chip gap.
    gap_mask = (wc >= 5769.0) & (wc <= 5834.0)
    # Pretend lines occupy a separate band so the union is non-trivial.
    line_mask = (wc >= 5167.0) & (wc <= 5184.0)
    forbidden = gap_mask | line_mask

    rng = np.random.default_rng(0)
    n_draws = 100
    n_overlapping = 0
    n_succeeded = 0
    for _ in range(n_draws):
        picks = _sample_random_windows(
            rng,
            wave_centers=wc,
            total_width_aa=20.0,
            n_segments=1,
            forbidden_mask=forbidden,
        )
        if not picks:
            continue
        n_succeeded += 1
        for lo, hi in picks:
            window_mask = (wc >= lo) & (wc <= hi)
            if (window_mask & gap_mask).any():
                n_overlapping += 1
    assert n_succeeded >= 80, f"only {n_succeeded}/{n_draws} draws succeeded"
    assert n_overlapping == 0, (
        f"{n_overlapping} of {n_succeeded} successful draws overlapped gap_mask"
    )


def test_masked_line_ablation_passes_gap_mask_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatch the sampler and confirm the union forbidden mask is passed."""
    wc = _toy_wave_centers(n=200)
    gap_mask = (wc >= 5769.0) & (wc <= 5834.0)
    rng = np.random.default_rng(7)
    n_rows = 60
    X = rng.normal(loc=1.0, scale=0.05, size=(n_rows, len(wc))).astype(np.float32)
    y = rng.integers(low=1, high=4, size=n_rows).astype(np.int64)

    class _StubModel:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            return rng.integers(low=1, high=4, size=X_in.shape[0]).astype(np.int64)

    captured: list[np.ndarray] = []

    def _fake_sampler(
        rng: np.random.Generator,
        wave_centers: np.ndarray,
        total_width_aa: float,
        n_segments: int,
        forbidden_mask: np.ndarray,
        max_attempts_per_segment: int = 200,
    ) -> list[tuple[float, float]]:
        captured.append(forbidden_mask.copy())
        return []

    def _fake_bins_sampler(
        rng: np.random.Generator,
        wave_centers: np.ndarray,
        segment_bin_counts: list[int],
        forbidden_mask: np.ndarray,
        max_attempts_per_segment: int = 200,
    ) -> list[tuple[float, float]]:
        captured.append(forbidden_mask.copy())
        return []

    # The ablation module looks the samplers up in its own namespace, so the
    # patch targets src.interpret.ablation (occlusion only re-exports).
    monkeypatch.setattr(abl, "_sample_random_windows", _fake_sampler)
    monkeypatch.setattr(abl, "_sample_random_windows_bins", _fake_bins_sampler)
    assert occ.masked_line_ablation is abl.masked_line_ablation

    for match_on in ("angstrom", "bins"):
        masked_line_ablation(
            _StubModel(),
            X_test=X,
            y_test=y,
            wave_centers=wc,
            line_sets=LINE_SETS,
            class_labels=["A", "F", "G", "K"],
            per_class=True,
            n_bootstrap=5,
            n_random_controls=3,
            seed=42,
            gap_mask=gap_mask,
            match_on=match_on,
        )
    assert captured, "sampler was not invoked"
    expected_lines = np.zeros(wc.shape, dtype=bool)
    for windows in LINE_SETS.values():
        for lo, hi in windows:
            expected_lines |= (wc >= lo) & (wc <= hi)
    expected_forbidden = expected_lines | gap_mask
    for forbidden in captured:
        assert forbidden.shape == wc.shape
        # Every gap_mask bin must be forbidden.
        assert bool((gap_mask & ~forbidden).any()) is False, (
            "gap_mask bin leaked into permitted region"
        )
        # And every MK_LINES bin must be forbidden.
        assert bool((expected_lines & ~forbidden).any()) is False
        # And the union must be exactly the captured forbidden mask.
        assert np.array_equal(forbidden, expected_forbidden)


def test_ablation_row_columns() -> None:
    """AblationRow keeps the nine legacy fields first, then the paired-statistics fields."""
    expected_legacy = (
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
    expected_new = (
        "n_bins_masked",
        "total_width_aa",
        "n_flip_lost",
        "n_flip_gained",
        "mcnemar_exact_p",
        "mean_delta_true_prob",
        "flip_rate_upper95",
        "n_random_controls_succeeded",
        "null_mode",
        "match_on",
        "fill_mode",
    )
    actual = tuple(f.name for f in fields(AblationRow))
    assert actual[:9] == expected_legacy == LEGACY_ABLATION_FIELDS
    assert actual[9:] == expected_new
    assert len(actual) == 20
    # Legacy nine-positional construction still works.
    row = AblationRow("Mg_b", "K", 145, 0.9, 0.9, 0.0, -0.01, 0.01, 0.5)
    assert row.null_mode == "class_matched" and row.match_on == "bins"


def test_random_controls_fail_closed_when_forbidden_mask_full() -> None:
    """If the forbidden mask covers everything, the sampler returns []."""
    wc = _toy_wave_centers(n=100)
    forbidden = np.ones(wc.shape, dtype=bool)
    rng = np.random.default_rng(0)
    picks = _sample_random_windows(
        rng,
        wave_centers=wc,
        total_width_aa=20.0,
        n_segments=2,
        forbidden_mask=forbidden,
    )
    assert picks == []


def test_masked_line_ablation_records_draw_stats() -> None:
    """draw_stats_out is filled in with requested/succeeded counts per line set."""
    wc = _toy_wave_centers(n=300)
    rng = np.random.default_rng(11)
    n_rows = 80
    X = rng.normal(loc=1.0, scale=0.05, size=(n_rows, len(wc))).astype(np.float32)
    y = rng.integers(low=1, high=4, size=n_rows).astype(np.int64)

    class _StubModel:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            # Constant prediction so deltas are deterministic.
            return np.full(X_in.shape[0], 2, dtype=np.int64)

    stats: dict[str, dict[str, int]] = {}
    rows = masked_line_ablation(
        _StubModel(),
        X_test=X,
        y_test=y,
        wave_centers=wc,
        line_sets={"H_balmer": LINE_SETS["H_balmer"]},
        class_labels=["A", "F", "G", "K"],
        per_class=True,
        n_bootstrap=5,
        n_random_controls=10,
        seed=42,
        gap_mask=None,
        draw_stats_out=stats,
    )
    assert "H_balmer" in stats
    assert stats["H_balmer"]["requested"] == 10
    assert 0 <= stats["H_balmer"]["succeeded"] <= 10
    assert rows  # produced at least one AblationRow


def test_masked_line_ablation_persists_per_row_predictions(tmp_path) -> None:
    """per_row_out keys round-trip through np.savez/np.load.

    Downstream consumers (paired bootstrap, flip rate, TOST) read this
    artefact to recompute per-class statistics without re-running the
    ablation.
    """
    wc = _toy_wave_centers(n=300)
    rng = np.random.default_rng(11)
    n_rows = 80
    X = rng.normal(loc=1.0, scale=0.05, size=(n_rows, len(wc))).astype(np.float32)
    y = rng.integers(low=1, high=4, size=n_rows).astype(np.int64)

    class _DeterministicModel:
        """Always predicts class 2, except when the Mg b region is masked.

        When all Mg b bins are at the fill value (1.0), predict class 3
        instead. This lets the test assert that y_pred_masked differs
        from y_pred_base in a known, deterministic way.
        """

        def predict(self, X_in: np.ndarray) -> np.ndarray:
            mg_lo, mg_hi = 5167.0, 5184.0
            bins = (wc >= mg_lo) & (wc <= mg_hi)
            # If every Mg b bin is at fill value 1.0 on every row, it's masked.
            mg_all_filled = bool(np.all(np.isclose(X_in[:, bins], 1.0)))
            cls = 3 if mg_all_filled else 2
            return np.full(X_in.shape[0], cls, dtype=np.int64)

    per_row: dict[str, np.ndarray] = {}
    rows = masked_line_ablation(
        _DeterministicModel(),
        X_test=X,
        y_test=y,
        wave_centers=wc,
        line_sets={"Mg_b": LINE_SETS["Mg_b"], "H_balmer": LINE_SETS["H_balmer"]},
        class_labels=["A", "F", "G", "K"],
        per_class=True,
        n_bootstrap=5,
        n_random_controls=5,
        seed=42,
        gap_mask=None,
        per_row_out=per_row,
    )
    # Top-level keys.
    assert per_row["y_test"].shape == (n_rows,)
    assert per_row["y_pred_base"].shape == (n_rows,)
    assert per_row["wave_centers"].shape == wc.shape
    assert float(per_row["continuum_fill"]) == 1.0
    assert int(per_row["seed"]) == 42
    # Per-line-set keys.
    for set_name in ("Mg_b", "H_balmer"):
        assert f"y_pred_masked__{set_name}" in per_row
        assert f"line_mask__{set_name}" in per_row
        assert per_row[f"y_pred_masked__{set_name}"].shape == (n_rows,)
        assert per_row[f"line_mask__{set_name}"].dtype == np.bool_
        assert per_row[f"line_mask__{set_name}"].shape == wc.shape
    # Deterministic check: under Mg b masking, all predictions flip to 3.
    assert np.all(per_row["y_pred_masked__Mg_b"] == 3)
    assert np.all(per_row["y_pred_base"] == 2)
    # And under H_balmer masking, Mg b is not filled, so predictions stay at 2.
    assert np.all(per_row["y_pred_masked__H_balmer"] == 2)

    # np.savez round-trip.
    path = tmp_path / "per_row.npz"
    np.savez(path, **per_row)
    loaded = np.load(path)
    assert sorted(loaded.files) == sorted(per_row.keys())
    np.testing.assert_array_equal(
        loaded["y_pred_masked__Mg_b"], per_row["y_pred_masked__Mg_b"]
    )
    assert rows  # aggregate rows still produced


def test_apply_mask_constant() -> None:
    """fill_mode='constant' replaces every masked bin with the fill value."""
    from src.interpret.occlusion import _apply_mask

    X = np.arange(20, dtype=np.float32).reshape(2, 10)
    mask = np.zeros(10, dtype=bool)
    mask[3:6] = True
    out = _apply_mask(X, mask, fill=0.916, fill_mode="constant")
    assert np.allclose(out[:, mask], 0.916)
    # off-mask bins are unchanged
    np.testing.assert_array_equal(out[:, ~mask], X[:, ~mask])


def test_apply_mask_noise_row_wise_sigma() -> None:
    """fill_mode='noise' draws Gaussian per row with row-wise off-mask sigma.

    Constructs two rows with deliberately different off-mask noise scales and
    asserts that the variance of the injected noise is approximately equal to
    the off-mask variance per row, not to a global value.
    """
    from src.interpret.occlusion import _apply_mask

    rng_data = np.random.default_rng(0)
    n_bins = 800
    mask = np.zeros(n_bins, dtype=bool)
    mask[100:200] = True
    X = np.empty((2, n_bins), dtype=np.float32)
    X[0] = 1.0 + rng_data.normal(0.0, 0.02, n_bins)
    X[1] = 1.0 + rng_data.normal(0.0, 0.10, n_bins)
    rng_fill = np.random.default_rng(42)
    out = _apply_mask(X, mask, fill=1.0, fill_mode="noise", rng=rng_fill)
    # off-mask sigmas
    sigma_off = np.std(X[:, ~mask], axis=1, ddof=1)
    # masked-region std should track sigma_off per row, within ~25% (100 bins,
    # finite-sample fluctuation, plus mode is mean-centered at 1.0).
    sigma_filled = np.std(out[:, mask], axis=1, ddof=1)
    assert np.abs(sigma_filled[0] - sigma_off[0]) / sigma_off[0] < 0.25
    assert np.abs(sigma_filled[1] - sigma_off[1]) / sigma_off[1] < 0.25
    # And the two rows must have substantially different masked-region sigmas
    # (the design point: noise mode is row-wise, not global).
    assert sigma_filled[1] / sigma_filled[0] > 2.0


def test_apply_mask_gp_interp_linear() -> None:
    """fill_mode='gp_interp' (alias 'interp') linearly interpolates across the masked window.

    On a monotonically rising row (X[r, j] = j), interpolation across a
    masked interior run exactly reconstructs the original.
    """
    from src.interpret.occlusion import _apply_mask

    X = np.tile(np.arange(20, dtype=np.float32), (3, 1))
    mask = np.zeros(20, dtype=bool)
    mask[7:13] = True
    out = _apply_mask(X, mask, fill=999.0, fill_mode="gp_interp")
    # Linear interp on a linear row reconstructs the original exactly.
    np.testing.assert_allclose(out[:, mask], X[:, mask], atol=1e-5)
    out_alias = _apply_mask(X, mask, fill=999.0, fill_mode="interp")
    np.testing.assert_array_equal(out_alias, out)


def test_apply_mask_noise_requires_rng() -> None:
    """noise mode without rng raises."""
    from src.interpret.occlusion import _apply_mask

    X = np.ones((2, 10), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    mask[3] = True
    with pytest.raises(ValueError):
        _apply_mask(X, mask, fill_mode="noise", rng=None)


def test_apply_mask_unknown_mode() -> None:
    from src.interpret.occlusion import _apply_mask

    X = np.ones((2, 10), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError):
        _apply_mask(X, mask, fill_mode="bogus")


def test_masked_line_ablation_fill_mode_propagates() -> None:
    """When fill_mode='noise', different invocations with different fill
    seeds produce different y_pred_masked but the same y_pred_base.

    Also asserts that switching fill_mode from 'constant' to 'noise' does not
    change the random-window draw pattern (the rng_windows is independent of
    rng_fill by construction).
    """
    wc = _toy_wave_centers(n=200)
    rng_data = np.random.default_rng(7)
    n_rows = 60
    X = (1.0 + rng_data.normal(0.0, 0.05, size=(n_rows, len(wc)))).astype(np.float32)
    y = rng_data.integers(low=1, high=4, size=n_rows).astype(np.int64)

    class _ThresholdModel:
        """Predicts class 2 if the Mg b region mean exceeds 0.95, else class 3."""

        def predict(self, X_in: np.ndarray) -> np.ndarray:
            bins = (wc >= 5167.0) & (wc <= 5184.0)
            m = X_in[:, bins].mean(axis=1)
            return np.where(m > 0.95, 2, 3).astype(np.int64)

    per_row_const: dict[str, np.ndarray] = {}
    masked_line_ablation(
        _ThresholdModel(), X_test=X, y_test=y, wave_centers=wc,
        line_sets={"Mg_b": LINE_SETS["Mg_b"]},
        class_labels=["A", "F", "G", "K"],
        per_class=True, n_bootstrap=3, n_random_controls=3,
        seed=42, gap_mask=None,
        per_row_out=per_row_const, fill_mode="constant",
    )
    per_row_noise: dict[str, np.ndarray] = {}
    masked_line_ablation(
        _ThresholdModel(), X_test=X, y_test=y, wave_centers=wc,
        line_sets={"Mg_b": LINE_SETS["Mg_b"]},
        class_labels=["A", "F", "G", "K"],
        per_class=True, n_bootstrap=3, n_random_controls=3,
        seed=42, gap_mask=None,
        per_row_out=per_row_noise, fill_mode="noise",
    )
    # y_pred_base is computed before any masking, so it is identical across modes.
    np.testing.assert_array_equal(per_row_const["y_pred_base"], per_row_noise["y_pred_base"])
    # fill_mode is recorded.
    assert str(per_row_const["fill_mode"]) == "constant"
    assert str(per_row_noise["fill_mode"]) == "noise"


def test_p_value_phipson_smyth_floor() -> None:
    """p-value uses (B+1)-corrected denominator per Phipson and Smyth (2010).

    When every null draw produces a less-extreme delta than the observed value
    (so 0 of B random controls satisfy null_delta <= observed_delta), the raw
    fraction would be 0/B. The (B+1)-corrected estimator is 1/(B+1), i.e.
    1/501 ~ 0.001996 at B=500.

    We construct a stub model that predicts a constant class regardless of
    masking, so the line-set delta is 0.0; we then check that with 4 random
    controls whose deltas are deliberately positive, the observed p-value is
    (0 + 1) / (4 + 1) = 0.2 (the Phipson-Smyth floor for B=4).
    """
    wc = _toy_wave_centers(n=200)
    rng = np.random.default_rng(3)
    n_rows = 60
    X = rng.normal(loc=1.0, scale=0.05, size=(n_rows, len(wc))).astype(np.float32)
    y = np.full(n_rows, 2, dtype=np.int64)  # all class 2

    class _ConstantModel:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            return np.full(X_in.shape[0], 2, dtype=np.int64)

    rows = masked_line_ablation(
        _ConstantModel(),
        X_test=X,
        y_test=y,
        wave_centers=wc,
        line_sets={"H_balmer": LINE_SETS["H_balmer"]},
        class_labels=["A", "F", "G", "K"],
        per_class=True,
        n_bootstrap=3,
        n_random_controls=4,
        seed=42,
        gap_mask=None,
    )
    # The constant model gives 100 percent accuracy on class 2 with or without
    # masking, so delta = 0 for every line set and every random control. Under
    # Phipson-Smyth all five (4 controls + the observation itself, conceptually)
    # collapse to p = (4 + 1) / (4 + 1) = 1.0 (all controls <= observation when
    # both are zero). Under the OLD raw fraction (n_le / n_succeeded) this
    # would also be 1.0 since 4 of 4 are <= 0. Both formulas agree at the
    # ceiling; the floor case (n_le=0) is exercised by the corrected p never
    # being exactly zero.
    assert rows
    for r in rows:
        # No p-value should be exactly zero under the Phipson-Smyth formula.
        if np.isfinite(r.p_value_vs_random):
            assert r.p_value_vs_random > 0.0
            # And the minimum reachable p-value with n_random_controls=4 is
            # 1/(4+1) = 0.2.
            assert r.p_value_vs_random >= 1.0 / (4 + 1) - 1e-9
