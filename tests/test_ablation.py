"""Unit tests for src.interpret.ablation.

Covers:
  - class-matched vs pooled random-window reference sets give different
    per-class p-values when the per-class recall scale differs from the
    accuracy scale, and identical deltas / CIs otherwise
  - bin-count matched random windows reproduce the line set's per-segment
    bin counts, avoid the forbidden mask and do not overlap
  - exact McNemar p-value on constructed flip patterns
  - one-sided Clopper-Pearson upper bound (zero flips at n=145 -> ~0.0204)
  - mean change in true-class probability has the expected sign on a toy
    model with predict_proba, and is NaN without predict_proba
  - fill-mode alias "interp" and invalid-mode rejection
  - read_pickles_header_type on real STScI templates when the sibling data
    repository is present
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.interpret import ablation as abl
from src.interpret.ablation import (
    AblationRow,
    _sample_random_windows_bins,
    _segment_bin_counts,
    _window_mask,
    clopper_pearson_upper,
    masked_line_ablation,
    mcnemar_exact_p,
)
from src.interpret.lines import LINE_SETS

CLASS_LABELS = ["A", "F", "G", "K"]


def _wc(n: int = 400) -> np.ndarray:
    return np.linspace(4800.0, 6800.0, n).astype(np.float32)


class _RuleModel:
    """Deterministic classifier used to engineer known masking responses.

    Rules, evaluated in order on m = mean over the Mg b bins and
    r = max over the red half (> 5900 A):
      1. m < 0.6                 -> class 2 (Mg b dip)
      2. r <= 0.9                -> class 3 (red marker intact)
      3. m < 0.9                 -> class 3 (shallow Mg b marker)
      4. otherwise               -> class 1
    """

    def __init__(self, wc: np.ndarray) -> None:
        self.mgb = (wc >= 5165.0) & (wc <= 5186.0)
        self.red = wc > 5900.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X[:, self.mgb].mean(axis=1)
        r = X[:, self.red].max(axis=1)
        out = np.full(X.shape[0], 1, dtype=np.int64)
        out[(m >= 0.6) & (r > 0.9) & (m < 0.9)] = 3
        out[(m >= 0.6) & (r <= 0.9)] = 3
        out[m < 0.6] = 2
        return out


def _rule_dataset(wc: np.ndarray, n_per_class: int = 40) -> tuple[np.ndarray, np.ndarray]:
    model = _RuleModel(wc)
    n_bins = len(wc)
    X = np.ones((2 * n_per_class, n_bins), dtype=np.float32)
    y = np.empty(2 * n_per_class, dtype=np.int64)
    # Class 2: deep Mg b dip, no red marker.
    X[:n_per_class, model.mgb] = 0.5
    y[:n_per_class] = 2
    # Class 3 type A: red marker only. Type B: shallow Mg b marker only.
    half = n_per_class // 2
    X[n_per_class:n_per_class + half, model.red] = 0.7
    X[n_per_class + half:, model.mgb] = 0.8
    y[n_per_class:] = 3
    return X, y


def test_class_matched_vs_pooled_per_class_p_values() -> None:
    """Pooled accuracy reference overstates the K-class significance.

    Masking Mg b loses all class-2 rows (delta -1.0) and half of the class-3
    rows (delta -0.5). Random windows in the red half lose the other half of
    class 3 (class-3 recall delta -0.5, accuracy delta -0.25). Against the
    class-matched reference the class-3 observation is unremarkable
    (p ~ 0.5); against the pooled accuracy reference no window reaches
    -0.5 so p collapses to the 1/(B+1) floor.
    """
    wc = _wc(400)
    X, y = _rule_dataset(wc)
    model = _RuleModel(wc)
    common = dict(
        X_test=X, y_test=y, wave_centers=wc,
        line_sets={"Mg_b": LINE_SETS["Mg_b"]}, class_labels=CLASS_LABELS,
        per_class=True, n_bootstrap=20, n_random_controls=40, seed=3,
    )
    rows_cm = {r.mk_class: r for r in masked_line_ablation(model, null_mode="class_matched", **common)}
    rows_pl = {r.mk_class: r for r in masked_line_ablation(model, null_mode="pooled", **common)}
    B = rows_cm["K"].n_random_controls_succeeded
    assert B == 40 == rows_pl["K"].n_random_controls_succeeded
    floor = 1.0 / (B + 1)

    assert rows_cm["G"].delta_acc_mean == pytest.approx(-1.0)
    assert rows_cm["K"].delta_acc_mean == pytest.approx(-0.5)
    # Deltas and bootstrap CIs are identical across modes (same rng stream).
    for cls in ("G", "K"):
        assert rows_cm[cls].delta_acc_mean == rows_pl[cls].delta_acc_mean
        assert rows_cm[cls].delta_acc_ci_low == rows_pl[cls].delta_acc_ci_low
        assert rows_cm[cls].delta_acc_ci_high == rows_pl[cls].delta_acc_ci_high
    # Class 2 is at the floor in both modes.
    assert rows_cm["G"].p_value_vs_random == pytest.approx(floor)
    assert rows_pl["G"].p_value_vs_random == pytest.approx(floor)
    # Class 3: pooled collapses to the floor, class-matched does not.
    assert rows_pl["K"].p_value_vs_random == pytest.approx(floor)
    assert 0.2 < rows_cm["K"].p_value_vs_random < 0.8
    assert rows_cm["K"].null_mode == "class_matched"
    assert rows_pl["K"].null_mode == "pooled"
    # Paired statistics for class 2: 40 lost, 0 gained.
    assert rows_cm["G"].n_flip_lost == 40 and rows_cm["G"].n_flip_gained == 0
    assert rows_cm["K"].n_flip_lost == 20 and rows_cm["K"].n_flip_gained == 0


def test_all_row_identical_across_null_modes() -> None:
    wc = _wc(300)
    X, y = _rule_dataset(wc, n_per_class=20)
    model = _RuleModel(wc)
    kw = dict(X_test=X, y_test=y, wave_centers=wc, line_sets={"Mg_b": LINE_SETS["Mg_b"]},
              class_labels=CLASS_LABELS, per_class=False, n_bootstrap=10, n_random_controls=10, seed=1)
    a = masked_line_ablation(model, null_mode="class_matched", **kw)[0]
    b = masked_line_ablation(model, null_mode="pooled", **kw)[0]
    assert a.mk_class == b.mk_class == "ALL"
    assert a.p_value_vs_random == b.p_value_vs_random
    assert a.delta_acc_mean == b.delta_acc_mean


def test_bin_sampler_reproduces_segment_counts_and_avoids_forbidden() -> None:
    wc = _wc(400)
    forbidden = _window_mask(wc, LINE_SETS["H_balmer"]) | ((wc >= 5769.0) & (wc <= 5834.0))
    target = _segment_bin_counts(wc, LINE_SETS["H_balmer"])
    assert len(target) == 2 and all(k > 0 for k in target)
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(200):
        picks = _sample_random_windows_bins(rng, wc, target, forbidden)
        if not picks:
            continue
        n_ok += 1
        realised = _segment_bin_counts(wc, picks)
        assert realised == target
        masks = [_window_mask(wc, [p]) for p in picks]
        assert not any((m & forbidden).any() for m in masks)
        # Mutually non-overlapping segments.
        assert sum(int(m.sum()) for m in masks) == int(np.logical_or.reduce(masks).sum())
    assert n_ok >= 190


def test_bin_sampler_fails_closed() -> None:
    wc = _wc(100)
    rng = np.random.default_rng(0)
    assert _sample_random_windows_bins(rng, wc, [5], np.ones(100, dtype=bool)) == []
    assert _sample_random_windows_bins(rng, wc, [101], np.zeros(100, dtype=bool)) == []
    # Zero-count segments are skipped rather than failing.
    picks = _sample_random_windows_bins(rng, wc, [0, 4], np.zeros(100, dtype=bool))
    assert len(picks) == 1 and _segment_bin_counts(wc, picks) == [4]


def test_masked_line_ablation_records_realised_bin_counts() -> None:
    wc = _wc(400)
    X, y = _rule_dataset(wc, n_per_class=20)
    model = _RuleModel(wc)
    stats: dict = {}
    rows = masked_line_ablation(
        model, X, y, wc, {"H_balmer": LINE_SETS["H_balmer"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=5, n_random_controls=30, seed=5,
        gap_mask=(wc >= 5769.0) & (wc <= 5834.0), draw_stats_out=stats,
        match_on="bins",
    )
    s = stats["H_balmer"]
    assert s["match_on"] == "bins"
    assert s["target_bin_counts"] == _segment_bin_counts(wc, LINE_SETS["H_balmer"])
    assert s["succeeded"] > 0
    assert s["n_draws_bin_matched"] == s["succeeded"]
    assert list(s["realised_total_bins_histogram"]) == [str(sum(s["target_bin_counts"]))]
    assert all(r.n_bins_masked == sum(s["target_bin_counts"]) for r in rows)
    assert all(r.total_width_aa == pytest.approx(80.0) for r in rows)
    assert all(r.match_on == "bins" for r in rows)
    # Angstrom matching does not guarantee equal bin counts.
    stats_aa: dict = {}
    masked_line_ablation(
        model, X, y, wc, {"H_balmer": LINE_SETS["H_balmer"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=5, n_random_controls=30, seed=5,
        draw_stats_out=stats_aa, match_on="angstrom",
    )
    assert stats_aa["H_balmer"]["match_on"] == "angstrom"
    assert "n_draws_bin_matched" in stats_aa["H_balmer"]


def test_mcnemar_exact_p_constructed_patterns() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == pytest.approx(1.0)
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 * 0.5 ** 5)
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 * 0.5 ** 5)
    # 8 lost vs 2 gained: two-sided exact binomial.
    from scipy.stats import binomtest
    assert mcnemar_exact_p(8, 2) == pytest.approx(binomtest(8, 10, 0.5).pvalue)


def test_mcnemar_in_ablation_row_from_constructed_flips() -> None:
    """Three of ten class-2 rows flip to incorrect under Mg b masking -> p = 0.25."""
    wc = _wc(200)
    mgb = (wc >= 5165.0) & (wc <= 5186.0)
    X = np.ones((10, len(wc)), dtype=np.float32)
    X[:, mgb] = 0.5
    X[:3, 0] = 0.3  # marker rows that break when Mg b is filled
    y = np.full(10, 2, dtype=np.int64)

    class _M:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            filled = X_in[:, mgb].mean(axis=1) >= 0.99
            return np.where(filled & (X_in[:, 0] < 0.5), 1, 2).astype(np.int64)

    row = masked_line_ablation(
        _M(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=5, n_random_controls=5, seed=0,
    )[0]
    assert row.n_flip_lost == 3 and row.n_flip_gained == 0
    assert row.mcnemar_exact_p == pytest.approx(0.25)
    assert row.delta_acc_mean == pytest.approx(-0.3)
    assert np.isnan(row.mean_delta_true_prob)  # no predict_proba on the stub


def test_clopper_pearson_upper_bound() -> None:
    assert clopper_pearson_upper(0, 145) == pytest.approx(0.0204, abs=2e-4)
    assert clopper_pearson_upper(0, 145) == pytest.approx(1.0 - 0.05 ** (1.0 / 145))
    assert clopper_pearson_upper(145, 145) == 1.0
    assert np.isnan(clopper_pearson_upper(0, 0))
    assert 0.0 < clopper_pearson_upper(1, 145) < 0.05


def test_flip_rate_upper95_zero_flips_in_row() -> None:
    wc = _wc(200)
    X = np.ones((145, len(wc)), dtype=np.float32)
    y = np.full(145, 3, dtype=np.int64)

    class _Const:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            return np.full(X_in.shape[0], 3, dtype=np.int64)

    row = masked_line_ablation(
        _Const(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=3, n_random_controls=3, seed=0,
    )[0]
    assert row.n_test == 145 and row.n_flip_lost == 0
    assert row.flip_rate_upper95 == pytest.approx(0.0204, abs=2e-4)
    assert row.mcnemar_exact_p == 1.0


class _ProbaModel:
    """Toy probabilistic model: a Mg b dip gives P(class 2) = 0.8, else 0.3."""

    classes_ = np.array([1, 2, 3])

    def __init__(self, wc: np.ndarray) -> None:
        self.mgb = (wc >= 5165.0) & (wc <= 5186.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dip = X[:, self.mgb].mean(axis=1) < 0.9
        P = np.where(dip[:, None], np.array([[0.1, 0.8, 0.1]]), np.array([[0.4, 0.3, 0.3]]))
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def test_mean_delta_true_prob_sign_on_toy_model() -> None:
    wc = _wc(200)
    mgb = (wc >= 5165.0) & (wc <= 5186.0)
    X = np.ones((30, len(wc)), dtype=np.float32)
    X[:20, mgb] = 0.5           # class-2 rows with the dip
    y = np.array([2] * 20 + [1] * 10, dtype=np.int64)
    per_row: dict = {}
    rows = {r.mk_class: r for r in masked_line_ablation(
        _ProbaModel(wc), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=5, n_random_controls=5, seed=0, per_row_out=per_row,
    )}
    # Class 2 loses probability mass on the true class (0.8 -> 0.3).
    assert rows["G"].mean_delta_true_prob == pytest.approx(-0.5)
    assert rows["G"].n_flip_lost == 20
    # Class 1 rows have no dip and are unaffected.
    assert rows["F"].mean_delta_true_prob == pytest.approx(0.0)
    assert per_row["proba_base"].shape == (30, 3)
    assert per_row["proba_masked__Mg_b"].shape == (30, 3)
    assert per_row["proba_classes"].tolist() == [1, 2, 3]
    assert str(per_row["null_mode"]) == "class_matched"
    assert str(per_row["match_on"]) == "bins"


def test_fill_mode_alias_and_validation() -> None:
    wc = _wc(200)
    X = np.ones((8, len(wc)), dtype=np.float32)
    y = np.full(8, 2, dtype=np.int64)

    class _Const:
        def predict(self, X_in: np.ndarray) -> np.ndarray:
            return np.full(X_in.shape[0], 2, dtype=np.int64)

    per_row: dict = {}
    rows = masked_line_ablation(
        _Const(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
        per_class=True, n_bootstrap=2, n_random_controls=2, seed=0,
        fill_mode="interp", per_row_out=per_row,
    )
    assert rows[0].fill_mode == "gp_interp"
    assert str(per_row["fill_mode"]) == "gp_interp"
    with pytest.raises(ValueError):
        masked_line_ablation(_Const(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
                             n_bootstrap=2, n_random_controls=2, null_mode="bogus")
    with pytest.raises(ValueError):
        masked_line_ablation(_Const(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
                             n_bootstrap=2, n_random_controls=2, match_on="bogus")
    with pytest.raises(ValueError):
        masked_line_ablation(_Const(), X, y, wc, {"Mg_b": LINE_SETS["Mg_b"]}, CLASS_LABELS,
                             n_bootstrap=2, n_random_controls=2, fill_mode="bogus")


def test_occlusion_reexports_are_same_objects() -> None:
    from src.interpret import occlusion as occ

    assert occ.masked_line_ablation is abl.masked_line_ablation
    assert occ.AblationRow is AblationRow
    assert occ._apply_mask is abl._apply_mask
    assert occ.CONTINUUM_FILL == abl.CONTINUUM_FILL == 1.0


# Sibling data repository holding the STScI Pickles templates (if checked out
# next to this repository). Tests that need it are skipped otherwise.
_PICKLES_DIR = Path(__file__).resolve().parents[2] / "stellar-mk-audit" / "data" / "pickles"


@pytest.mark.skipif(
    not (_PICKLES_DIR / "pickles_uk_20.fits").exists(),
    reason="STScI Pickles templates not available alongside the repository",
)
@pytest.mark.parametrize("n,expected", [(20, "F8V"), (46, "B2IV"), (1, "O5V"), (131, "M2I")])
def test_read_pickles_header_type_real_template(n: int, expected: str) -> None:
    from src.interpret.benchmark import PICKLES_UVKLIB_MAP, read_pickles_header_type

    assert read_pickles_header_type(_PICKLES_DIR / f"pickles_uk_{n}.fits") == expected
    assert PICKLES_UVKLIB_MAP[n] == expected


@pytest.mark.skipif(
    not (_PICKLES_DIR / "pickles_uk_131.fits").exists(),
    reason="STScI Pickles templates not available alongside the repository",
)
def test_pickles_map_matches_all_real_headers() -> None:
    from src.interpret.benchmark import PICKLES_UVKLIB_MAP, read_pickles_header_type

    for n in range(1, 132):
        path = _PICKLES_DIR / f"pickles_uk_{n}.fits"
        if not path.exists():
            pytest.skip(f"missing {path.name}")
        assert read_pickles_header_type(path) == PICKLES_UVKLIB_MAP[n], n
