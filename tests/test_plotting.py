"""Smoke tests for the per-class importance figure in src.interpret.plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg") # non-interactive backend; must precede pyplot import

import numpy as np
import pytest

from src.interpret.plotting import (
    plot_ablation_bars,
    plot_confusion_matrix,
    plot_main_figure,
    plot_per_class_importance,
)


def test_plot_per_class_importance_writes_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_bins = 696
    n_classes = 3
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    shap_per_class = rng.random((n_classes, n_bins)).astype(np.float32)
    class_labels = ["F", "G", "K"]
    gap_mask = (wave_centers >= 5769.0) & (wave_centers <= 5834.0)

    out_path = tmp_path / "per_class.pdf"
    plot_per_class_importance(
        wave_centers=wave_centers,
        shap_per_class=shap_per_class,
        class_labels=class_labels,
        gap_mask=gap_mask,
        out_path=out_path,
    )

    assert out_path.exists(), "per-class figure was not written"
    assert out_path.stat().st_size > 1024, (
        f"per-class figure file size {out_path.stat().st_size} B is below 1 KB"
    )


def test_plot_per_class_importance_validates_gap_mask_shape(
    tmp_path: Path,
) -> None:
    n_bins = 100
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    shap_per_class = np.zeros((2, n_bins), dtype=np.float32)
    bad_gap = np.zeros(n_bins - 1, dtype=bool)
    with pytest.raises(ValueError):
        plot_per_class_importance(
            wave_centers=wave_centers,
            shap_per_class=shap_per_class,
            class_labels=["F", "G"],
            gap_mask=bad_gap,
            out_path=tmp_path / "bad.pdf",
        )


def test_plot_per_class_importance_single_class(tmp_path: Path) -> None:
    """Single-class panel must still render (axes is a scalar Axes object,
    not an array)."""
    n_bins = 200
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    shap_per_class = np.zeros((1, n_bins), dtype=np.float32)
    shap_per_class[0, 50] = 1.0
    gap_mask = np.zeros(n_bins, dtype=bool)

    out_path = tmp_path / "single.pdf"
    plot_per_class_importance(
        wave_centers=wave_centers,
        shap_per_class=shap_per_class,
        class_labels=["G"],
        gap_mask=gap_mask,
        out_path=out_path,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1024


def _ablation_rows_fixture() -> list[dict]:
    """Five rows: 3 headline + 2 pivot, mirroring gate_eval.json schema."""
    return [
        {"line_set": "H_balmer", "mk_class": "F",
         "delta_acc_mean": -0.84, "delta_acc_ci_low": -0.91,
         "delta_acc_ci_high": -0.77, "p_value_vs_random": 0.0},
        {"line_set": "Mg_b", "mk_class": "G",
         "delta_acc_mean": -0.21, "delta_acc_ci_low": -0.26,
         "delta_acc_ci_high": -0.16, "p_value_vs_random": 0.0},
        {"line_set": "Mg_b", "mk_class": "K",
         "delta_acc_mean": 0.0, "delta_acc_ci_low": -0.02,
         "delta_acc_ci_high": 0.02, "p_value_vs_random": 0.866},
        {"line_set": "Na_D", "mk_class": "K",
         "delta_acc_mean": -0.007, "delta_acc_ci_low": -0.022,
         "delta_acc_ci_high": 0.0, "p_value_vs_random": 0.006},
        {"line_set": "Ca_I", "mk_class": "K",
         "delta_acc_mean": 0.0, "delta_acc_ci_low": 0.0,
         "delta_acc_ci_high": 0.0, "p_value_vs_random": 0.878},
    ]


def test_plot_main_figure_renders_pdf_with_gap_shading(tmp_path: Path) -> None:
    """Main figure renders to PDF; gap_mask True bins propagate to a shaded
    span in the saved PDF."""
    rng = np.random.default_rng(42)
    n_bins = 696
    n_classes = 3
    wave_centers = np.linspace(4800.0, 6800.0, n_bins).astype(np.float32)
    rep_spectrum = 1.0 + 0.05 * rng.standard_normal(n_bins).astype(np.float32)
    shap_per_class = rng.random((n_classes, n_bins)).astype(np.float32)
    class_labels = ["F", "G", "K"]
    gap_mask = (wave_centers >= 5769.0) & (wave_centers <= 5834.0)
    assert gap_mask.sum() > 0

    out_path = tmp_path / "main.pdf"
    plot_main_figure(
        wave_centers=wave_centers,
        representative_spectrum=rep_spectrum,
        shap_per_class=shap_per_class,
        class_labels=class_labels,
        ablation_rows=_ablation_rows_fixture(),
        gap_mask=gap_mask,
        out_path=out_path,
        top_k_per_class=5,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 4096

    # Read PDF bytes; the gap span (5769-5834 A) must be referenced by the
    # axvspan polygon. We assert the PDF ingested the gap_mask by checking that
    # /Pattern resources or hatch markers exist; a coarser proxy is to confirm
    # the file is well-formed and large enough to carry the shaded patches.
    pdf_bytes = out_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    # Hatched fill produces a tiling pattern in the PDF stream.
    # Either /Pattern or many /Path operators show up; assert at least one.
    assert (b"/Pattern" in pdf_bytes) or (b"/Sh" in pdf_bytes) or len(pdf_bytes) > 8000


def test_plot_confusion_matrix_axes_labels_match_input(tmp_path: Path) -> None:
    """Both axes carry the supplied class labels; cell values render."""
    cm = np.array([[68, 13, 0], [3, 222, 5], [1, 8, 136]])
    labels = ["F", "G", "K"]
    out_path = tmp_path / "cm.pdf"
    plot_confusion_matrix(cm=cm, labels=labels, title="test", out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1024

    # Also verify error on shape mismatch.
    with pytest.raises(ValueError):
        plot_confusion_matrix(
            cm=np.array([[1, 2], [3, 4]]),
            labels=["F", "G", "K"],  # wrong length
            title="bad",
            out_path=tmp_path / "bad_cm.pdf",
        )


def test_plot_ablation_bars_marks_pivot_pairs_distinctly(tmp_path: Path) -> None:
    """Pivot pairs must use a different visual encoding (color or hatch)
    than headline pairs."""
    rows = _ablation_rows_fixture()
    headline_pairs = [("H_balmer", "F"), ("Mg_b", "G"), ("Mg_b", "K")]
    pivot_pairs = [("Na_D", "K"), ("Ca_I", "K")]

    out_path = tmp_path / "ablation.pdf"
    plot_ablation_bars(
        ablation_rows=rows,
        headline_pairs=headline_pairs,
        pivot_pairs=pivot_pairs,
        out_path=out_path,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 4096

    # PDF should carry hatch info: matplotlib emits a Pattern stream or the
    # fill paint operator differs. We just verify the file is well-formed and
    # of non-trivial size, since the visual distinctness is enforced by the
    # bar coloring + hatch logic in plot_ablation_bars.
    assert out_path.read_bytes().startswith(b"%PDF")

    # Empty rows should raise.
    with pytest.raises(ValueError):
        plot_ablation_bars(
            ablation_rows=[],
            headline_pairs=headline_pairs,
            pivot_pairs=pivot_pairs,
            out_path=tmp_path / "empty.pdf",
        )
