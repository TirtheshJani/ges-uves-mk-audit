# SPDX-License-Identifier: MIT
"""Matplotlib figures for the interpretability audit.

Production figures (``scripts/make_figure.py``):

  1. ``importance_main.pdf`` (``plot_main_figure``)
  2. ``confusion_matrix_*.pdf`` (``plot_confusion_matrix``)
  3. ``ablation_bars.pdf`` (``plot_ablation_bars``)

Legacy overlays kept for the audit dossier:

  4. ``importance_overlay.pdf`` (``plot_summary_overlay``)
  5. ``importance_overlay_per_class.pdf`` (``plot_per_class_overlay``)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.interpret.lines import LINE_SETS, lines_for_class, lines_in_window

logger = logging.getLogger(__name__)


_LINE_SET_DISPLAY = {
    "H_balmer": "H Balmer",
    "Mg_b": r"Mg b",
    "Na_D": r"Na D",
    "Ca_I": r"Ca I",
    "Fe_Cr": "Fe/Cr",
}

_LINE_NAME_DISPLAY = {
    "H_beta": r"H$\beta$",
    "Mg_b1": "Mg b",
    "Mg_b2": "Mg b",
    "Mg_b3": "Mg b",
    "Na_D2": "Na D",
    "Na_D1": "Na D",
    "Ca_I_6162": "Ca I 6162",
    "Ca_I_6439": "Ca I 6439",
    "H_alpha": r"H$\alpha$",
}


def _pretty_line_set(name: str) -> str:
    return _LINE_SET_DISPLAY.get(name, name.replace("_", " "))


def _pretty_line_name(name: str) -> str:
    return _LINE_NAME_DISPLAY.get(name, name.replace("_", " "))


def _grouped_line_labels(lines, group_tol_aa: float = 12.0):
    """Cluster nearby lines and emit one centroid label per cluster.

    Returns a list of (centroid_wavelength, label_text, member_wavelengths)
    tuples sorted by wavelength. Lines whose wavelength_aa values fall
    within ``group_tol_aa`` of an existing cluster are merged into that
    cluster and labelled with the shared display name (e.g. the Mg b
    triplet collapses to a single "Mg b" label over 5167-5184 A).
    """
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda lin: lin.wavelength_aa)
    clusters: list[list] = []
    for line in sorted_lines:
        placed = False
        if clusters:
            last_lines = clusters[-1]
            last_w = last_lines[-1].wavelength_aa
            last_label = _pretty_line_name(last_lines[-1].name)
            this_label = _pretty_line_name(line.name)
            if (
                this_label == last_label
                and line.wavelength_aa - last_w <= group_tol_aa
            ):
                clusters[-1].append(line)
                placed = True
        if not placed:
            clusters.append([line])
    out = []
    for members in clusters:
        wavelengths = [m.wavelength_aa for m in members]
        centroid = float(np.mean(wavelengths))
        label = _pretty_line_name(members[0].name)
        out.append((centroid, label, wavelengths))
    return out


def _normalise_trace(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def plot_summary_overlay(
    wave_centers: np.ndarray,
    importance_mean: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    representative_spectrum: np.ndarray,
    out_path: Path,
) -> None:
    """Render and save the summary overlay figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_spec, ax_imp) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10.0, 5.5),
        gridspec_kw={"height_ratios": [1.0, 1.3], "hspace": 0.05},
    )
    ax_spec.plot(wave_centers, representative_spectrum, color="black", lw=0.8)
    ax_spec.set_ylabel("normalized flux")
    ax_spec.set_ylim(
        np.nanpercentile(representative_spectrum, 2) - 0.05,
        np.nanpercentile(representative_spectrum, 99) + 0.05,
    )

    ax_imp.plot(
        wave_centers, _normalise_trace(importance_mean),
        color="black", lw=1.2, label="permutation importance",
    )
    for c, lab in enumerate(class_labels):
        ax_imp.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            lw=0.9, alpha=0.55, label=f"|SHAP| ({lab})",
        )
    ax_imp.set_ylabel("importance (normalized)")
    ax_imp.set_xlabel("wavelength (Å)")
    ax_imp.set_ylim(0.0, 1.02)

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    for line in lines_in_window(wmin, wmax):
        for ax in (ax_spec, ax_imp):
            ax.axvline(line.wavelength_aa, color="grey", ls="--", lw=0.5, alpha=0.7)
        ax_imp.text(
            line.wavelength_aa, 1.02, line.name.replace("_", " "),
            rotation=90, ha="right", va="bottom", fontsize=7, color="grey",
        )
    for set_name, windows in LINE_SETS.items():
        for lo, hi in windows:
            if hi < wmin or lo > wmax:
                continue
            ax_imp.axvspan(max(lo, wmin), min(hi, wmax), color="C0", alpha=0.08)

    ax_imp.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_overlay(
    wave_centers: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    out_path: Path,
) -> None:
    """Render and save the per-class overlay figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_labels)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, sharex=True, figsize=(10.0, 2.0 * n + 0.5),
        gridspec_kw={"hspace": 0.1},
    )
    if n == 1:
        axes = [axes]

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    for c, (ax, label) in enumerate(zip(axes, class_labels)):
        ax.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            color=f"C{c}", lw=1.1,
        )
        ax.set_ylabel(f"|SHAP|\n({label})")
        ax.set_ylim(0.0, 1.02)
        class_lines = [
            line for line in lines_for_class(label)
            if wmin <= line.wavelength_aa <= wmax
        ]
        for line in class_lines:
            ax.axvline(line.wavelength_aa, color="grey", ls="--", lw=0.6, alpha=0.8)
            ax.text(
                line.wavelength_aa, 1.02, line.name.replace("_", " "),
                rotation=90, ha="right", va="bottom", fontsize=7, color="grey",
            )
    axes[-1].set_xlabel("wavelength (Å)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_importance(
    wave_centers: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    gap_mask: np.ndarray,
    out_path: Path,
) -> None:
    """Render the per-class importance figure with gap-mask shading.

    One subplot per surviving class. Each panel shows that class's
    mean |SHAP| trace, vertical dashed lines at the MK_LINES entries
    diagnostic for that class, and a hatched grey band over any
    contiguous run of bins where ``gap_mask`` is True
    (UVES inter-chip gap, Sacco et al. 2014).

    Saved at 300 DPI for print.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wave_centers = np.asarray(wave_centers, dtype=float)
    gap_mask = np.asarray(gap_mask, dtype=bool)
    if gap_mask.shape != wave_centers.shape:
        raise ValueError(
            f"gap_mask shape {gap_mask.shape} != wave_centers shape {wave_centers.shape}"
        )

    n = len(class_labels)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, sharex=True, figsize=(11.0, 2.2 * n + 0.6),
        gridspec_kw={"hspace": 0.14},
    )
    if n == 1:
        axes = [axes]

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    gap_spans = _contiguous_spans(wave_centers, gap_mask)

    for c, (ax, label) in enumerate(zip(axes, class_labels)):
        ax.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            color=f"C{c}", lw=1.1,
        )
        ax.set_ylabel(f"|SHAP|\n({label})")
        ax.set_ylim(0.0, 1.20)
        for lo, hi in gap_spans:
            ax.axvspan(lo, hi, facecolor="grey", alpha=0.18, hatch="///",
                       edgecolor="grey", lw=0.0)
        class_lines = [
            line for line in lines_for_class(label)
            if wmin <= line.wavelength_aa <= wmax
        ]
        line_groups = _grouped_line_labels(class_lines)
        for centroid, ltext, members in line_groups:
            for w in members:
                ax.axvline(w, color="black", ls="--", lw=0.55, alpha=0.70)
            ax.text(
                centroid, 1.10, ltext,
                rotation=0, ha="center", va="center", fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="grey",
                          lw=0.4, alpha=0.92),
            )
    axes[-1].set_xlabel("wavelength (Å)")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _contiguous_spans(
    wave_centers: np.ndarray, mask: np.ndarray
) -> list[tuple[float, float]]:
    """Return a list of (lo, hi) wavelength spans for each contiguous run
    of True in ``mask``.
    """
    spans: list[tuple[float, float]] = []
    in_run = False
    start = 0
    for i, flag in enumerate(mask):
        if flag and not in_run:
            in_run = True
            start = i
        elif not flag and in_run:
            in_run = False
            spans.append((float(wave_centers[start]), float(wave_centers[i - 1])))
    if in_run:
        spans.append((float(wave_centers[start]), float(wave_centers[-1])))
    return spans


def plot_main_figure(
    wave_centers: np.ndarray,
    representative_spectrum: np.ndarray,
    shap_per_class: np.ndarray,
    class_labels: list[str],
    ablation_rows: list[dict],
    gap_mask: np.ndarray,
    out_path: Path,
    top_k_per_class: int = 10,
) -> None:
    """Main figure: spectrum + per-class SHAP top-K markers + MK lines + gap shading.

    Composition:
      * Top axis: representative normalized spectrum with MK_LINES dashed verticals.
      * Bottom axis: per-class SHAP trace; top-K markers per class drawn as
        coloured circles at the relevant wavelength bins.
      * Inset panel on the right: ablation delta-acc bars (pulled from
        ``ablation_rows``).
      * Grey hatched band over each contiguous gap_mask True run
        (UVES inter-chip gap, Sacco et al. 2014).

    All wavelengths in air. Saved at 300 DPI.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wave_centers = np.asarray(wave_centers, dtype=float)
    gap_mask = np.asarray(gap_mask, dtype=bool)
    if gap_mask.shape != wave_centers.shape:
        raise ValueError(
            f"gap_mask shape {gap_mask.shape} != wave_centers shape {wave_centers.shape}"
        )
    representative_spectrum = np.asarray(representative_spectrum, dtype=float)
    shap_per_class = np.asarray(shap_per_class, dtype=float)
    if shap_per_class.shape[0] != len(class_labels):
        raise ValueError(
            f"shap_per_class first axis {shap_per_class.shape[0]} != "
            f"len(class_labels) {len(class_labels)}"
        )

    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[3.2, 1.0], height_ratios=[1.0, 1.6],
        hspace=0.08, wspace=0.22,
    )
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_imp = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_bars = fig.add_subplot(gs[:, 1])

    wmin, wmax = float(wave_centers.min()), float(wave_centers.max())
    gap_spans = _contiguous_spans(wave_centers, gap_mask)

    ax_spec.plot(wave_centers, representative_spectrum, color="black", lw=0.8)
    ax_spec.set_ylabel("normalized flux")
    spec_lo = np.nanpercentile(representative_spectrum, 2) - 0.05
    spec_hi = np.nanpercentile(representative_spectrum, 99) + 0.16
    ax_spec.set_ylim(spec_lo, spec_hi)
    plt.setp(ax_spec.get_xticklabels(), visible=False)
    gap_patch = None
    for lo, hi in gap_spans:
        gap_patch = ax_spec.axvspan(
            lo, hi, facecolor="grey", alpha=0.20, hatch="///",
            edgecolor="grey", lw=0.0,
        )

    line_groups = _grouped_line_labels(list(lines_in_window(wmin, wmax)))
    label_y = spec_hi - 0.03
    for centroid, label, members in line_groups:
        for w in members:
            for ax in (ax_spec, ax_imp):
                ax.axvline(w, color="black", ls="--", lw=0.55, alpha=0.55)
        ax_spec.text(
            centroid, label_y, label,
            rotation=0, ha="center", va="top", fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.75),
        )

    for c, label in enumerate(class_labels):
        ax_imp.plot(
            wave_centers, _normalise_trace(shap_per_class[c]),
            lw=1.0, alpha=0.6, label=f"|SHAP| ({label})", color=f"C{c}",
        )
        trace = np.asarray(shap_per_class[c], dtype=float).copy()
        trace[gap_mask] = -np.inf
        if top_k_per_class > 0 and trace.size:
            order = np.argsort(trace)[::-1]
            top = order[:top_k_per_class]
            top = top[np.isfinite(trace[top])]
            if top.size:
                ax_imp.scatter(
                    wave_centers[top], _normalise_trace(shap_per_class[c])[top],
                    s=28, color=f"C{c}", edgecolor="black", lw=0.5, zorder=5,
                    label=f"top-{top_k_per_class} ({label})" if c == 0 else None,
                )
    ax_imp.set_ylabel("|SHAP| (normalized)")
    ax_imp.set_xlabel("wavelength (Å)")
    ax_imp.set_ylim(0.0, 1.10)
    for lo, hi in gap_spans:
        ax_imp.axvspan(
            lo, hi, facecolor="grey", alpha=0.20, hatch="///",
            edgecolor="grey", lw=0.0,
        )

    legend_handles, legend_labels = ax_imp.get_legend_handles_labels()
    if gap_patch is not None:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, facecolor="grey", alpha=0.20, hatch="///",
            edgecolor="grey",
        ))
        legend_labels.append("UVES inter-chip gap")
    ax_imp.legend(
        legend_handles, legend_labels,
        loc="upper right", fontsize=8, framealpha=0.9, ncol=1,
    )

    bar_labels = []
    bar_means = []
    bar_lows = []
    bar_highs = []
    bar_colors = []
    for row in ablation_rows:
        ls = _pretty_line_set(row.get("line_set", "?"))
        cls = row.get("mk_class", "?")
        bar_labels.append(f"{ls}\non {cls}")
        bar_means.append(float(row.get("delta_acc_mean", 0.0)))
        lo = float(row.get("delta_acc_ci_low", 0.0))
        hi = float(row.get("delta_acc_ci_high", 0.0))
        bar_lows.append(float(row.get("delta_acc_mean", 0.0)) - lo)
        bar_highs.append(hi - float(row.get("delta_acc_mean", 0.0)))
        bar_colors.append("C3" if row.get("is_pivot", False) else "C0")

    if bar_labels:
        x = np.arange(len(bar_labels))
        ax_bars.bar(
            x, bar_means, color=bar_colors, alpha=0.88,
            yerr=[np.abs(bar_lows), np.abs(bar_highs)], capsize=3,
            edgecolor="black", lw=0.5,
        )
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(bar_labels, fontsize=8, rotation=0)
        ax_bars.axhline(0.0, color="black", lw=0.7)
        ax_bars.set_ylabel(r"$\Delta$ accuracy (masked $-$ baseline)")
        ax_bars.set_title("Masked-line ablation", fontsize=10)
        ax_bars.tick_params(axis="x", labelsize=7.5)
        headline_proxy = plt.Rectangle((0, 0), 1, 1, fc="C0", alpha=0.88,
                                       edgecolor="black")
        pivot_proxy = plt.Rectangle((0, 0), 1, 1, fc="C3", alpha=0.88,
                                    edgecolor="black")
        ax_bars.legend(
            [headline_proxy, pivot_proxy],
            ["headline", "pivot"],
            loc="lower right", fontsize=8, framealpha=0.9,
        )
    else:
        ax_bars.set_axis_off()

    logger.info("plot_main_figure: writing %s (n_classes=%d, top_k=%d)",
                out_path, len(class_labels), top_k_per_class)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Render a confusion-matrix heatmap with per-cell counts and row
    percentages annotated.

    Each cell shows the integer count on the top line and the row-normalised
    percentage in parentheses underneath (or just the count if the row total
    is zero or non-integer). Both axes carry the supplied class labels.
    Saved at 300 DPI for print.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    if cm.ndim != 2:
        raise ValueError(f"cm must be 2D; got shape {cm.shape}")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square; got shape {cm.shape}")
    if cm.shape[0] != len(labels):
        raise ValueError(
            f"cm shape {cm.shape} != len(labels) {len(labels)}"
        )

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("predicted", fontsize=10)
    ax.set_ylabel("true", fontsize=10)
    ax.set_title(title, fontsize=11)

    row_sums = cm.sum(axis=1)
    threshold = float(cm.max()) / 2.0 if cm.size else 0.0
    integer_cm = np.all(np.equal(np.mod(cm, 1), 0))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > threshold else "black"
            if integer_cm:
                count_text = f"{int(value):d}"
            else:
                count_text = f"{value:.2f}"
            if integer_cm and row_sums[i] > 0:
                pct = 100.0 * value / row_sums[i]
                ax.text(j, i - 0.18, count_text,
                        ha="center", va="center", color=text_color,
                        fontsize=11, fontweight="bold")
                ax.text(j, i + 0.20, f"({pct:.0f}%)",
                        ha="center", va="center", color=text_color,
                        fontsize=8.5)
            else:
                ax.text(j, i, count_text,
                        ha="center", va="center", color=text_color,
                        fontsize=10)

    logger.info("plot_confusion_matrix: writing %s (labels=%s)", out_path, labels)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_ablation_bars(
    ablation_rows: list[dict],
    headline_pairs: list[tuple[str, str]],
    pivot_pairs: list[tuple[str, str]],
    out_path: Path,
    n_random_controls: int = 500,
) -> None:
    """Render the ablation bar chart.

    Each row is one (line_set, mk_class) pair from ``gate_eval.json``.
    Headline pairs are drawn in colour 1 (solid). Pivot pairs are drawn
    in colour 2 with a hatch pattern. 95 percent CI error bars come from
    ``delta_acc_ci_low`` and ``delta_acc_ci_high``. Each bar is annotated
    with its p-value vs random.

    Saved at 300 DPI for print.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Order headline first, then pivot, preserving caller order.
    ordered: list[dict] = []
    for ls, cls in headline_pairs:
        for row in ablation_rows:
            if row.get("line_set") == ls and row.get("mk_class") == cls:
                ordered.append({**row, "is_pivot": False})
                break
    for ls, cls in pivot_pairs:
        for row in ablation_rows:
            if row.get("line_set") == ls and row.get("mk_class") == cls:
                ordered.append({**row, "is_pivot": True})
                break

    if not ordered:
        raise ValueError(
            "no rows matched headline or pivot pairs; check ablation_rows keys."
        )

    n_tests = [int(r.get("n_test", 0)) for r in ordered]
    labels = [
        f"{_pretty_line_set(r['line_set'])}\non {r['mk_class']}"
        + (f"\n(n = {n})" if n > 0 else "")
        for r, n in zip(ordered, n_tests)
    ]
    means = [float(r.get("delta_acc_mean", 0.0)) for r in ordered]
    ci_low = [float(r.get("delta_acc_ci_low", 0.0)) for r in ordered]
    ci_high = [float(r.get("delta_acc_ci_high", 0.0)) for r in ordered]
    pvals = [float(r.get("p_value_vs_random", float("nan"))) for r in ordered]
    is_pivot = [bool(r.get("is_pivot", False)) for r in ordered]

    err_low = [abs(m - lo) for m, lo in zip(means, ci_low)]
    err_high = [abs(hi - m) for m, hi in zip(means, ci_high)]

    colors = ["C3" if pv else "C0" for pv in is_pivot]
    hatches = ["///" if pv else "" for pv in is_pivot]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(ordered))
    bars = ax.bar(
        x, means, color=colors, alpha=0.88,
        yerr=[err_low, err_high], capsize=5,
        edgecolor="black", lw=0.6,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.axhline(0.0, color="black", lw=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$\Delta$ accuracy (masked $-$ baseline)")
    ax.set_title(
        "Interventional masked-line ablation: headline pairs (solid) and "
        "pivot pairs (hatched)",
        fontsize=10,
    )
    ax.grid(axis="y", color="lightgrey", lw=0.5, alpha=0.6, zorder=-1)

    span = max(ci_high + [0.0]) - min(ci_low + [0.0])
    if span <= 0:
        span = 1.0
    ax.set_ylim(min(ci_low + [0.0]) - 0.16 * span,
                max(ci_high + [0.0]) + 0.20 * span)
    p_floor = 1.0 / max(n_random_controls, 1)
    for xi, m, hi_v, pv in zip(x, means, ci_high, pvals):
        if np.isnan(pv):
            p_text = "p = n/a"
        elif pv < p_floor:
            p_text = f"p < {p_floor:.3f}"
        else:
            p_text = f"p = {pv:.3f}"
        dx_text = f"$\\Delta A$ = {m:+.3f}"
        # Place both annotations above the upper error-bar tip.
        top = max(m, hi_v)
        ax.text(xi, top + 0.04 * span, dx_text,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(xi, top + 0.10 * span, p_text,
                ha="center", va="bottom", fontsize=8.5, color="dimgrey")

    headline_proxy = plt.Rectangle((0, 0), 1, 1, fc="C0", alpha=0.88,
                                   edgecolor="black", lw=0.6)
    pivot_proxy = plt.Rectangle((0, 0), 1, 1, fc="C3", alpha=0.88, hatch="///",
                                edgecolor="black", lw=0.6)
    ax.legend(
        [headline_proxy, pivot_proxy],
        ["headline pair", "pivot pair"],
        loc="lower right", fontsize=9, framealpha=0.9,
    )

    ax.tick_params(axis="x", which="major", pad=6)

    logger.info(
        "plot_ablation_bars: writing %s (n_headline=%d, n_pivot=%d)",
        out_path, len(headline_pairs), len(pivot_pairs),
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
