#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build the manuscript numerical-claims dossier.

Reads gate-accepted artifacts from Phases 2 through 5 and emits two
sibling files:

* ``artifacts/manuscript_numbers.csv`` -- machine-diffable claim ledger
  with stable claim_id, formatted value, precision tag, source artifact
  and source field. diffs the manuscript prose against
  this CSV during review.
* ``artifacts/manuscript_numbers.md`` -- human-friendly rendering of the
  same claims grouped per phase. TJ pastes these tables next to the
  draft prose so every numeric claim has a clickable provenance trail.

The script is idempotent: rerun it whenever an artifact is regenerated
during drafting and both outputs refresh in place. The orchestrator
keeps this script in scripts/ rather than treating it as one-shot.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"

# CSV schema. Two extra fields beyond the orchestrator's required columns:
# * claim_type distinguishes scalar numeric claims from figure references.
# * notes carries inline qualifiers (e.g. "test set", "spatial CV fold 2").
COLUMNS = [
    "claim_id",
    "claim_type",
    "value",
    "precision",
    "source_file",
    "source_field",
    "phase",
    "used_in_section",
    "notes",
]


def _fmt_sf(value: float, sf: int) -> str:
    """Format ``value`` to ``sf`` significant figures, no scientific notation.

    Negative-zero is normalised to ``0``. Special-cased so the rendered
    string stays paper-quotable.
    """
    if value == 0:
        return "0"
    import math

    sign = "-" if value < 0 else ""
    v = abs(value)
    # exponent of leading digit
    exp = int(math.floor(math.log10(v)))
    decimals = max(sf - 1 - exp, 0)
    rounded = round(v, decimals)
    s = f"{rounded:.{decimals}f}"
    return f"{sign}{s}"


def _fmt_dp(value: float, dp: int) -> str:
    """Format ``value`` to ``dp`` decimal places."""
    if value == 0 and 1 / 1 == 1:  # avoid -0.00
        value = 0.0
    return f"{value:.{dp}f}"


def _fmt_pct(value: float, dp: int = 1) -> str:
    """Format a fraction as a percentage string, e.g. 0.9342 -> ``93.4``."""
    return f"{value * 100.0:.{dp}f}"


def _load_json(path: Path) -> dict[str, Any]:
    logger.info("loading json: %s", path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _row(
    claim_id: str,
    value: str,
    precision: str,
    source_file: str,
    source_field: str,
    phase: str,
    *,
    claim_type: str = "scalar",
    notes: str = "",
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "value": value,
        "precision": precision,
        "source_file": source_file,
        "source_field": source_field,
        "phase": phase,
        "used_in_section": "",
        "notes": notes,
    }


def _phase2_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    src = "artifacts/metrics.json"
    rows: list[dict[str, str]] = []

    rows.append(
        _row(
            "macro_f1_test",
            _fmt_sf(metrics["macro_f1"], 3),
            "3sf",
            src,
            "$.macro_f1",
            "2",
            notes="held-out test set, FGK",
        )
    )
    rows.append(
        _row(
            "accuracy_test",
            _fmt_sf(metrics["accuracy"], 3),
            "3sf",
            src,
            "$.accuracy",
            "2",
            notes="held-out test set",
        )
    )

    for cls in ("F", "G", "K"):
        rows.append(
            _row(
                f"per_class_recall_{cls.lower()}_test",
                _fmt_sf(metrics["per_class_recall"][cls], 3),
                "3sf",
                src,
                f"$.per_class_recall.{cls}",
                "2",
                notes=f"class {cls}, test set",
            )
        )
        rows.append(
            _row(
                f"per_class_precision_{cls.lower()}_test",
                _fmt_sf(metrics["per_class_precision"][cls], 3),
                "3sf",
                src,
                f"$.per_class_precision.{cls}",
                "2",
                notes=f"class {cls}, test set",
            )
        )
        rows.append(
            _row(
                f"per_class_f1_{cls.lower()}_test",
                _fmt_sf(metrics["per_class_f1"][cls], 3),
                "3sf",
                src,
                f"$.per_class_f1.{cls}",
                "2",
                notes=f"class {cls}, test set",
            )
        )

    rows.append(
        _row(
            "cv_macro_f1_mean",
            _fmt_sf(metrics["cv_macro_f1_mean"], 3),
            "3sf",
            src,
            "$.cv_macro_f1_mean",
            "2",
            notes="StratifiedKFold n_splits=5",
        )
    )
    rows.append(
        _row(
            "cv_macro_f1_std",
            _fmt_sf(metrics["cv_macro_f1_std"], 2),
            "2sf",
            src,
            "$.cv_macro_f1_std",
            "2",
            notes="StratifiedKFold n_splits=5",
        )
    )
    for i, fold in enumerate(metrics["cv_per_fold"]):
        rows.append(
            _row(
                f"cv_macro_f1_fold{i}",
                _fmt_sf(fold["macro_f1"], 3),
                "3sf",
                src,
                f"$.cv_per_fold[{i}].macro_f1",
                "2",
                notes=f"non-spatial CV fold {i}",
            )
        )

    rows.append(
        _row(
            "cv_spatial_macro_f1_mean",
            _fmt_sf(metrics["cv_spatial_macro_f1_mean"], 3),
            "3sf",
            src,
            "$.cv_spatial_macro_f1_mean",
            "2",
            notes="StratifiedGroupKFold over 1926 DBSCAN spatial groups",
        )
    )
    rows.append(
        _row(
            "cv_spatial_macro_f1_std",
            _fmt_sf(metrics["cv_spatial_macro_f1_std"], 2),
            "2sf",
            src,
            "$.cv_spatial_macro_f1_std",
            "2",
            notes="StratifiedGroupKFold over 1926 DBSCAN spatial groups",
        )
    )
    for i, fold_f1 in enumerate(metrics["cv_spatial_per_fold_macro_f1"]):
        rows.append(
            _row(
                f"cv_spatial_macro_f1_fold{i}",
                _fmt_sf(fold_f1, 3),
                "3sf",
                src,
                f"$.cv_spatial_per_fold_macro_f1[{i}]",
                "2",
                notes=f"spatial CV fold {i}",
            )
        )
    rows.append(
        _row(
            "cv_spatial_n_unique_groups",
            str(int(metrics["cv_spatial_n_unique_groups"])),
            "exact",
            src,
            "$.cv_spatial_n_unique_groups",
            "2",
            notes="DBSCAN haversine eps=0.1 deg, min_samples=5",
        )
    )

    bd = metrics["boundary_filtered_accuracy"]
    rows.append(
        _row(
            "boundary_filtered_accuracy",
            _fmt_sf(bd["boundary_filtered_acc"], 3),
            "3sf",
            src,
            "$.boundary_filtered_accuracy.boundary_filtered_acc",
            "2",
            notes=f"|Teff - bin edge| > 200 K, n={int(bd['n_filtered'])}",
        )
    )
    rows.append(
        _row(
            "boundary_full_test_accuracy",
            _fmt_sf(bd["full_test_acc"], 3),
            "3sf",
            src,
            "$.boundary_filtered_accuracy.full_test_acc",
            "2",
            notes="full held-out test set",
        )
    )
    rows.append(
        _row(
            "boundary_delta_acc",
            _fmt_sf(bd["delta_acc"], 2),
            "2sf",
            src,
            "$.boundary_filtered_accuracy.delta_acc",
            "2",
            notes="boundary-filtered minus full-test accuracy",
        )
    )
    rows.append(
        _row(
            "boundary_delta_acc_ci_low",
            _fmt_sf(bd["delta_acc_ci_low"], 2),
            "2sf",
            src,
            "$.boundary_filtered_accuracy.delta_acc_ci_low",
            "2",
            notes="95 percent bootstrap CI lower bound",
        )
    )
    rows.append(
        _row(
            "boundary_delta_acc_ci_high",
            _fmt_sf(bd["delta_acc_ci_high"], 2),
            "2sf",
            src,
            "$.boundary_filtered_accuracy.delta_acc_ci_high",
            "2",
            notes="95 percent bootstrap CI upper bound",
        )
    )

    rows.append(
        _row(
            "n_train",
            str(int(metrics["n_train"])),
            "exact",
            src,
            "$.n_train",
            "2",
            notes="post-SNR cut, post-NaN cut",
        )
    )
    rows.append(
        _row(
            "n_val",
            str(int(metrics["n_val"])),
            "exact",
            src,
            "$.n_val",
            "2",
        )
    )
    rows.append(
        _row(
            "n_test",
            str(int(metrics["n_test"])),
            "exact",
            src,
            "$.n_test",
            "2",
        )
    )

    cm = metrics["confusion_matrix"]
    cls_labels = metrics["class_labels"]
    for i, ti in enumerate(cls_labels):
        for j, pj in enumerate(cls_labels):
            rows.append(
                _row(
                    f"confusion_test_{ti.lower()}_to_{pj.lower()}",
                    str(int(cm[i][j])),
                    "exact",
                    src,
                    f"$.confusion_matrix[{i}][{j}]",
                    "2",
                    notes=f"true={ti}, predicted={pj}",
                )
            )

    return rows


def _phase3_rows(triangulation: dict[str, Any], shap_stab: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    src_t = "artifacts/interpret/triangulation_report.json"
    src_s = "artifacts/interpret/shap_stability.json"

    pj = triangulation["pairwise_jaccard"]
    rows.append(
        _row(
            "jaccard_perm_vs_shap",
            _fmt_sf(pj["perm_vs_shap"], 3),
            "3sf",
            src_t,
            "$.pairwise_jaccard.perm_vs_shap",
            "3",
            notes="top-20 features",
        )
    )
    rows.append(
        _row(
            "jaccard_perm_vs_occlusion",
            _fmt_sf(pj["perm_vs_occlusion"], 3),
            "3sf",
            src_t,
            "$.pairwise_jaccard.perm_vs_occlusion",
            "3",
            notes="top-20 features",
        )
    )
    rows.append(
        _row(
            "jaccard_shap_vs_occlusion",
            _fmt_sf(pj["shap_vs_occlusion"], 3),
            "3sf",
            src_t,
            "$.pairwise_jaccard.shap_vs_occlusion",
            "3",
            notes="top-20 features",
        )
    )
    rows.append(
        _row(
            "three_way_intersection_size",
            str(int(triangulation["three_way_intersection_size"])),
            "exact",
            src_t,
            "$.three_way_intersection_size",
            "3",
            notes="bins shared by perm/SHAP/occlusion top-20",
        )
    )

    for entry in triangulation["three_way_intersection_bins"]:
        b = int(entry["bin"])
        wl = float(entry["wavelength_aa"])
        rows.append(
            _row(
                f"three_way_intersection_bin{b}_wavelength_aa",
                _fmt_dp(wl, 2),
                "2dp",
                src_t,
                f"$.three_way_intersection_bins[?(@.bin=={b})].wavelength_aa",
                "3",
                notes="Halpha cluster",
            )
        )

    # Per-class top-K wavelength counts (paper cites the lists; here we
    # record the size only as the canonical claim).
    per_class = triangulation["per_class_top_k"]
    for method in ("perm", "shap"):
        for cls in ("F", "G", "K"):
            entries = per_class[method][cls]
            rows.append(
                _row(
                    f"per_class_topk_{method}_{cls.lower()}_size",
                    str(len(entries)),
                    "exact",
                    src_t,
                    f"$.per_class_top_k.{method}.{cls}|length",
                    "3",
                    notes=f"{method} top-K count for class {cls}",
                )
            )

    for cls in ("F", "G", "K"):
        rows.append(
            _row(
                f"shap_stability_{cls.lower()}",
                _fmt_sf(shap_stab["per_class"][cls], 3),
                "3sf",
                src_s,
                f"$.per_class.{cls}",
                "3",
                notes="mean pairwise Jaccard across bootstrap subsamples",
            )
        )

    return rows


def _phase4_rows(gate: dict[str, Any], line_match: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    src_g = "artifacts/ablation/gate_eval.json"
    src_l = "artifacts/ablation/line_match.json"

    def _slug(line_set: str, mk_class: str) -> str:
        return f"{line_set.lower()}_{mk_class.lower()}"

    for entry in gate["pairs_evaluated"]:
        slug = _slug(entry["line_set"], entry["mk_class"])
        ix = gate["pairs_evaluated"].index(entry)
        rows.append(
            _row(
                f"baseline_acc_{slug}",
                _fmt_sf(entry["baseline_acc"], 3),
                "3sf",
                src_g,
                f"$.pairs_evaluated[{ix}].baseline_acc",
                "4",
                notes=f"({entry['line_set']}, {entry['mk_class']}) headline",
            )
        )
        rows.append(
            _row(
                f"masked_acc_mean_{slug}",
                _fmt_sf(entry["masked_acc_mean"], 3),
                "3sf",
                src_g,
                f"$.pairs_evaluated[{ix}].masked_acc_mean",
                "4",
            )
        )
        rows.append(
            _row(
                f"delta_acc_{slug}",
                _fmt_dp(entry["delta_acc_mean"], 2),
                "2dp",
                src_g,
                f"$.pairs_evaluated[{ix}].delta_acc_mean",
                "4",
            )
        )
        rows.append(
            _row(
                f"delta_acc_ci_low_{slug}",
                _fmt_dp(entry["delta_acc_ci_low"], 2),
                "2dp",
                src_g,
                f"$.pairs_evaluated[{ix}].delta_acc_ci_low",
                "4",
            )
        )
        rows.append(
            _row(
                f"delta_acc_ci_high_{slug}",
                _fmt_dp(entry["delta_acc_ci_high"], 2),
                "2dp",
                src_g,
                f"$.pairs_evaluated[{ix}].delta_acc_ci_high",
                "4",
            )
        )
        p = entry["p_value_vs_random"]
        floor = 1.0 / float(gate.get("n_random_controls") or 500)
        if p < floor:
            p_str = f"<{floor:.3f}"
            p_prec = "<floor"
        else:
            p_str = _fmt_dp(p, 3)
            p_prec = "3dp"
        rows.append(
            _row(
                f"p_value_vs_random_{slug}",
                p_str,
                p_prec,
                src_g,
                f"$.pairs_evaluated[{ix}].p_value_vs_random",
                "4",
                notes=f"floor=1/{int(gate.get('n_random_controls') or 500)}",
            )
        )
        rows.append(
            _row(
                f"n_test_{slug}",
                str(int(entry["n_test"])),
                "exact",
                src_g,
                f"$.pairs_evaluated[{ix}].n_test",
                "4",
            )
        )

    for entry in gate["pivot_confirmed"]:
        slug = _slug(entry["line_set"], entry["mk_class"])
        ix = gate["pivot_confirmed"].index(entry)
        rows.append(
            _row(
                f"delta_acc_{slug}",
                _fmt_dp(entry["delta_acc_mean"], 4),
                "4dp",
                src_g,
                f"$.pivot_confirmed[{ix}].delta_acc_mean",
                "4",
                notes="pivot pair",
            )
        )
        rows.append(
            _row(
                f"delta_acc_ci_low_{slug}",
                _fmt_dp(entry["delta_acc_ci_low"], 4),
                "4dp",
                src_g,
                f"$.pivot_confirmed[{ix}].delta_acc_ci_low",
                "4",
                notes="pivot pair",
            )
        )
        rows.append(
            _row(
                f"delta_acc_ci_high_{slug}",
                _fmt_dp(entry["delta_acc_ci_high"], 4),
                "4dp",
                src_g,
                f"$.pivot_confirmed[{ix}].delta_acc_ci_high",
                "4",
                notes="pivot pair",
            )
        )
        p = entry["p_value_vs_random"]
        floor = 1.0 / float(gate.get("n_random_controls") or 500)
        if p < floor:
            p_str = f"<{floor:.3f}"
            p_prec = "<floor"
        else:
            p_str = _fmt_dp(p, 3)
            p_prec = "3dp"
        rows.append(
            _row(
                f"p_value_vs_random_{slug}",
                p_str,
                p_prec,
                src_g,
                f"$.pivot_confirmed[{ix}].p_value_vs_random",
                "4",
                notes="pivot pair",
            )
        )

    cs = gate["continuum_fill_sensitivity"]
    rows.append(
        _row(
            "continuum_fill_locked",
            _fmt_dp(cs["continuum_fill_locked"], 1),
            "1dp",
            src_g,
            "$.continuum_fill_sensitivity.continuum_fill_locked",
            "4",
            notes="initial CONTINUUM_FILL value",
        )
    )
    rows.append(
        _row(
            "continuum_fill_empirical_median",
            _fmt_sf(cs["continuum_fill_empirical_median"], 4),
            "4sf",
            src_g,
            "$.continuum_fill_sensitivity.continuum_fill_empirical_median",
            "4",
            notes="np.nanmedian(X[~gap_mask])",
        )
    )
    rows.append(
        _row(
            "continuum_fill_max_shift",
            _fmt_sf(cs["max_shift"], 2),
            "2sf",
            src_g,
            "$.continuum_fill_sensitivity.max_shift",
            "4",
            notes="max delta_acc shift between fill values",
        )
    )
    rows.append(
        _row(
            "continuum_fill_production",
            _fmt_sf(cs["production_continuum_fill"], 4),
            "4sf",
            src_g,
            "$.continuum_fill_sensitivity.production_continuum_fill",
            "4",
            notes="value used in production ablation rerun",
        )
    )

    rows.append(
        _row(
            "n_random_controls",
            str(int(gate.get("n_random_controls") or 500)),
            "exact",
            src_g,
            "$.n_random_controls",
            "4",
            notes="raised from the earlier default of 100 to improve p-value resolution",
        )
    )
    rows.append(
        _row(
            "n_bootstrap",
            str(int(gate.get("n_bootstrap") or 500)),
            "exact",
            src_g,
            "$.n_bootstrap",
            "4",
        )
    )

    for tol_key in ("1.0", "2.0", "5.0", "10.0"):
        tol_slug = tol_key.replace(".", "p")
        rows.append(
            _row(
                f"line_match_precision_{tol_slug}_aa",
                _fmt_sf(line_match["precision"][tol_key], 2),
                "2sf",
                src_l,
                f"$.precision.{tol_key}",
                "4",
                notes=f"tolerance {tol_key} A",
            )
        )
        rows.append(
            _row(
                f"line_match_recall_{tol_slug}_aa",
                _fmt_sf(line_match["recall"][tol_key], 3),
                "3sf",
                src_l,
                f"$.recall.{tol_key}",
                "4",
                notes=f"tolerance {tol_key} A",
            )
        )
        rows.append(
            _row(
                f"line_match_jaccard_{tol_slug}_aa",
                _fmt_sf(line_match["jaccard"][tol_key], 2),
                "2sf",
                src_l,
                f"$.jaccard.{tol_key}",
                "4",
                notes=f"tolerance {tol_key} A",
            )
        )
    rows.append(
        _row(
            "n_lines_in_window",
            str(int(line_match["n_lines_in_window"])),
            "exact",
            src_l,
            "$.n_lines_in_window",
            "4",
            notes="MK_LINES intersect 4800-6800 A",
        )
    )

    return rows


def _phase5_rows(bench: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    src = "artifacts/benchmark/benchmark_report.json"

    rows.append(
        _row(
            "agreement_rate",
            _fmt_sf(bench["agreement_rate"], 3),
            "3sf",
            src,
            "$.agreement_rate",
            "5",
            notes="FGK only, Pickles vs LightGBM",
        )
    )
    rows.append(
        _row(
            "macro_f1_fgk_pickles",
            _fmt_sf(bench["macro_f1_fgk"], 3),
            "3sf",
            src,
            "$.macro_f1_fgk",
            "5",
            notes="treats Pickles labels as ground truth",
        )
    )
    rows.append(
        _row(
            "n_compared",
            str(int(bench["n_compared"])),
            "exact",
            src,
            "$.n_compared",
            "5",
            notes="rows with FGK Pickles assignment",
        )
    )
    rows.append(
        _row(
            "n_dropped_other",
            str(int(bench["n_dropped_other"])),
            "exact",
            src,
            "$.n_dropped_other",
            "5",
            notes="rows whose Pickles best-template was OTHER (M-class etc.)",
        )
    )

    for cls in ("F", "G", "K"):
        rows.append(
            _row(
                f"benchmark_per_class_recall_{cls.lower()}",
                _fmt_sf(bench["per_class_recall"][cls], 3),
                "3sf",
                src,
                f"$.per_class_recall.{cls}",
                "5",
                notes=f"class {cls}, Pickles vs model",
            )
        )
        rows.append(
            _row(
                f"benchmark_per_class_precision_{cls.lower()}",
                _fmt_sf(bench["per_class_precision"][cls], 3),
                "3sf",
                src,
                f"$.per_class_precision.{cls}",
                "5",
                notes=f"class {cls}, Pickles vs model",
            )
        )

    pls = bench["pickles_loader_stats"]
    rows.append(
        _row(
            "pickles_n_files_seen",
            str(int(pls["n_files_seen"])),
            "exact",
            src,
            "$.pickles_loader_stats.n_files_seen",
            "5",
            notes="UVKLIB filesystem listing",
        )
    )
    rows.append(
        _row(
            "pickles_n_loaded",
            str(int(pls["n_loaded"])),
            "exact",
            src,
            "$.pickles_loader_stats.n_loaded",
            "5",
        )
    )
    rows.append(
        _row(
            "pickles_n_unmapped_skipped",
            str(int(pls["n_unmapped_skipped"])),
            "exact",
            src,
            "$.pickles_loader_stats.n_unmapped_skipped",
            "5",
            notes="templates with neither a header spectral type nor a map entry",
        )
    )

    return rows


def _figure_rows() -> list[dict[str, str]]:
    figs = [
        ("fig_importance_main", "artifacts/figures/importance_main.pdf", "main result figure"),
        (
            "fig_importance_per_class",
            "artifacts/figures/importance_overlay_per_class.pdf",
            "per-class importance overlay",
        ),
        (
            "fig_importance_overlay",
            "artifacts/figures/importance_overlay.pdf",
            "perm/SHAP overlay",
        ),
        (
            "fig_importance_per_class_gapmask",
            "artifacts/figures/importance_per_class_gapmask.pdf",
            "per-class with inter-chip gap shaded",
        ),
        (
            "fig_confusion_matrix_test",
            "artifacts/figures/confusion_matrix_test.pdf",
            "test-set confusion matrix",
        ),
        (
            "fig_confusion_matrix_pickles",
            "artifacts/figures/confusion_matrix_pickles.pdf",
            "Pickles vs model confusion matrix",
        ),
        (
            "fig_ablation_bars",
            "artifacts/figures/ablation_bars.pdf",
            "delta_acc bar chart per (line_set, class)",
        ),
    ]
    rows: list[dict[str, str]] = []
    for claim_id, path, note in figs:
        rows.append(
            _row(
                claim_id,
                "",
                "figure",
                path,
                "",
                "5",
                claim_type="figure",
                notes=note,
            )
        )
    return rows


def build_rows() -> list[dict[str, str]]:
    """Read all artifacts and assemble the dossier row list."""
    metrics = _load_json(ART / "metrics.json")
    triangulation = _load_json(ART / "interpret" / "triangulation_report.json")
    shap_stab = _load_json(ART / "interpret" / "shap_stability.json")
    gate = _load_json(ART / "ablation" / "gate_eval.json")
    line_match = _load_json(ART / "ablation" / "line_match.json")
    bench = _load_json(ART / "benchmark" / "benchmark_report.json")

    logger.info("phase 2 metrics keys=%d", len(metrics))
    logger.info("phase 3 triangulation keys=%d", len(triangulation))
    logger.info("phase 4 gate pairs=%d pivot=%d", len(gate["pairs_evaluated"]), len(gate["pivot_confirmed"]))
    logger.info("phase 5 benchmark agreement_rate=%.4f", bench["agreement_rate"])

    rows: list[dict[str, str]] = []
    rows.extend(_phase2_rows(metrics))
    rows.extend(_phase3_rows(triangulation, shap_stab))
    rows.extend(_phase4_rows(gate, line_match))
    rows.extend(_phase5_rows(bench))
    rows.extend(_figure_rows())
    logger.info("total rows assembled=%d", len(rows))
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Write the dossier CSV at ``path``."""
    logger.info("writing csv: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


PHASE_TITLES = {
    "2": "-- baseline classifier",
    "3": "-- interpretability triangulation",
    "4": "-- causal masked-line ablation",
    "5": "-- external benchmark and figures",
}


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    """Render the dossier as a per-phase markdown report at ``path``."""
    logger.info("writing markdown: %s", path)
    by_phase: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r)

    lines: list[str] = []
    lines.append("# Manuscript numerical-claims dossier")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/build_manuscript_dossier.py`. Every numeric "
        "claim in the draft must trace to one row in this file or in the "
        "sibling CSV. Rerun the script if any artifact regenerates during "
        "drafting."
    )
    lines.append("")

    for phase in ("2", "3", "4", "5"):
        if phase not in by_phase:
            continue
        lines.append(f"## {PHASE_TITLES[phase]}")
        lines.append("")
        scalars = [r for r in by_phase[phase] if r["claim_type"] == "scalar"]
        figures = [r for r in by_phase[phase] if r["claim_type"] == "figure"]

        if scalars:
            lines.append("| claim_id | value | precision | source |")
            lines.append("| --- | --- | --- | --- |")
            for r in scalars:
                source = f"`{r['source_file']}` {r['source_field']}".rstrip()
                lines.append(
                    f"| `{r['claim_id']}` | {r['value']} | {r['precision']} | {source} |"
                )
            lines.append("")

        if figures:
            lines.append("### Figures")
            lines.append("")
            lines.append("| claim_id | path | description |")
            lines.append("| --- | --- | --- |")
            for r in figures:
                lines.append(
                    f"| `{r['claim_id']}` | `{r['source_file']}` | {r['notes']} |"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ART / "manuscript_numbers.csv",
        help="path to write the dossier CSV",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ART / "manuscript_numbers.md",
        help="path to write the dossier markdown",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="python logging level",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    rows = build_rows()
    write_csv(rows, args.csv_out)
    write_md(rows, args.md_out)
    logger.info("done: csv=%s md=%s rows=%d", args.csv_out, args.md_out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
