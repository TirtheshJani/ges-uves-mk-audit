#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit LaTeX macros for every number quoted in submission/manuscript.tex.

Reads the JSON artifacts under ``artifacts/`` and writes

  submission/revision_numbers.tex     \\newcommand macros (one per quoted number)
  submission/ablation_table.tex       masked-line ablation table, all line sets x classes
  submission/ablation_k_table.tex     K-class rows under the 2 x 2 continuum / fill design
  submission/ablation_strata_table.tex  luminosity and boundary-distance strata
  submission/line_sets_table.tex      copy of artifacts/revision/line_sets_table.tex
  artifacts/revision/manuscript_numbers_index.json   macro -> source file and field

Two groups of macros are produced.

Group A (always available): classifier metrics, spatial-CV summary, line-peak
triangulation, equivalent-width comparison, scale decomposition, retraining under
alternative normalisations, luminosity mix, and the template cross-check. Each
value is read from the artifact named in the index file.

Group B (ablation): read from ``artifacts/revision/ablation_recalibrated.json``
and ``artifacts/revision/ablation_stratified.json``. When a file is absent every
macro that would come from it is defined as ``\\textbf{[TBD-ablation]}`` and the
corresponding table body is a single placeholder row, so the manuscript still
compiles and the open slots are visible in the typeset output.

Run from the repository root: ``python scripts/revision/manuscript_numbers.py``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

TBD = r"\textbf{[TBD-ablation]}"

RUN_PREFIX = {
    "production__gp_interp": "Prim",
    "production__constant": "Const",
    "polynomial_n5__gp_interp": "Poly",
    "polynomial_n5__constant": "PolyConst",
}
SET_CODE = {"H_balmer": "Hb", "Mg_b": "Mg", "Na_D": "Na", "Ca_I": "Ca", "Fe_Cr": "Fe"}
SET_ORDER = ["H_balmer", "Mg_b", "Na_D", "Ca_I", "Fe_Cr"]
SET_TEX = {
    "H_balmer": r"H Balmer",
    "Mg_b": r"Mg\,\textsc{b}",
    "Na_D": r"Na\,\textsc{d}",
    "Ca_I": r"Ca\,\textsc{i}",
    "Fe_Cr": r"Fe\,\textsc{i}/Cr\,\textsc{i}",
}
CLASSES = ["F", "G", "K"]
ABL_FIELDS = {  # macro suffix -> (row key, kind)
    "N": ("n_test", "int"),
    "Base": ("baseline_acc", "f3"),
    "Masked": ("masked_acc_mean", "f3"),
    "Delta": ("delta_acc_mean", "sf3"),
    "CiLo": ("delta_acc_ci_low", "sf3"),
    "CiHi": ("delta_acc_ci_high", "sf3"),
    "P": ("p_value_vs_random", "p"),
    "Lost": ("n_flip_lost", "int"),
    "Gained": ("n_flip_gained", "int"),
    "Mcn": ("mcnemar_exact_p", "p"),
    "Dp": ("mean_delta_true_prob", "sf3"),
    "Upper": ("flip_rate_upper95", "f3"),
    "Bins": ("n_bins_masked", "int"),
    "Width": ("total_width_aa", "f0"),
    "NullB": (None, "nullb"),  # number of random windows at least as extreme as observed: p*(B+1) - 1
}
STRATA_PARTS = {
    ("luminosity", "dwarf"): "Dwarf",
    ("luminosity", "giant"): "Giant",
    ("boundary", "within_150K"): "Near",
    ("boundary", "beyond_150K"): "Far",
}
STR_FIELDS = {
    "N": ("n", "int"),
    "Delta": ("delta_recall", "sf3"),
    "CiLo": ("delta_ci_low", "sf3"),
    "CiHi": ("delta_ci_high", "sf3"),
    "Lost": ("n_flip_lost", "int"),
    "Gained": ("n_flip_gained", "int"),
    "Dp": ("mean_delta_true_prob", "sf3"),
}


# --------------------------------------------------------------------------- formatting
def _ens(s: str) -> str:
    return r"\ensuremath{" + s + "}"


def f_num(x: float | None, d: int = 3, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return r"\textbf{n/a}"
    s = f"{x:.{d}f}"
    if float(s) == 0.0:  # a value that rounds to zero carries no sign
        return _ens(s.lstrip("-"))
    if signed and not s.startswith("-"):
        s = "+" + s
    return _ens(s)


def f_int(x: Any) -> str:
    return str(int(x))


def f_pct(x: float, d: int = 1) -> str:
    return f"{100.0 * x:.{d}f}"


def f_p(x: float | None) -> str:
    """p-value: fixed decimals for p >= 1e-3, scientific notation below."""
    if x is None or not math.isfinite(x):
        return r"\textbf{n/a}"
    if x >= 1e-3:
        return _ens(f"{x:.3f}") if x >= 0.01 else _ens(f"{x:.4f}")
    mant, exp = f"{x:.1e}".split("e")
    return _ens(rf"{mant}\times10^{{{int(exp)}}}")


def fmt(x: Any, kind: str) -> str:
    if kind == "int":
        return f_int(x)
    if kind == "f0":
        return f_num(x, 0)
    if kind == "f1":
        return f_num(x, 1)
    if kind == "f2":
        return f_num(x, 2)
    if kind == "f3":
        return f_num(x, 3)
    if kind == "f4":
        return f_num(x, 4)
    if kind == "sf3":
        return f_num(x, 3, signed=True)
    if kind == "p":
        return f_p(x)
    if kind == "pct1":
        return f_pct(x, 1)
    if kind == "pct0":
        return f_pct(x, 0)
    if kind == "str":
        return str(x)
    raise ValueError(kind)


class Macros:
    """Ordered collection of macro definitions with provenance."""

    def __init__(self) -> None:
        self.defs: dict[str, str] = {}
        self.source: dict[str, str] = {}
        self.tbd: list[str] = []

    def add(self, name: str, value: str, source: str) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self.defs:
            raise ValueError(f"duplicate macro {name}")
        self.defs[name] = value
        self.source[name] = source

    def add_tbd(self, name: str, source: str) -> None:
        self.add(name, TBD, source + " (absent: placeholder)")
        self.tbd.append(name)


def write_lf(path: Path, text: str) -> None:
    """Write text with LF line endings on every platform (.tex sources are LF-only)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_json(rel: str) -> dict[str, Any]:
    with (REPO_ROOT / rel).open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- group A
def group_a(m: Macros) -> None:
    met_p = "artifacts/metrics.json"
    met = load_json(met_p)
    n_test, n_train, n_val = met["n_test"], met["n_train"], met["n_val"]
    m.add("nTest", f_int(n_test), met_p + ":n_test")
    m.add("nTrain", f_int(n_train), met_p + ":n_train")
    m.add("nVal", f_int(n_val), met_p + ":n_val")
    m.add("testMacroF", f_num(met["macro_f1"]), met_p + ":macro_f1")
    m.add("testAcc", f_num(met["accuracy"]), met_p + ":accuracy")
    prec, rec, f1 = met["per_class_precision"], met["per_class_recall"], met["per_class_f1"]
    for c in CLASSES:
        m.add(f"testPrec{c}", f_num(prec[c]), f"{met_p}:per_class_precision.{c}")
        m.add(f"testRec{c}", f_num(rec[c]), f"{met_p}:per_class_recall.{c}")
        m.add(f"testFOne{c}", f_num(f1[c]), f"{met_p}:per_class_f1.{c}")
    m.add("testMacroP", f_num(sum(prec.values()) / 3), met_p + ":mean(per_class_precision)")
    m.add("testMacroR", f_num(sum(rec.values()) / 3), met_p + ":mean(per_class_recall)")
    cm = met["confusion_matrix"]
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            m.add(f"conf{ci}{cj}", f_int(cm[i][j]), f"{met_p}:confusion_matrix[{i}][{j}]")
        m.add(f"nClass{ci}", f_int(sum(cm[i])), f"{met_p}:sum(confusion_matrix[{i}])")
    m.add("adjFG", f_pct((cm[0][1] + cm[1][0]) / n_test), met_p + ":(F->G + G->F)/n_test, percent")
    m.add("adjGK", f_pct((cm[1][2] + cm[2][1]) / n_test), met_p + ":(G->K + K->G)/n_test, percent")
    m.add("adjFKRows", f_int(cm[0][2] + cm[2][0]), met_p + ":F->K + K->F rows")
    m.add("adjFKPct", f_pct((cm[0][2] + cm[2][0]) / n_test, 2), met_p + ":(F->K + K->F)/n_test, percent")

    m.add("cvMean", f_num(met["cv_macro_f1_mean"]), met_p + ":cv_macro_f1_mean")
    m.add("cvSd", f_num(met["cv_macro_f1_std"]), met_p + ":cv_macro_f1_std (population SD over folds)")
    m.add("cvSpatMean", f_num(met["cv_spatial_macro_f1_mean"]), met_p + ":cv_spatial_macro_f1_mean")
    m.add("cvSpatSd", f_num(met["cv_spatial_macro_f1_std"]), met_p + ":cv_spatial_macro_f1_std")
    m.add("cvGap", f_num(met["cv_macro_f1_mean"] - met["cv_spatial_macro_f1_mean"]), met_p + ":difference of CV means")
    z = (met["macro_f1"] - met["cv_spatial_macro_f1_mean"]) / met["cv_spatial_macro_f1_std"]
    m.add("cvTestZ", f_num(z, 1), met_p + ":(test macro-F1 - spatial CV mean)/spatial CV SD")
    sg = met["cv_spatial_singleton_stats"]
    m.add("nSpatGroups", f_int(sg["n_unique_groups"]), met_p + ":cv_spatial_singleton_stats.n_unique_groups")
    m.add("nSingleton", f_int(sg["n_singleton_groups"]), met_p + ":cv_spatial_singleton_stats.n_singleton_groups")
    m.add("nClusters", f_int(sg["n_non_singleton_clusters"]), met_p + ":cv_spatial_singleton_stats.n_non_singleton_clusters")
    m.add("singGroupPct", f_pct(sg["singleton_group_fraction"]), met_p + ":singleton_group_fraction, percent")
    m.add("singRowPct", f_pct(sg["singleton_row_fraction"]), met_p + ":singleton_row_fraction, percent")
    m.add("nTrainVal", f_int(sg["n_total_rows"]), met_p + ":cv_spatial_singleton_stats.n_total_rows")
    m.add("nClusterRows", f_int(sg["n_rows_clustered"]), met_p + ":cv_spatial_singleton_stats.n_rows_clustered")
    import pandas as pd

    lab_n = len(pd.read_parquet(REPO_ROOT / "artifacts" / "ges_mk_labels.parquet"))
    m.add("nMatched", f_int(lab_n), "artifacts/ges_mk_labels.parquet:n_rows")
    m.add("dbscanEps", f_num(sg["eps_deg"], 1), met_p + ":cv_spatial_singleton_stats.eps_deg")
    m.add("dbscanMin", f_int(sg["min_samples"]), met_p + ":cv_spatial_singleton_stats.min_samples")
    bf = met["boundary_filtered_accuracy"]
    m.add("bdFullAcc", f_num(bf["full_test_acc"]), met_p + ":boundary_filtered_accuracy.full_test_acc")
    m.add("bdFiltAcc", f_num(bf["boundary_filtered_acc"]), met_p + ":boundary_filtered_accuracy.boundary_filtered_acc")
    m.add("bdDelta", f_num(bf["delta_acc"]), met_p + ":boundary_filtered_accuracy.delta_acc")
    m.add("bdCiLo", f_num(bf["delta_acc_ci_low"]), met_p + ":boundary_filtered_accuracy.delta_acc_ci_low")
    m.add("bdCiHi", f_num(bf["delta_acc_ci_high"]), met_p + ":boundary_filtered_accuracy.delta_acc_ci_high")
    m.add("bdN", f_int(bf["n_filtered"]), met_p + ":boundary_filtered_accuracy.n_filtered")
    m.add("bdThresh", f_int(bf["threshold_k"]), met_p + ":boundary_filtered_accuracy.threshold_k")

    # triangulation
    tri_p = "artifacts/interpret/triangulation_report.json"
    tri = load_json(tri_p)
    pj = tri["pairwise_jaccard"]
    m.add("jacPS", f_num(pj["perm_vs_shap"]), tri_p + ":pairwise_jaccard.perm_vs_shap")
    m.add("jacPO", f_num(pj["perm_vs_occlusion"]), tri_p + ":pairwise_jaccard.perm_vs_occlusion")
    m.add("jacSO", f_num(pj["shap_vs_occlusion"]), tri_p + ":pairwise_jaccard.shap_vs_occlusion")
    m.add("triTopK", f_int(tri["top_k"]), tri_p + ":top_k")
    wl = [b["wavelength_aa"] for b in tri["three_way_intersection_bins"]]
    m.add("triNThree", f_int(tri["three_way_intersection_size"]), tri_p + ":three_way_intersection_size")
    m.add("triWlMin", f_num(min(wl), 0), tri_p + ":min three_way_intersection_bins.wavelength_aa")
    m.add("triWlMax", f_num(max(wl), 0), tri_p + ":max three_way_intersection_bins.wavelength_aa")
    m.add("nGapBins", f_int(tri["gap_mask_n_bins_true"]), tri_p + ":gap_mask_n_bins_true")
    m.add("gapLo", f_num(tri["gap_mask_wavelength_range_aa"][0], 0), tri_p + ":gap_mask_wavelength_range_aa[0]")
    m.add("gapHi", f_num(tri["gap_mask_wavelength_range_aa"][1], 0), tri_p + ":gap_mask_wavelength_range_aa[1]")

    # equivalent widths
    ew_p = "artifacts/ablation/mg_b_ew_test.json"
    ew = load_json(ew_p)
    for c in CLASSES:
        m.add(f"ewMed{c}", f_num(ew["summary_ew_mg_b_total"][c]["median_aa"], 2), f"{ew_p}:summary_ew_mg_b_total.{c}.median_aa")
    ks = ew["ks_test_K_vs_G"]
    m.add("ewDiff", f_num(ks["K_minus_G_median_aa"], 2), ew_p + ":ks_test_K_vs_G.K_minus_G_median_aa")
    m.add("ewKsP", f_p(ks["p_value"]), ew_p + ":ks_test_K_vs_G.p_value")
    m.add("ewLineHW", f_num(ew["line_half_width_aa"], 0), ew_p + ":line_half_width_aa")
    m.add("ewContHW", f_num(ew["continuum_half_width_aa"], 0), ew_p + ":continuum_half_width_aa")
    m.add("meanBinWidth", f_num(ew["mean_bin_width_aa"], 2), ew_p + ":mean_bin_width_aa")
    m.add("meanBinWidthFine", f_num(ew["mean_bin_width_aa"] / 5.0, 2), ew_p + ":mean_bin_width_aa / 5 (rebin factor)")

    # legacy fill sensitivity and robustness retrains
    ge_p = "artifacts/ablation/gate_eval.json"
    ge = load_json(ge_p)
    cf = ge["continuum_fill_sensitivity"]
    m.add("fillLocked", f_num(cf["continuum_fill_locked"], 1), ge_p + ":continuum_fill_sensitivity.continuum_fill_locked")
    m.add("fillEmpirical", f_num(cf["continuum_fill_empirical_median"], 4), ge_p + ":continuum_fill_sensitivity.continuum_fill_empirical_median")
    m.add("fillMaxShift", f_num(cf["max_shift"], 3), ge_p + ":continuum_fill_sensitivity.max_shift")
    m.add("fillTrigger", f_num(cf["shift_threshold"], 2), ge_p + ":continuum_fill_sensitivity.shift_threshold")
    uw_p = "artifacts/sensitivity/uniform_weights_mg_b_k.json"
    uw = load_json(uw_p)
    m.add("uwDelta", f_num(uw["mg_b_k"]["delta_acc_mean"], 3, signed=True), uw_p + ":mg_b_k.delta_acc_mean")
    m.add("uwMacroF", f_num(uw["test_macro_f1"]), uw_p + ":test_macro_f1")
    hg_p = "artifacts/sensitivity/hyperparam_grid_mg_b_k.json"
    hg = load_json(hg_p)
    m.add("gridMaxDelta", f_num(hg["max_abs_delta_acc"]), hg_p + ":max_abs_delta_acc")
    m.add("gridNCells", f_int(hg["n_cells"]), hg_p + ":n_cells")

    # scale decomposition
    sd_p = "artifacts/revision/scale_decomposition.json"
    sd = load_json(sd_p)
    iv = sd["interventions"]
    iv_code = {"production": "Prod", "polynomial_n5": "Poly", "polynomial_n5_gap_restored": "PolyGap",
               "common_median": "Common", "uniform_x1.09": "UniNine", "uniform_x_best": "UniBest", "gk_swap": "Swap"}
    for key, code in iv_code.items():
        d = iv[key]
        for c in CLASSES:
            m.add(f"sd{code}Rec{c}", f_num(d["per_class_recall"][c]), f"{sd_p}:interventions.{key}.per_class_recall.{c}")
        m.add(f"sd{code}MacroF", f_num(d["macro_f1"]), f"{sd_p}:interventions.{key}.macro_f1")
        m.add(f"sd{code}GtoK", f_int(d["g_to_k"]), f"{sd_p}:interventions.{key}.g_to_k")
        m.add(f"sd{code}Level", f_num(d["median_level"], 3), f"{sd_p}:interventions.{key}.median_level")
    m.add("sdCostCommon", f_num(iv["production"]["macro_f1"] - iv["common_median"]["macro_f1"]), sd_p + ":production.macro_f1 - common_median.macro_f1")
    m.add("sdCostSwap", f_num(iv["production"]["macro_f1"] - iv["gk_swap"]["macro_f1"]), sd_p + ":production.macro_f1 - gk_swap.macro_f1")
    m.add("sdCostUni", f_num(iv["production"]["macro_f1"] - iv["uniform_x1.09"]["macro_f1"]), sd_p + ":production.macro_f1 - uniform_x1.09.macro_f1")
    m.add("sdUniMult", f_num(sd["inputs"]["uniform_multiplier"], 2), sd_p + ":inputs.uniform_multiplier")
    m.add("sdBestMult", f_num(sd["median_levels"]["best_uniform_multiplier"], 3), sd_p + ":median_levels.best_uniform_multiplier")
    m.add("sdPolyShiftPct", f_pct(sd["median_levels"]["polynomial_n5_global_test_median"] / sd["median_levels"]["production_global_test_median"] - 1.0, 1),
          sd_p + ":median_levels ratio - 1, percent")
    m.add("sdMedProdGlobal", f_num(sd["median_levels"]["production_global_test_median"], 4), sd_p + ":median_levels.production_global_test_median")
    m.add("sdMedPolyGlobal", f_num(sd["median_levels"]["polynomial_n5_global_test_median"], 4), sd_p + ":median_levels.polynomial_n5_global_test_median")
    m.add("sdMedTrainGlobal", f_num(sd["median_levels"]["production_train_global_median"], 4), sd_p + ":median_levels.production_train_global_median")
    m.add("sdNGapBins", f_int(sd["inputs"]["n_gap_bins"]), sd_p + ":inputs.n_gap_bins")
    gv = sd["gap_bin_treatment"]["gap_bin_values_production_test"]
    m.add("sdGapMean", f_num(gv["mean"], 2), sd_p + ":gap_bin_treatment.gap_bin_values_production_test.mean")
    m.add("sdGapSd", f_num(gv["std"], 2), sd_p + ":gap_bin_treatment.gap_bin_values_production_test.std")
    m.add("sdPolyOrder", f_int(sd["inputs"]["polynomial_order"]), sd_p + ":inputs.polynomial_order")
    m.add("sdClipSigma", f_num(sd["inputs"]["sigma_clip"], 1), sd_p + ":inputs.sigma_clip")
    m.add("sdClipIters", f_int(sd["inputs"]["sigma_clip_iters"]), sd_p + ":inputs.sigma_clip_iters")
    m.add("sdNBins", f_int(sd["inputs"]["n_bins"]), sd_p + ":inputs.n_bins")
    one = sd["one_feature_classifier"]
    m.add("oneAcc", f_num(one["test_accuracy_3class"]), sd_p + ":one_feature_classifier.test_accuracy_3class")
    m.add("oneBal", f_num(one["test_balanced_accuracy_3class"]), sd_p + ":one_feature_classifier.test_balanced_accuracy_3class")
    m.add("oneAucGK", f_num(one["roc_auc_g_vs_k_raw_median"], 2), sd_p + ":one_feature_classifier.roc_auc_g_vs_k_raw_median")
    m.add("oneAucFG", f_num(one["roc_auc_f_vs_g_raw_median"], 2), sd_p + ":one_feature_classifier.roc_auc_f_vs_g_raw_median")
    rm = sd["row_median_distributions"]
    for tag, key in (("Prod", "production"), ("Poly", "polynomial_n5")):
        for c in CLASSES:
            m.add(f"med{tag}{c}", f_num(rm[key]["per_class_median_of_row_medians"][c]), f"{sd_p}:row_median_distributions.{key}.per_class_median_of_row_medians.{c}")
        for pair, code in (("F-G", "FG"), ("G-K", "GK"), ("F-K", "FK")):
            m.add(f"ks{tag}{code}", f_p(rm[key]["ks_pairwise"][pair]["p_value"]), f"{sd_p}:row_median_distributions.{key}.ks_pairwise.{pair}.p_value")
    pm_ = rm["production"]["per_class_median_of_row_medians"]
    pp_ = rm["polynomial_n5"]["per_class_median_of_row_medians"]
    m.add("spreadProd", f_num(100 * (pm_["F"] - pm_["K"]), 1), sd_p + ":100*(F - K) per-class medians, production, percentage points")
    m.add("spreadPoly", f_num(100 * abs(pp_["F"] - pp_["K"]), 1), sd_p + ":100*|F - K| per-class medians, polynomial_n5, percentage points")
    m.add("sdGtoKPoly", f_int(iv["polynomial_n5"]["g_to_k"]), sd_p + ":polynomial_n5.g_to_k")
    cmp_ = iv["polynomial_n5"]["confusion_matrix"]
    m.add("sdPolyGstayPct", f_pct(cmp_[1][1] / sum(cmp_[1])), sd_p + ":polynomial_n5 G->G / G row total, percent")
    m.add("sdPolyGtoKPct", f_pct(cmp_[1][2] / sum(cmp_[1])), sd_p + ":polynomial_n5 G->K / G row total, percent")
    m.add("sdPolyGtoFPct", f_pct(cmp_[1][0] / sum(cmp_[1])), sd_p + ":polynomial_n5 G->F / G row total, percent")
    m.add("sdPolyGtoF", f_int(cmp_[1][0]), sd_p + ":polynomial_n5 G->F")
    m.add("sdPolyGstay", f_int(cmp_[1][1]), sd_p + ":polynomial_n5 G->G")
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            m.add(f"polyConf{ci}{cj}", f_int(cmp_[i][j]), f"{sd_p}:polynomial_n5.confusion_matrix[{i}][{j}]")

    # retraining
    rt_p = "artifacts/revision/retrain_normalisation.json"
    rt = load_json(rt_p)
    rcode = {"production": "Prod", "polynomial_n5": "Poly", "polynomial_n5_gap_restored": "PolyGap", "row_median": "RowMed"}
    for key, code in rcode.items():
        r = rt["retrained"][key]
        t = r["test"]
        m.add(f"rt{code}MacroF", f_num(t["macro_f1"]), f"{rt_p}:retrained.{key}.test.macro_f1")
        for c in CLASSES:
            m.add(f"rt{code}Rec{c}", f_num(t["per_class_recall"][c]), f"{rt_p}:retrained.{key}.test.per_class_recall.{c}")
        m.add(f"rt{code}GtoK", f_int(t["g_to_k"]), f"{rt_p}:retrained.{key}.test.g_to_k")
        m.add(f"rt{code}Iter", f_int(t["best_iteration"]), f"{rt_p}:retrained.{key}.test.best_iteration")
        ab = r["ablation_Mg_b"]
        for cls in ("G", "K"):
            a = ab[f"Mg_b_{cls}"]
            m.add(f"rt{code}Mg{cls}Delta", f_num(a["delta"], 3, signed=True), f"{rt_p}:retrained.{key}.ablation_Mg_b.Mg_b_{cls}.delta")
            m.add(f"rt{code}Mg{cls}CiLo", f_num(a["delta_ci_low"], 3, signed=True), f"{rt_p}:retrained.{key}.ablation_Mg_b.Mg_b_{cls}.delta_ci_low")
            m.add(f"rt{code}Mg{cls}CiHi", f_num(a["delta_ci_high"], 3, signed=True), f"{rt_p}:retrained.{key}.ablation_Mg_b.Mg_b_{cls}.delta_ci_high")
            m.add(f"rt{code}Mg{cls}P", f_p(a["p_value_vs_random"]), f"{rt_p}:retrained.{key}.ablation_Mg_b.Mg_b_{cls}.p_value_vs_random")
    cvr = rt["retrained"]["production"]["cv_train_val"]
    m.add("rtCvMean", f_num(cvr["macro_f1_mean"]), rt_p + ":retrained.production.cv_train_val.macro_f1_mean")
    m.add("rtCvSd", f_num(cvr["macro_f1_sd"]), rt_p + ":retrained.production.cv_train_val.macro_f1_sd (sample SD)")
    rep = rt["retrained"]["production"]["reproduces_deposited_model"]
    m.add("rtReproDiff", f_num(rep["abs_difference_macro_f1"], 3), rt_p + ":reproduces_deposited_model.abs_difference_macro_f1")
    m.add("rtDeltaMacroPoly", f_num(rt["retrained"]["production"]["test"]["macro_f1"] - rt["retrained"]["polynomial_n5"]["test"]["macro_f1"]),
          rt_p + ":production.macro_f1 - polynomial_n5.macro_f1")
    m.add("rtDeltaMacroRow", f_num(rt["retrained"]["production"]["test"]["macro_f1"] - rt["retrained"]["row_median"]["test"]["macro_f1"]),
          rt_p + ":production.macro_f1 - row_median.macro_f1")
    m.add("rtNControls", f_int(rt["inputs"]["n_random_controls"]), rt_p + ":inputs.n_random_controls")
    m.add("rtNBoot", f_int(rt["inputs"]["n_bootstrap"]), rt_p + ":inputs.n_bootstrap")
    hp = rt["inputs"]["hyperparameters"]
    m.add("hpDepth", f_int(hp["max_depth"]), rt_p + ":inputs.hyperparameters.max_depth")
    m.add("hpLeaves", f_int(hp["num_leaves"]), rt_p + ":inputs.hyperparameters.num_leaves")
    m.add("hpLr", f_num(hp["learning_rate"], 2), rt_p + ":inputs.hyperparameters.learning_rate")
    m.add("hpNEst", f_int(hp["n_estimators"]), rt_p + ":inputs.hyperparameters.n_estimators")
    m.add("hpEarly", f_int(rt["inputs"]["early_stopping_rounds"]), rt_p + ":inputs.early_stopping_rounds")
    m.add("hpMinChild", f_int(hp["min_child_samples"]), rt_p + ":inputs.hyperparameters.min_child_samples")
    m.add("hpSub", f_num(hp["subsample"], 1), rt_p + ":inputs.hyperparameters.subsample")
    m.add("hpCol", f_num(hp["colsample_bytree"], 1), rt_p + ":inputs.hyperparameters.colsample_bytree")
    m.add("hpSeed", f_int(hp["random_state"]), rt_p + ":inputs.hyperparameters.random_state")
    m.add("bestIter", f_int(rt["deposited_model_test"]["best_iteration"]), rt_p + ":deposited_model_test.best_iteration")

    # luminosity mix
    lm_p = "artifacts/revision/luminosity_mix.json"
    lm = load_json(lm_p)
    mix = lm["luminosity_mix"]
    for c in CLASSES:
        a, t = mix["all"][c], mix["test"][c]
        m.add(f"lumN{c}", f_int(a["n"]), f"{lm_p}:luminosity_mix.all.{c}.n")
        m.add(f"lumDwarfPct{c}", f_pct(a["dwarf_fraction"]), f"{lm_p}:luminosity_mix.all.{c}.dwarf_fraction, percent")
        m.add(f"lumGiantPct{c}", f_pct(1 - a["dwarf_fraction"]), f"{lm_p}:1 - luminosity_mix.all.{c}.dwarf_fraction, percent")
        m.add(f"lumTestDwarfPct{c}", f_pct(t["dwarf_fraction"]), f"{lm_p}:luminosity_mix.test.{c}.dwarf_fraction, percent")
        m.add(f"lumLogg{c}", f_num(a["median_logg"], 2), f"{lm_p}:luminosity_mix.all.{c}.median_logg")
        m.add(f"lumTeff{c}", f_num(a["median_teff_k"], 0), f"{lm_p}:luminosity_mix.all.{c}.median_teff_k")
        m.add(f"lumFeh{c}", f_num(a["median_feh"], 2), f"{lm_p}:luminosity_mix.all.{c}.median_feh")
    m.add("lumNAll", f_int(mix["all"]["ALL"]["n"]), lm_p + ":luminosity_mix.all.ALL.n")
    m.add("lumNDwarfKtest", f_int(mix["test"]["K"]["n_dwarf"]), lm_p + ":luminosity_mix.test.K.n_dwarf")
    m.add("lumNGiantKtest", f_int(mix["test"]["K"]["n"] - mix["test"]["K"]["n_dwarf"]), lm_p + ":luminosity_mix.test.K n - n_dwarf")
    m.add("lumLoggCut", f_num(3.5, 1), lm_p + ":inputs.dwarf_definition (log g > 3.5)")
    acc = lm["test_accuracy_by_luminosity_class"]
    for c in CLASSES:
        for lum, tag in (("dwarf", "Dwarf"), ("giant", "Giant")):
            d = acc[c][lum]
            m.add(f"lumAcc{c}{tag}", f_num(d["accuracy"]), f"{lm_p}:test_accuracy_by_luminosity_class.{c}.{lum}.accuracy")
            m.add(f"lumAccN{c}{tag}", f_int(d["n"]), f"{lm_p}:test_accuracy_by_luminosity_class.{c}.{lum}.n")
    gk = acc["G_to_K_errors"]
    m.add("gkErrN", f_int(gk["n"]), lm_p + ":test_accuracy_by_luminosity_class.G_to_K_errors.n")
    m.add("gkErrDwarf", f_int(gk["n_dwarf"]), lm_p + ":test_accuracy_by_luminosity_class.G_to_K_errors.n_dwarf")
    m.add("gkErrLogg", f_num(gk["median_logg"], 2), lm_p + ":test_accuracy_by_luminosity_class.G_to_K_errors.median_logg")
    m.add("gkErrTeff", f_num(gk["median_teff_k"], 0), lm_p + ":test_accuracy_by_luminosity_class.G_to_K_errors.median_teff_k")
    bd = lm["test_boundary_distance"]
    for c in CLASSES:
        m.add(f"bdMed{c}", f_num(bd[c]["median_boundary_distance_k"], 0), f"{lm_p}:test_boundary_distance.{c}.median_boundary_distance_k")
        m.add(f"bdNear{c}", f_int(bd[c]["n_within_150K"]), f"{lm_p}:test_boundary_distance.{c}.n_within_150K")
        m.add(f"bdAccNear{c}", f_num(bd[c]["accuracy_within_150K"]), f"{lm_p}:test_boundary_distance.{c}.accuracy_within_150K")
        m.add(f"bdAccFar{c}", f_num(bd[c]["accuracy_beyond_150K"]), f"{lm_p}:test_boundary_distance.{c}.accuracy_beyond_150K")
    m.add("bdCut", f_int(lm["inputs"]["boundary_near_threshold_k"]), lm_p + ":inputs.boundary_near_threshold_k")

    # template cross-check
    bm_p = "artifacts/revision/benchmark_corrected.json"
    bm = load_json(bm_p)
    pmeth = bm["per_method"]
    mcode = {"median_filter_200": "MfTwo", "median_filter_50": "MfFifty", "polynomial_n5": "PolyN", "polynomial_n5_window": "PolyWin"}
    for key, code in mcode.items():
        d = pmeth[key]
        m.add(f"pkN{code}", f_int(d["n_compared_fgk"]), f"{bm_p}:per_method.{key}.n_compared_fgk")
        agr = d["agreement_rate_model_vs_pickles"]
        m.add(f"pkAgree{code}", f_num(agr) if agr is not None else r"\textbf{n/a}", f"{bm_p}:per_method.{key}.agreement_rate_model_vs_pickles")
        m.add(f"pkSuper{code}", f_pct(d["fraction_best_template_supergiant"], 0), f"{bm_p}:per_method.{key}.fraction_best_template_supergiant, percent")
        f1v = d["macro_f1_fgk"]
        m.add(f"pkMacro{code}", f_num(f1v) if f1v is not None else r"\textbf{n/a}", f"{bm_p}:per_method.{key}.macro_f1_fgk")
    m.add("pkNTemplates", f_int(bm["inputs"]["n_templates"]), bm_p + ":inputs.n_templates")
    m.add("pkGate", f_num(0.55, 2), "pre-specified agreement gate (decision log, benchmark entry)")
    m.add("pkGateMinN", f_int(100), "pre-specified minimum n_compared")
    m.add("pkEmbedAgree", f_int(bm["embedded_map_vs_header"]["n_agree"]), bm_p + ":embedded_map_vs_header.n_agree")
    m.add("pkAnyPass", "yes" if bm["any_method_passes_gate"] else "no", bm_p + ":any_method_passes_gate")
    for c in CLASSES:
        m.add(f"pkPrecMfTwo{c}", f_num(pmeth["median_filter_200"]["per_class_precision"][c]), f"{bm_p}:per_method.median_filter_200.per_class_precision.{c}")
        m.add(f"pkRecMfTwo{c}", f_num(pmeth["median_filter_200"]["per_class_recall"][c]), f"{bm_p}:per_method.median_filter_200.per_class_recall.{c}")
        m.add(f"pkPrecMfFifty{c}", f_num(pmeth["median_filter_50"]["per_class_precision"][c]), f"{bm_p}:per_method.median_filter_50.per_class_precision.{c}")
        m.add(f"pkRecMfFifty{c}", f_num(pmeth["median_filter_50"]["per_class_recall"][c]), f"{bm_p}:per_method.median_filter_50.per_class_recall.{c}")


# --------------------------------------------------------------------------- group B
def _row(rows: list[dict[str, Any]], line_set: str, cls: str) -> dict[str, Any] | None:
    for r in rows:
        if r["line_set"] == line_set and r["mk_class"] == cls:
            return r
    return None


def _tbd_row(cols: int) -> str:
    return rf"\multicolumn{{{cols}}}{{c}}{{{TBD}}} \\" + "\n"


def group_b(m: Macros, out_dir: Path) -> None:
    rec_p = "artifacts/revision/ablation_recalibrated.json"
    str_p = "artifacts/revision/ablation_stratified.json"
    rec = load_json(rec_p) if (REPO_ROOT / rec_p).exists() else None
    strat = load_json(str_p) if (REPO_ROOT / str_p).exists() else None

    # ---- scalar settings
    def scalar(name: str, getter, kind: str, src: str, payload) -> None:
        if payload is None:
            m.add_tbd(name, src)
            return
        m.add(name, fmt(getter(payload), kind), src)

    scalar("ablNBoot", lambda p: p["inputs"]["n_bootstrap"], "int", rec_p + ":inputs.n_bootstrap", rec)
    scalar("ablNControls", lambda p: p["inputs"]["n_random_controls"], "int", rec_p + ":inputs.n_random_controls", rec)
    scalar("ablFloor", lambda p: 1.0 / (min(d["succeeded"] for d in p["runs"][p["primary_run"]]["draw_stats"].values()) + 1), "p",
           rec_p + ":1/(min succeeded draws + 1), primary run", rec)
    scalar("ablFillTrainProd", lambda p: p["inputs"]["continuum_fill_train_median_production"], "f4", rec_p + ":inputs.continuum_fill_train_median_production", rec)
    scalar("ablFillTrainPoly", lambda p: p["inputs"]["continuum_fill_train_median_polynomial_n5"], "f4", rec_p + ":inputs.continuum_fill_train_median_polynomial_n5", rec)
    scalar("ablGateStatus", lambda p: p["headline_gate"]["gate_status"], "str", rec_p + ":headline_gate.gate_status", rec)
    scalar("ablGatePassed", lambda p: p["headline_gate"]["n_pairs_passed"], "int", rec_p + ":headline_gate.n_pairs_passed", rec)

    # ---- per run, per (line set, class) macros
    for run, rp in RUN_PREFIX.items():
        rows = rec["runs"][run]["rows"] if rec else None
        for ls in SET_ORDER:
            for cls in CLASSES:
                row = _row(rows, ls, cls) if rows else None
                for suf, (key, kind) in ABL_FIELDS.items():
                    name = f"abl{rp}{SET_CODE[ls]}{cls}{suf}"
                    src = f"{rec_p}:runs.{run}.rows[{ls},{cls}].{key}"
                    if row is None:
                        m.add_tbd(name, src)
                    elif kind == "nullb":
                        nb = round(row["p_value_vs_random"] * (row["n_random_controls_succeeded"] + 1)) - 1
                        m.add(name, f_int(nb), f"{rec_p}:runs.{run}.rows[{ls},{cls}]: round(p_value_vs_random*(n_random_controls_succeeded+1)) - 1")
                    else:
                        m.add(name, fmt(row[key], kind), src)
    # K-class summaries over the primary run, baseline accuracy, lost-row destinations
    if rec:
        prim = rec["runs"][rec["primary_run"]]["rows"]
        krows = [r for r in prim if r["mk_class"] == "K"]
        m.add("ablKMaxDrop", f_num(max(-r["delta_acc_mean"] for r in krows)), rec_p + ":max(-delta_acc_mean) over K rows, primary run")
        m.add("ablKMaxRise", f_num(max(r["delta_acc_mean"] for r in krows)), rec_p + ":max(delta_acc_mean) over K rows, primary run")
        m.add("ablKMaxLost", f_int(max(r["n_flip_lost"] for r in krows)), rec_p + ":max n_flip_lost over K rows, primary run")
        m.add("ablKMaxGained", f_int(max(r["n_flip_gained"] for r in krows)), rec_p + ":max n_flip_gained over K rows, primary run")
        m.add("ablKMaxUpper", f_num(max(r["flip_rate_upper95"] for r in krows)), rec_p + ":max flip_rate_upper95 over K rows, primary run")
        for run, rp in RUN_PREFIX.items():
            dest = rec["lost_row_destinations_focus_pairs"][run]
            for tag, key in (("MgG", "Mg_b__G"), ("HbF", "H_balmer__F")):
                for c, n in dest[key].items():
                    m.add(f"ablDest{rp}{tag}To{c}", f_int(n), f"{rec_p}:lost_row_destinations_focus_pairs.{run}.{key}.{c}")
        m.add("ablBaseAccPrim", f_num(rec["runs"]["production__gp_interp"]["baseline_accuracy"]), rec_p + ":runs.production__gp_interp.baseline_accuracy")
        m.add("ablBaseAccPoly", f_num(rec["runs"]["polynomial_n5__gp_interp"]["baseline_accuracy"]), rec_p + ":runs.polynomial_n5__gp_interp.baseline_accuracy")
        m.add("ablBaseRecGPoly", f_num(rec["runs"]["polynomial_n5__gp_interp"]["per_class_baseline_recall"]["G"]), rec_p + ":runs.polynomial_n5__gp_interp.per_class_baseline_recall.G")
        m.add("ablRuntimePrim", f_int(round(rec["runs"]["production__gp_interp"]["runtime_s"])), rec_p + ":runs.production__gp_interp.runtime_s")
        m.add("ablRuntimeTotal", f_int(round(sum(r["runtime_s"] for r in rec["runs"].values()))), rec_p + ":sum of runs.*.runtime_s")
    else:
        for n in ("ablKMaxDrop", "ablKMaxRise", "ablKMaxLost", "ablKMaxGained", "ablKMaxUpper",
                  "ablBaseAccPrim", "ablBaseAccPoly", "ablBaseRecGPoly", "ablRuntimePrim", "ablRuntimeTotal"):
            m.add_tbd(n, rec_p)
        for rp in RUN_PREFIX.values():
            for n in (f"ablDest{rp}MgGToF", f"ablDest{rp}MgGToK", f"ablDest{rp}HbFToG", f"ablDest{rp}HbFToK"):
                m.add_tbd(n, rec_p)

    # ---- stratified macros (primary run)
    pairs = [(ls, cls) for ls in SET_ORDER for cls in CLASSES]
    prun = strat["inputs"]["primary_run"] if strat else None
    for ls, cls in pairs:
        pair_d = strat["runs"][prun]["pairs"].get(f"{ls}__{cls}") if strat else None
        for (stratum, part), tag in STRATA_PARTS.items():
            for suf, (key, kind) in STR_FIELDS.items():
                name = f"str{SET_CODE[ls]}{cls}{tag}{suf}"
                src = f"{str_p}:runs.{prun}.pairs.{ls}__{cls}.{stratum}.{part}.{key}"
                if pair_d is None:
                    m.add_tbd(name, src)
                else:
                    cell = pair_d[stratum][part]
                    if cell.get("n", 0) == 0 or key not in cell:
                        m.add(name, r"\textbf{--}" if suf != "N" else "0", src)
                    else:
                        m.add(name, fmt(cell[key], kind), src)
    if strat:
        foc = strat["focus"]["Mg_b__G_lost_rows_primary"]
        m.add("focusMgGLost", f_int(foc["n_lost"]), str_p + ":focus.Mg_b__G_lost_rows_primary.n_lost")
        m.add("focusMgGTeff", f_num(foc["median_teff_k"], 0), str_p + ":focus.Mg_b__G_lost_rows_primary.median_teff_k")
        m.add("focusMgGToF", f_int(foc["destination_counts"]["F"]), str_p + ":focus.Mg_b__G_lost_rows_primary.destination_counts.F")
        m.add("focusMgGToK", f_int(foc["destination_counts"]["K"]), str_p + ":focus.Mg_b__G_lost_rows_primary.destination_counts.K")
        m.add("focusMgGNear", f_int(foc["n_within_150K"]), str_p + ":focus.Mg_b__G_lost_rows_primary.n_within_150K")
        m.add("focusMgGDwarf", f_int(foc["n_dwarf"]), str_p + ":focus.Mg_b__G_lost_rows_primary.n_dwarf")
        kf = strat["focus"]["K_class"]
        m.add("focusKNearInterior", f_int(kf["n_within_150K_interior"]), str_p + ":focus.K_class.n_within_150K_interior")
        m.add("focusKMedInterior", f_num(kf["median_boundary_distance_interior_k"], 0), str_p + ":focus.K_class.median_boundary_distance_interior_k")
        # K rows that change class under any line set (primary run), by luminosity and boundary distance
        kl: dict[int, dict[str, Any]] = {}
        k_giant_lost = k_giant_gained = 0
        for ls in SET_ORDER:
            pr = strat["runs"][prun]["pairs"][f"{ls}__K"]
            for r in pr["lost_rows"]:
                kl[r["feature_index"]] = r
            k_giant_lost += pr["luminosity"]["giant"].get("n_flip_lost", 0)
            k_giant_gained += pr["luminosity"]["giant"].get("n_flip_gained", 0)
        m.add("kLostDistinct", f_int(len(kl)), str_p + ":distinct feature_index over pairs *__K lost_rows, primary run")
        m.add("kLostDwarf", f_int(sum(1 for r in kl.values() if r["luminosity"] == "dwarf")), str_p + ":distinct K lost rows that are dwarfs")
        m.add("kLostNear", f_int(sum(1 for r in kl.values() if r["within_150K"])), str_p + ":distinct K lost rows within 150 K of the edge of their own bin")
        m.add("kGiantLost", f_int(k_giant_lost), str_p + ":sum over line sets of luminosity.giant.n_flip_lost, K rows")
        m.add("kGiantGained", f_int(k_giant_gained), str_p + ":sum over line sets of luminosity.giant.n_flip_gained, K rows")
    else:
        for n in ("focusMgGLost", "focusMgGTeff", "focusMgGToF", "focusMgGToK", "focusMgGNear", "focusMgGDwarf", "focusKNearInterior",
                  "focusKMedInterior", "kLostDistinct", "kLostDwarf", "kLostNear", "kGiantLost", "kGiantGained"):
            m.add_tbd(n, str_p)

    # ---- tables
    def cell(r: dict[str, Any] | None, key: str, kind: str) -> str:
        return fmt(r[key], kind) if r else TBD

    # main table (primary run)
    lines = [
        "% Generated by scripts/revision/manuscript_numbers.py from artifacts/revision/ablation_recalibrated.json",
        r"\begin{tabular}{llrrcccrr}",
        r"\toprule",
        r"Line set & Class & $n$ & $\dA$ & 95\% CI & Flips (lost/gained) & $p_{\mathrm{McN}}$ & $\Delta\bar{P}_{\mathrm{true}}$ & $p_{\mathrm{MC}}$ \\",
        r"\midrule",
    ]
    if rec:
        rows = rec["runs"][rec["primary_run"]]["rows"]
        for i, ls in enumerate(SET_ORDER):
            for j, cls in enumerate(CLASSES):
                r = _row(rows, ls, cls)
                first = SET_TEX[ls] if j == 0 else ""
                ci = f"[{r['delta_acc_ci_low']:+.3f}, {r['delta_acc_ci_high']:+.3f}]"
                lines.append(
                    f"{first} & {cls} & {r['n_test']} & {fmt(r['delta_acc_mean'], 'sf3')} & {_ens(ci)} & "
                    f"{r['n_flip_lost']}/{r['n_flip_gained']} & {fmt(r['mcnemar_exact_p'], 'p')} & "
                    f"{fmt(r['mean_delta_true_prob'], 'sf3')} & {fmt(r['p_value_vs_random'], 'p')} \\\\"
                )
            if i < len(SET_ORDER) - 1:
                lines.append(r"\addlinespace")
    else:
        lines.append(_tbd_row(9).rstrip("\n"))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write_lf(out_dir / "ablation_table.tex", "\n".join(lines))

    # K-class table over the 2 x 2 design
    lines = [
        "% Generated by scripts/revision/manuscript_numbers.py from artifacts/revision/ablation_recalibrated.json",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Line set & Prod., interp. & Prod., constant & Poly., interp. & Poly., constant \\",
        r"\midrule",
    ]
    if rec:
        for ls in SET_ORDER:
            cells = []
            for run in RUN_PREFIX:
                r = _row(rec["runs"][run]["rows"], ls, "K")
                cells.append(f"{fmt(r['delta_acc_mean'], 'sf3')} ({r['n_flip_lost']}/{r['n_flip_gained']})")
            lines.append(f"{SET_TEX[ls]} & " + " & ".join(cells) + r" \\")
    else:
        lines.append(_tbd_row(5).rstrip("\n"))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write_lf(out_dir / "ablation_k_table.tex", "\n".join(lines))

    # strata table
    lines = [
        "% Generated by scripts/revision/manuscript_numbers.py from artifacts/revision/ablation_stratified.json",
        r"\begin{tabular}{llrrcr}",
        r"\toprule",
        r"Pair & Stratum & $n$ & $\dA$ [95\% CI] & Flips (l/g) & $\Delta\bar{P}_{\mathrm{true}}$ \\",
        r"\midrule",
    ]
    focus_pairs = [("H_balmer", "F"), ("H_balmer", "G"), ("H_balmer", "K"), ("Mg_b", "G"), ("Mg_b", "K"), ("Na_D", "K"), ("Ca_I", "K"), ("Fe_Cr", "K")]
    if strat:
        for k, (ls, cls) in enumerate(focus_pairs):
            pair_d = strat["runs"][prun]["pairs"][f"{ls}__{cls}"]
            first = True
            for (stratum, part), tag in STRATA_PARTS.items():
                c = pair_d[stratum][part]
                label = f"{SET_TEX[ls]}, {cls}" if first else ""
                first = False
                part_tex = {"dwarf": "dwarfs", "giant": "giants", "within_150K": r"$\leq 150$ K from boundary", "beyond_150K": r"$> 150$ K from boundary"}[part]
                if c.get("n", 0) == 0 or "delta_recall" not in c:
                    lines.append(f"{label} & {part_tex} & 0 & -- & -- & -- \\\\")
                else:
                    ci = f"[{c['delta_ci_low']:+.3f}, {c['delta_ci_high']:+.3f}]"
                    lines.append(
                        f"{label} & {part_tex} & {c['n']} & {fmt(c['delta_recall'], 'sf3')} {_ens(ci)} & "
                        f"{c['n_flip_lost']}/{c['n_flip_gained']} & {fmt(c['mean_delta_true_prob'], 'sf3')} \\\\"
                    )
            if k < len(focus_pairs) - 1:
                lines.append(r"\addlinespace")
    else:
        lines.append(_tbd_row(6).rstrip("\n"))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write_lf(out_dir / "ablation_strata_table.tex", "\n".join(lines))


# --------------------------------------------------------------------------- output
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "submission")
    p.add_argument("--index", type=Path, default=REPO_ROOT / "artifacts/revision/manuscript_numbers_index.json")
    args = p.parse_args(argv)

    m = Macros()
    group_a(m)
    n_a = len(m.defs)
    group_b(m, args.out_dir)

    header = [
        "% Generated by scripts/revision/manuscript_numbers.py. Do not edit by hand.",
        "% Group A: values read from artifacts/*.json (classifier, normalisation, luminosity, template cross-check).",
        r"% Group B (macros starting abl, str, focus): values read from artifacts/revision/ablation_recalibrated.json",
        "% and ablation_stratified.json.",
        f"% {len(m.defs)} macros, {len(m.tbd)} unresolved.",
    ]
    if m.tbd:  # fallback so the manuscript still compiles while the ablation artifacts are absent
        header.append(r"\providecommand{\TBDablation}{\textbf{[TBD-ablation]}}")
    body = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.defs.items()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_lf(args.out_dir / "revision_numbers.tex", "\n".join(header + body) + "\n")
    write_lf(args.out_dir / "line_sets_table.tex",
             (REPO_ROOT / "artifacts/revision/line_sets_table.tex").read_text(encoding="utf-8").replace("\r\n", "\n"))
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("w", encoding="utf-8") as f:
        json.dump({"n_macros": len(m.defs), "n_group_a": n_a, "n_placeholders": len(m.tbd),
                   "placeholders": m.tbd, "macros": m.source}, f, indent=1)
    print(f"wrote {args.out_dir / 'revision_numbers.tex'}: {len(m.defs)} macros ({len(m.tbd)} placeholders)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
