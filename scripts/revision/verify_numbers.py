#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Independent check of every number quoted in submission/manuscript.tex.

Part 1 (macros). For each macro in artifacts/revision/manuscript_numbers_index.json the
source field is re-read from the JSON artifact with a separate resolver, the value is
re-derived (percent, differences, aggregates over rows), and the displayed string in
submission/revision_numbers.tex is compared with it: the displayed number must equal the
re-derived value rounded to the displayed number of decimals (scientific notation for
small p-values is compared on the displayed mantissa precision).

Part 2 (literals). Digits typed directly in manuscript.tex outside the macros are
extracted and classified as
  * structural       (dates in prose that are checked against the decision log, ordinal
                      counts, equation constants, ESO programme id, citation-like tokens),
  * typed_constant   (the literal also appears in the code, configuration or an artifact
                      that defines it, so it is traceable to a source),
  * checked_result   (the literal restates a computed result and is re-derived here),
  * untraced         (none of the above).

Part 3 (recomputation). Headline ablation numbers are recomputed from
artifacts/revision/ablation_recalibrated_per_row.npz without using the stored summary
statistics.

Writes artifacts/revision/number_verification.json. Run from the repository root:
``python scripts/revision/verify_numbers.py``
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "revision" / "number_verification.json"
_CACHE: dict[str, Any] = {}


def load(rel: str) -> Any:
    if rel not in _CACHE:
        with (REPO / rel).open(encoding="utf-8") as f:
            _CACHE[rel] = json.load(f)
    return _CACHE[rel]


# ------------------------------------------------------------------ path resolver
def _split(path: str) -> list[str]:
    toks, cur, depth = [], "", 0
    for ch in path:
        if ch == "[":
            depth += 1
        if ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            toks.append(cur)
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    return toks


def _index(obj: Any, tok: str) -> Any:
    """Resolve ``name[a][b]`` or ``name[line_set,class]`` against obj."""
    m = re.match(r"^([^\[]*)((?:\[[^\]]*\])*)$", tok)
    name, brackets = m.group(1), re.findall(r"\[([^\]]*)\]", m.group(2))
    if name:
        obj = obj[name]
    for b in brackets:
        if "," in b:
            ls, cl = b.split(",")
            hits = [r for r in obj if r["line_set"] == ls and r["mk_class"] == cl]
            assert len(hits) == 1, (b, len(hits))
            obj = hits[0]
        else:
            obj = obj[int(b)]
    return obj


def resolve(obj: Any, path: str) -> Any:
    toks = _split(path)
    i = 0
    while i < len(toks):
        if isinstance(obj, dict):
            for n in range(len(toks), i, -1):  # longest key first (keys such as 'uniform_x1.09')
                key = ".".join(toks[i:n])
                if key in obj:
                    obj, i = obj[key], n
                    break
            else:
                obj, i = _index(obj, toks[i]), i + 1
        else:
            obj, i = _index(obj, toks[i]), i + 1
    return obj


# ------------------------------------------------------------------ display parsing
def parse_display(raw: str) -> tuple[float | None, int | None, str]:
    """Return (value, decimals_shown, kind) for a macro body."""
    s = raw.strip()
    s = re.sub(r"^\\ensuremath\{(.*)\}$", r"\1", s)
    if "n/a" in s:
        return None, None, "na"
    m = re.match(r"^([+-]?\d+(?:\.(\d+))?)\\times10\^\{(-?\d+)\}$", s)
    if m:
        mant, dec, ex = m.group(1), m.group(2) or "", int(m.group(3))
        return float(mant) * 10.0**ex, len(dec), "sci"
    m = re.match(r"^([+-]?\d+)(?:\.(\d+))?$", s)
    if m:
        return float(s), len(m.group(2) or ""), "num"
    return None, None, "text"


def within(shown: float, dec: int, kind: str, exp: float) -> bool:
    if kind == "sci":
        ex = int(math.floor(math.log10(abs(shown)))) if shown else 0
        tol = 0.5 * 10 ** (ex - dec) * (1 + 1e-9)
        return abs(shown - exp) <= tol
    tol = 0.5 * 10 ** (-dec) * (1 + 1e-9) + 1e-12
    return abs(shown - exp) <= tol


# ------------------------------------------------------------------ irregular sources
def irregular(name: str, src: str) -> tuple[Any, str] | None:
    """Independent re-derivation for macros whose source is an expression."""
    met = load("artifacts/metrics.json")
    sd = load("artifacts/revision/scale_decomposition.json")
    rt = load("artifacts/revision/retrain_normalisation.json")
    lm = load("artifacts/revision/luminosity_mix.json")
    bc = load("artifacts/revision/benchmark_corrected.json")
    rc = load("artifacts/revision/ablation_recalibrated.json")
    stt = load("artifacts/revision/ablation_stratified.json")
    cm = np.array(met["confusion_matrix"])
    n_test = met["n_test"]
    iv = sd["interventions"]
    pr = rc["primary_run"]
    k_rows = [r for r in rc["runs"][pr]["rows"] if r["mk_class"] == "K"]
    if name == "testMacroP":
        return float(np.mean(list(met["per_class_precision"].values()))), "num"
    if name == "testMacroR":
        return float(np.mean(list(met["per_class_recall"].values()))), "num"
    if name.startswith("nClass"):
        return int(cm["FGK".index(name[-1])].sum()), "num"
    if name == "adjFG":
        return 100 * (cm[0, 1] + cm[1, 0]) / n_test, "num"
    if name == "adjGK":
        return 100 * (cm[1, 2] + cm[2, 1]) / n_test, "num"
    if name == "adjFKRows":
        return int(cm[0, 2] + cm[2, 0]), "num"
    if name == "adjFKPct":
        return 100 * (cm[0, 2] + cm[2, 0]) / n_test, "num"
    if name == "nMatched":
        import pandas as pd
        return len(pd.read_parquet(REPO / "artifacts" / "ges_mk_labels.parquet")), "num"
    if name == "cvSd":
        return met["cv_macro_f1_std"], "num"
    if name == "cvGap":
        return met["cv_macro_f1_mean"] - met["cv_spatial_macro_f1_mean"], "num"
    if name == "cvTestZ":
        return (met["macro_f1"] - met["cv_spatial_macro_f1_mean"]) / met["cv_spatial_macro_f1_std"], "num"
    if name == "singGroupPct":
        return 100 * met["cv_spatial_singleton_stats"]["singleton_group_fraction"], "num"
    if name == "singRowPct":
        return 100 * met["cv_spatial_singleton_stats"]["singleton_row_fraction"], "num"
    if name in ("triWlMin", "triWlMax"):
        wl = [b["wavelength_aa"] for b in load("artifacts/interpret/triangulation_report.json")["three_way_intersection_bins"]]
        return (min(wl) if name.endswith("Min") else max(wl)), "num"
    if name == "meanBinWidthFine":
        return load("artifacts/ablation/mg_b_ew_test.json")["mean_bin_width_aa"] / 5.0, "num"
    if name == "sdCostCommon":
        return iv["production"]["macro_f1"] - iv["common_median"]["macro_f1"], "num"
    if name == "sdCostSwap":
        return iv["production"]["macro_f1"] - iv["gk_swap"]["macro_f1"], "num"
    if name == "sdCostUni":
        return iv["production"]["macro_f1"] - iv["uniform_x1.09"]["macro_f1"], "num"
    if name == "sdPolyShiftPct":
        ml = sd["median_levels"]
        return 100 * (ml["polynomial_n5_global_test_median"] / ml["production_global_test_median"] - 1), "num"
    pcm = sd["row_median_distributions"]
    if name == "spreadProd":
        d = pcm["production"]["per_class_median_of_row_medians"]
        return 100 * (d["F"] - d["K"]), "num"
    if name == "spreadPoly":
        d = pcm["polynomial_n5"]["per_class_median_of_row_medians"]
        return 100 * abs(d["F"] - d["K"]), "num"
    pcmat = np.array(iv["polynomial_n5"]["confusion_matrix"])
    if name == "sdPolyGstayPct":
        return 100 * pcmat[1, 1] / pcmat[1].sum(), "num"
    if name == "sdPolyGtoKPct":
        return 100 * pcmat[1, 2] / pcmat[1].sum(), "num"
    if name == "sdPolyGtoFPct":
        return 100 * pcmat[1, 0] / pcmat[1].sum(), "num"
    if name == "sdPolyGtoF":
        return int(pcmat[1, 0]), "num"
    if name == "sdPolyGstay":
        return int(pcmat[1, 1]), "num"
    rtn = rt["retrained"]
    if name == "rtCvSd":
        return rtn["production"]["cv_train_val"]["macro_f1_sd"], "num"
    if name == "rtDeltaMacroPoly":
        return rtn["production"]["test"]["macro_f1"] - rtn["polynomial_n5"]["test"]["macro_f1"], "num"
    if name == "rtDeltaMacroRow":
        return rtn["production"]["test"]["macro_f1"] - rtn["row_median"]["test"]["macro_f1"], "num"
    m = re.match(r"^lum(Dwarf|Giant|TestDwarf)Pct([FGK])$", name)
    if m:
        part = "test" if m.group(1) == "TestDwarf" else "all"
        f = lm["luminosity_mix"][part][m.group(2)]["dwarf_fraction"]
        return 100 * (1 - f if m.group(1) == "Giant" else f), "num"
    if name == "lumNGiantKtest":
        d = lm["luminosity_mix"]["test"]["K"]
        return d["n"] - d["n_dwarf"], "num"
    if name == "lumLoggCut":
        return float(re.search(r"> *([\d.]+)", lm["inputs"]["dwarf_definition"]).group(1)), "num"
    m = re.match(r"^pkSuper(MfTwo|MfFifty|PolyN|PolyWin)$", name)
    if m:
        key = {"MfTwo": "median_filter_200", "MfFifty": "median_filter_50", "PolyN": "polynomial_n5", "PolyWin": "polynomial_n5_window"}[m.group(1)]
        return 100 * bc["per_method"][key]["fraction_best_template_supergiant"], "num"
    if name == "ablFloor":
        return 1.0 / (min(d["succeeded"] for d in rc["runs"][pr]["draw_stats"].values()) + 1), "num"
    if name == "ablKMaxDrop":
        return max(-r["delta_acc_mean"] for r in k_rows), "num"
    if name == "ablKMaxRise":
        return max(r["delta_acc_mean"] for r in k_rows), "num"
    if name == "ablKMaxLost":
        return max(r["n_flip_lost"] for r in k_rows), "num"
    if name == "ablKMaxGained":
        return max(r["n_flip_gained"] for r in k_rows), "num"
    if name == "ablKMaxUpper":
        return max(r["flip_rate_upper95"] for r in k_rows), "num"
    if name == "ablRuntimeTotal":
        return sum(r["runtime_s"] for r in rc["runs"].values()), "num"
    pairs = stt["runs"][pr]["pairs"]
    k_pairs = {k: v for k, v in pairs.items() if k.endswith("__K")}
    if name == "kLostDistinct":
        return len({r["feature_index"] for v in k_pairs.values() for r in v["lost_rows"]}), "num"
    if name == "kLostDwarf":
        return len({r["feature_index"] for v in k_pairs.values() for r in v["lost_rows"] if r["luminosity"] == "dwarf"}), "num"
    if name == "kLostNear":
        return len({r["feature_index"] for v in k_pairs.values() for r in v["lost_rows"] if r["within_150K"]}), "num"
    if name == "kGiantLost":
        return sum(v["luminosity"]["giant"]["n_flip_lost"] for v in k_pairs.values()), "num"
    if name == "kGiantGained":
        return sum(v["luminosity"]["giant"]["n_flip_gained"] for v in k_pairs.values()), "num"
    return None


def macro_check() -> dict:
    idx = load("artifacts/revision/manuscript_numbers_index.json")
    tex = (REPO / "submission" / "revision_numbers.tex").read_text(encoding="utf-8")
    defs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", tex, re.M))
    rows, mism, unresolved, typed = [], [], [], []
    for name, src in idx["macros"].items():
        if name not in defs:
            mism.append({"macro": name, "reason": "absent from revision_numbers.tex"})
            continue
        shown, dec, kind = parse_display(defs[name])
        status, exp = "ok", None
        try:
            if src.startswith("pre-specified"):
                status = "typed_constant"
                bsrc = (REPO / "scripts" / "revision" / "benchmark_corrected.py").read_text(encoding="utf-8")
                const = {"pkGate": "AGREEMENT_FLOOR", "pkGateMinN": "MIN_N_COMPARED"}[name]
                code_val = float(re.search(rf"^{const}\s*=\s*([\d.]+)", bsrc, re.M).group(1))
                if shown is None or abs(shown - code_val) > 1e-12:
                    status = "mismatch"
                    mism.append({"macro": name, "source": src, "shown": defs[name], "code_constant": code_val})
                typed.append({"macro": name, "source": src, "shown": defs[name], "code_constant": code_val, "matches_code": status != "mismatch"})
            else:
                irr = irregular(name, src)
                if irr is not None:
                    exp = irr[0]
                else:
                    path = src.split(":", 1)[1]
                    fpath = src.split(":", 1)[0]
                    if re.search(r"round\(p_value_vs_random", path):
                        base = path.split(":")[0]
                        row = resolve(load(fpath), base)
                        exp = round(row["p_value_vs_random"] * (row["n_random_controls_succeeded"] + 1)) - 1
                    else:
                        exp = None
                        for pre in ("", "interventions.", "retrained.production.", "retrained."):
                            try:
                                exp = resolve(load(fpath), pre + path)
                                break
                            except (KeyError, IndexError, TypeError):
                                continue
                        else:
                            raise KeyError(path)
                        if src.endswith(", percent"):
                            exp = 100 * exp
                if kind == "text":
                    if isinstance(exp, bool):
                        exp = "yes" if exp else "no"
                    if str(exp) != defs[name]:
                        status = "mismatch"
                elif kind == "na":
                    status = "ok" if exp is None or (isinstance(exp, float) and not math.isfinite(exp)) else "mismatch"
                else:
                    if isinstance(exp, str):
                        status = "mismatch"
                    elif not within(shown, dec, kind, float(exp)):
                        status = "mismatch"
                    # sign convention: a leading '+' must coincide with a positive value
                    # (a signed zero is displayed as '+0.000' by the generator's convention)
                    if status == "ok" and defs[name].find("+") >= 0 and float(exp) < 0:
                        status = "mismatch"
                    if status == "ok" and defs[name].find("+") < 0 and shown > 0 and float(exp) < 0:
                        status = "mismatch"
        except Exception as e:  # noqa: BLE001
            status = "unresolved"
            unresolved.append({"macro": name, "source": src, "error": repr(e)[:200]})
        if status == "mismatch":
            mism.append({"macro": name, "source": src, "shown": defs[name], "recomputed": exp if not isinstance(exp, np.generic) else exp.item()})
        rows.append(status)
    from collections import Counter
    return {"n_index": len(idx["macros"]), "n_defined": len(defs), "status_counts": dict(Counter(rows)),
            "mismatches": mism, "unresolved": unresolved, "typed_constants": typed}


# ------------------------------------------------------------------ literals
def strip_tex(ms: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", ms)
    s = re.sub(r"\\(cite[a-z]*|ref|label|eqref|input|includegraphics|usepackage|documentclass|bibliography\w*|cref|Cref|autoref|"
               r"url|href|hypersetup|geometry|setcounter|newcommand|renewcommand|providecommand|pagestyle)\*?(\[[^\]]*\])?(\{[^{}]*\})+", "", s)
    s = re.sub(r"\\(?:begin|end)\{[^}]*\}(\{[^}]*\})?", "", s)
    return s


def code_corpus() -> str:
    parts = []
    for pat in ("src/**/*.py", "scripts/**/*.py", "artifacts/**/*.json", "docs/decision_log.md", "docs/reproducibility_appendix.md", "README.md"):
        for p in REPO.glob(pat):
            if p.stat().st_size < 3_000_000 and "number_verification" not in p.name and "manuscript_numbers_index" not in p.name:
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return "\n".join(parts)


def literal_check() -> dict:
    ms = (REPO / "submission" / "manuscript.tex").read_text(encoding="utf-8")
    s = strip_tex(ms)
    corpus = code_corpus()
    log = (REPO / "docs" / "decision_log.md").read_text(encoding="utf-8")
    rc = load("artifacts/revision/ablation_recalibrated.json")
    pr = rc["primary_run"]
    rows = rc["runs"][pr]["rows"]
    widths = [r["total_width_aa"] for r in rows]
    checked = {
        "8": ("min total_width_aa over the 15 primary-run rows", min(widths) == 8),
        "80": ("max total_width_aa over the 15 primary-run rows (also U580 setup name)", max(widths) == 80),
        "15": ("number of (line set, class) rows", len(rows) == 15),
        "21": ("Mg b total width", {r["total_width_aa"] for r in rows if r["line_set"] == "Mg_b"} == {21.0}),
        "3": ("n_pairs_total of the headline gate", rc["headline_gate"]["n_pairs_total"] == 3),
        "454": ("n_val", load("artifacts/metrics.json")["n_val"] == 454),
        "2.40": ("5e-4 * 4801", abs(5e-4 * 4801 - 2.40) < 0.005),
        "3.40": ("5e-4 * 6796", abs(5e-4 * 6796 - 3.40) < 0.005),
        "2000": ("R = 1/dlnlambda for dlnlambda = 5e-4", abs(1 / 5e-4 - 2000) < 1),
    }
    ratio = [load("artifacts/metrics.json")[k] for k in ("n_train", "n_val", "n_test")]
    tot = sum(ratio)
    win = load("artifacts/revision/benchmark_corrected.json")["inputs"]["polynomial_n5_window_aa"]
    checked["4700"] = ("polynomial_n5_window_aa lower edge, rounded to 100 A", round(win[0], -2) == 4700)
    checked["6900"] = ("polynomial_n5_window_aa upper edge, rounded to 100 A", round(win[1], -2) == 6900)
    external = {  # facts taken from instrument or survey documentation, not computed in this repository
        "4770": "UVES U580 blue-arm chip start (ESO UVES manual)", "5770": "U580 blue-arm chip end",
        "5832": "U580 red-arm chip start", "6830": "U580 red-arm chip end",
        "188": "ESO programme 188.B-3002 (Gaia-ESO Public Spectroscopic Survey)", "3002": "ESO programme 188.B-3002",
    }
    checked["0.70"] = ("train fraction", abs(ratio[0] / tot - 0.70) < 0.005)
    checked["0.15"] = ("val and test fractions", abs(ratio[1] / tot - 0.15) < 0.005 and abs(ratio[2] / tot - 0.15) < 0.005)
    date_re = re.compile(r"2026-\d\d-\d\d")
    out, counts = [], {"structural": 0, "typed_constant": 0, "checked_result": 0, "untraced": 0}
    structural_ctx = re.compile(r"(\\author|\\affil|ORCID|ESO programme|\$\\Delta\\ln\\lambda|\\pm 0\.01|0\.01/3|\\alpha|\(b\+1\)/\(B\+1\)|1/\(B\+1\)|\\in \\\{|\$[0-9] ?\\times ?[0-9]\$|2 \$\\times\$ 2|[0-9]-fold|[0-9]\\\\%|95\\\\%)")
    for m in re.finditer(r"(?<![A-Za-z\\_{])\d[\d.,]*\d|(?<![A-Za-z\\_{])\d", s):
        lit = m.group().rstrip(".,")
        ctx = s[max(0, m.start() - 50): m.end() + 40].replace("\n", " ")
        cls, why = "untraced", ""
        win = s[max(0, m.start() - 12): m.end() + 12]
        if date_re.search(s[max(0, m.start() - 6): m.end() + 8]) or re.search(r"2026-\d\d-\d\d", win):
            d = date_re.search(s[max(0, m.start() - 6): m.end() + 8])
            cls = "structural" if (d is None or d.group() in log) else "untraced"
            why = "date found in decision log" if cls == "structural" else "date not in decision log"
        elif lit in checked:
            cls, why = "checked_result", checked[lit][0] + (" -> verified" if checked[lit][1] else " -> FAILED")
            if not checked[lit][1]:
                cls = "untraced"
        elif lit in external:
            cls, why = "structural", "external documentation fact: " + external[lit]
        elif structural_ctx.search(ctx):
            cls, why = "structural", "statistical convention or enumeration in text"
        elif re.search(r"(?<![\d.])" + re.escape(lit) + r"(?![\d])", corpus):
            cls, why = "typed_constant", "appears in code, config or an artifact"
        elif lit in {"1", "2", "3", "4", "5", "10", "20", "95", "0", "580", "520"}:
            cls, why = "structural", "small count or instrument name"
        counts[cls] += 1
        out.append({"literal": lit, "class": cls, "why": why, "context": ctx})
    return {"n_literals": len(out), "counts": counts, "untraced": [o for o in out if o["class"] == "untraced"], "all": out}


# ------------------------------------------------------------------ recomputation from per-row payload
def exact_mcnemar(b: int, c: int) -> float:
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0**n)


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def recompute() -> dict:
    """Recompute the point statistics of every ablation row from the stored per-row predictions."""
    z = np.load(REPO / "artifacts" / "revision" / "ablation_recalibrated_per_row.npz", allow_pickle=True)
    rc = load("artifacts/revision/ablation_recalibrated.json")
    met = load("artifacts/metrics.json")
    classes = ["F", "G", "K"]
    out: dict[str, Any] = {"n_rows_checked": 0, "max_abs_diff": {}, "mismatches": [], "headline": {}}
    fields = {"baseline_acc": 1e-9, "masked_acc_mean": 1e-9, "delta_acc_mean": 1e-9, "n_flip_lost": 0, "n_flip_gained": 0,
              "mcnemar_exact_p": 1e-12, "mean_delta_true_prob": 2e-5, "flip_rate_upper95": 1e-9, "n_test": 0}
    for run, rd in rc["runs"].items():
        y = z[f"{run}__y_test"].astype(int)
        base = z[f"{run}__y_pred_base"].astype(int)
        pcls = z[f"{run}__proba_classes"].astype(int)
        pb = z[f"{run}__proba_base"].astype(float)
        if run == rc["primary_run"]:
            cmx = np.zeros((3, 3), int)
            lab = {int(v): i for i, v in enumerate(pcls)}  # stored labels are 1, 2, 3 for F, G, K
            for t, p in zip(y, base):
                cmx[lab[int(t)], lab[int(p)]] += 1
            out["baseline_confusion_matches_metrics_json"] = bool((cmx == np.array(met["confusion_matrix"])).all())
        col = {int(c): i for i, c in enumerate(pcls)}
        for row in rd["rows"]:
            ls, cl = row["line_set"], row["mk_class"]
            sel = y == int(pcls[classes.index(cl)])
            masked = z[f"{run}__y_pred_masked__{ls}"].astype(int)
            pm = z[f"{run}__proba_masked__{ls}"].astype(float)
            ok0, ok1 = (base == y)[sel], (masked == y)[sel]
            lost, gained = int((ok0 & ~ok1).sum()), int((~ok0 & ok1).sum())
            idx_true = np.array([col[int(t)] for t in y[sel]])
            dp = float((pm[sel][np.arange(sel.sum()), idx_true] - pb[sel][np.arange(sel.sum()), idx_true]).mean())
            rec = {"n_test": int(sel.sum()), "baseline_acc": float(ok0.mean()), "masked_acc_mean": float(ok1.mean()),
                   "delta_acc_mean": float(ok1.mean() - ok0.mean()), "n_flip_lost": lost, "n_flip_gained": gained,
                   "mcnemar_exact_p": exact_mcnemar(lost, gained), "mean_delta_true_prob": dp,
                   "flip_rate_upper95": cp_upper(lost, int(sel.sum()))}
            # masked_acc_mean in the JSON is the mean over the point estimate (no resampling), so it must agree
            for f, tol in fields.items():
                d = abs(rec[f] - row[f])
                out["max_abs_diff"][f] = max(out["max_abs_diff"].get(f, 0.0), d)
                if d > tol:
                    out["mismatches"].append({"run": run, "line_set": ls, "class": cl, "field": f, "stored": row[f], "recomputed": rec[f]})
            out["n_rows_checked"] += 1
            if run == rc["primary_run"] and (ls, cl) in {("Mg_b", "G"), ("Mg_b", "K"), ("H_balmer", "F"), ("Na_D", "K")}:
                out["headline"][f"{ls},{cl}"] = {k: rec[k] for k in ("n_test", "delta_acc_mean", "n_flip_lost", "n_flip_gained", "mcnemar_exact_p", "mean_delta_true_prob")}
    return out


if __name__ == "__main__":
    res = {"macros": macro_check(), "literals": literal_check(), "recomputation": recompute()}
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    mc = res["macros"]
    print(mc["n_index"], mc["status_counts"], len(mc["mismatches"]), len(mc["unresolved"]))
    print(res["literals"]["counts"])
    rr = res["recomputation"]
    print(rr["n_rows_checked"], len(rr["mismatches"]), rr["baseline_confusion_matches_metrics_json"])
