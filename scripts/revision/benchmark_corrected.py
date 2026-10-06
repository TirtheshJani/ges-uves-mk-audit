#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Pickles (1998) template cross-check with header-derived template types.

The spectral type of each UVKLIB template is read from the FITS primary
header of the STScI distribution (``COMMENT1 = 'spectral type: G2V'``,
``COMMENT2 = 'metallicity: ...'``) rather than from the index->type table
embedded in ``src.interpret.benchmark``.  The header-derived map is checked
against the STScI README ordering (files 1-45 luminosity class V, 46-59 IV,
60-105 III, 106-113 II, 114-131 I) and saved in the output JSON.

The matching procedure otherwise follows ``scripts/run_benchmark.py``:
templates are continuum-normalised (median_filter_200, median_filter_50 or
polynomial_n5), resampled onto the 696-bin feature grid, and the
minimum-chi-squared template is chosen for every held-out test row.  The
test rows themselves are used exactly as stored in ``features.npz``
(pipeline-normalised, not re-normalised), as in the original run.  All 131
templates take part in the matching; rows whose best template is not
F/G/K are reported separately and excluded from the agreement statistics.

Output: artifacts/revision/benchmark_corrected.json
Run from the repo root:
``python scripts/revision/benchmark_corrected.py --pickles-dir <dir with pickles_uk_N.fits>``
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.interpret.benchmark import (  # noqa: E402
    PICKLES_UVKLIB_MAP,
    Template,
    VALID_CONTINUUM_METHODS,
    benchmark_report,
    best_template_per_spectrum,
    collapse_to_mk,
    continuum_normalize,
    load_template_fits,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "F", 2: "G", 3: "K"}
SEED = 42
AGREEMENT_FLOOR = 0.55
MIN_N_COMPARED = 100
README_LUMINOSITY_RANGES = {"V": (1, 45), "IV": (46, 59), "III": (60, 105), "II": (106, 113), "I": (114, 131)}
_TYPE_RE = re.compile(r"spectral type:\s*(?P<t>\S+)", re.IGNORECASE)
_LUM_RE = re.compile(r"^(?P<temp>[OBAFGKM]\d*(?:\.\d+)?)(?P<lum>I{1,3}V?|IV|V)?$")


def read_header_type(path: Path) -> dict:
    """Return the spectral type, metallicity note and original filename from the FITS header."""
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
    text = "\n".join(f"{k}={h[k]}" for k in h if k.upper().startswith("COMMENT"))
    m = _TYPE_RE.search(text)
    if m is None:
        raise ValueError(f"no spectral type card in {path.name}")
    metallicity = ""
    orig = ""
    for k in h:
        v = str(h[k])
        if "metallicity" in v.lower():
            metallicity = v.split(":", 1)[-1].strip()
        if "original filename" in v.lower():
            orig = v.split(":", 1)[-1].strip()
    return {"type": m.group("t").strip(), "metallicity": metallicity, "original_filename": orig}


def luminosity_class(type_str: str) -> str:
    m = _LUM_RE.match(type_str.upper())
    if m is None or m.group("lum") is None:
        return "unknown"
    return m.group("lum")


def luminosity_group(lum: str) -> str:
    if lum == "V":
        return "dwarf"
    if lum == "IV":
        return "subgiant"
    if lum == "III":
        return "giant"
    if lum in ("II", "I"):
        return "supergiant"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--model", type=Path, default=Path("artifacts/lgbm_mk.pkl"))
    p.add_argument("--pickles-dir", type=Path,
                   default=Path(r"C:\Users\TJ\Documents\GitHub\stellar-mk-audit\data\pickles"))
    p.add_argument("--out", type=Path, default=Path("artifacts/revision/benchmark_corrected.json"))
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    np.random.seed(SEED)

    files = sorted(args.pickles_dir.glob("pickles_uk_*.fits"),
                   key=lambda q: int(re.search(r"(\d+)", q.stem).group(1)))
    if len(files) != 131:
        raise SystemExit(f"expected 131 Pickles templates, found {len(files)} in {args.pickles_dir}")

    # --- header-derived index -> type map ----------------------------------
    header_map: dict[int, dict] = {}
    for f in files:
        n = int(re.search(r"(\d+)", f.stem).group(1))
        info = read_header_type(f)
        info["luminosity_class"] = luminosity_class(info["type"])
        info["mk_class"] = collapse_to_mk(info["type"])
        header_map[n] = info
    readme_check = {}
    for lum, (lo, hi) in README_LUMINOSITY_RANGES.items():
        got = Counter(header_map[n]["luminosity_class"] for n in range(lo, hi + 1))
        readme_check[lum] = {"indices": [lo, hi], "header_luminosity_counts": dict(got),
                             "consistent": set(got) == {lum}}
    embedded_vs_header = {
        str(n): {"embedded": PICKLES_UVKLIB_MAP[n], "header": header_map[n]["type"]}
        for n in sorted(PICKLES_UVKLIB_MAP) if PICKLES_UVKLIB_MAP[n] != header_map[n]["type"]
    }
    n_embedded_match = sum(PICKLES_UVKLIB_MAP[n] == header_map[n]["type"] for n in PICKLES_UVKLIB_MAP)
    logger.info("header map: %d/%d embedded entries agree with the FITS headers; README ranges consistent: %s",
                n_embedded_match, len(PICKLES_UVKLIB_MAP), all(v["consistent"] for v in readme_check.values()))

    # --- audit side (unchanged from run_benchmark.py) -----------------------
    fp = np.load(args.features, allow_pickle=False)
    X = fp["X"]
    y = fp["y"].astype(np.int64)
    wc = fp["wave_centers"]
    test_idx = fp["test_idx"]
    X_test, y_test = X[test_idx], y[test_idx]
    with args.model.open("rb") as f:
        model = pickle.load(f)
    y_model = np.array([CLASS_NAMES[int(i)] for i in model.predict(X_test)])
    y_true = np.array([CLASS_NAMES[int(i)] for i in y_test])
    class_labels = ["F", "G", "K"]

    raw_templates = {n: load_template_fits(f) for n, f in zip(sorted(header_map), files)}

    # The three deposited conventions normalise the template over its full
    # 1150-25000 A range before resampling.  A fourth, clearly labelled
    # variant restricts the polynomial fit to the audit window (+-100 A)
    # because a single 5th-order Legendre over the full UVKLIB range cannot
    # follow the continuum inside the 4800-6800 A window.
    conventions = list(VALID_CONTINUUM_METHODS) + ["polynomial_n5_window"]
    w_lo, w_hi = float(wc.min()) - 100.0, float(wc.max()) + 100.0
    per_method = {}
    for method in conventions:
        templates: list[Template] = []
        for n in sorted(header_map):
            wave, flux = raw_templates[n]
            if method == "polynomial_n5_window":
                sel = (wave >= w_lo) & (wave <= w_hi)
                wave = wave[sel]
                flux_norm = continuum_normalize(wave, flux[sel], method="polynomial_n5")
            else:
                flux_norm = continuum_normalize(wave, flux, method=method)
            on_grid = np.interp(wc, wave, flux_norm, left=np.nan, right=np.nan).astype(np.float32)
            templates.append(Template(filename=files[n - 1].name, mk_type=header_map[n]["type"],
                                      mk_class=header_map[n]["mk_class"], flux_on_grid=on_grid))
        best = best_template_per_spectrum(X_test, templates)
        best_type = np.array([templates[i].mk_type for i in best])
        best_class = np.array([templates[i].mk_class for i in best])
        best_lum = np.array([luminosity_group(luminosity_class(t)) for t in best_type])
        best_letter = np.array([t[0].upper() for t in best_type])

        fgk = np.isin(best_class, class_labels)
        if fgk.any():
            report = benchmark_report(y_pred_model=y_model[fgk], y_pickles_mk=best_class[fgk],
                                      class_labels=class_labels)
            report_vs_truth = benchmark_report(y_pred_model=y_true[fgk], y_pickles_mk=best_class[fgk],
                                               class_labels=class_labels)
        else:
            empty = {"agreement_rate": float("nan"), "n_compared": 0, "macro_f1_fgk": float("nan"),
                     "per_class_precision": {c: float("nan") for c in class_labels},
                     "per_class_recall": {c: float("nan") for c in class_labels},
                     "confusion_matrix": [[0, 0, 0, 0] for _ in range(4)]}
            report = report_vs_truth = empty
        # Legacy-style statistics: keep every non-OTHER row (A templates included) as in run_benchmark.py.
        legacy_mask = best_class != "OTHER"
        legacy_agreement = (float(np.mean(best_class[legacy_mask] == y_model[legacy_mask]))
                            if legacy_mask.any() else float("nan"))
        gate_pass = bool(report["n_compared"] > MIN_N_COMPARED and report["agreement_rate"] >= AGREEMENT_FLOOR)
        per_method[method] = {
            "n_test": int(len(X_test)),
            "n_compared_fgk": int(fgk.sum()),
            "n_best_template_A": int(np.sum(best_letter == "A")),
            "n_best_template_M": int(np.sum(best_letter == "M")),
            "n_best_template_OBA_other": int(np.sum(np.isin(best_letter, ["O", "B"]))),
            "best_template_letter_counts": dict(Counter(best_letter.tolist())),
            "best_template_luminosity_counts": dict(Counter(best_lum.tolist())),
            "fraction_best_template_dwarf": float(np.mean(best_lum == "dwarf")),
            "fraction_best_template_subgiant": float(np.mean(best_lum == "subgiant")),
            "fraction_best_template_giant": float(np.mean(best_lum == "giant")),
            "fraction_best_template_supergiant": float(np.mean(best_lum == "supergiant")),
            "best_template_luminosity_by_true_class": {
                c: dict(Counter(best_lum[y_true == c].tolist())) for c in class_labels
            },
            "best_template_type_counts_top": dict(Counter(best_type.tolist()).most_common(12)),
            "agreement_rate_model_vs_pickles": report["agreement_rate"],
            "per_class_precision": {c: report["per_class_precision"][c] for c in class_labels},
            "per_class_recall": {c: report["per_class_recall"][c] for c in class_labels},
            "macro_f1_fgk": report["macro_f1_fgk"],
            "confusion_rows_pickles_cols_model": [row[:3] for row in report["confusion_matrix"][:3]],
            "confusion_labels": class_labels,
            "agreement_rate_truth_vs_pickles": report_vs_truth["agreement_rate"],
            "per_class_recall_truth_vs_pickles": {c: report_vs_truth["per_class_recall"][c] for c in class_labels},
            "legacy_style_agreement_including_A_rows": legacy_agreement,
            "legacy_style_n_compared": int(legacy_mask.sum()),
            "gate": {"required_agreement": AGREEMENT_FLOOR, "min_n_compared": MIN_N_COMPARED,
                     "observed_agreement": report["agreement_rate"], "n_compared": report["n_compared"],
                     "status": "PASS" if gate_pass else "FAIL"},
        }
        logger.info("%-18s n_fgk=%d agreement=%.3f (truth %.3f) M=%d A=%d giants+supergiants=%.2f gate=%s",
                    method, fgk.sum(), report["agreement_rate"], report_vs_truth["agreement_rate"],
                    per_method[method]["n_best_template_M"], per_method[method]["n_best_template_A"],
                    per_method[method]["fraction_best_template_giant"] + per_method[method]["fraction_best_template_supergiant"],
                    per_method[method]["gate"]["status"])

    payload = {
        "description": ("Pickles 1998 UVKLIB template cross-check of the production classifier on the "
                        "held-out test rows, with template spectral types read from the FITS headers; "
                        "three template continuum conventions; audit rows used as stored."),
        "inputs": {"features": str(args.features), "model": str(args.model), "pickles_dir": str(args.pickles_dir),
                   "n_templates": len(files), "n_test": int(len(test_idx)),
                   "continuum_methods_deposited": list(VALID_CONTINUUM_METHODS),
                   "continuum_methods_additional": ["polynomial_n5_window"],
                   "polynomial_n5_window_aa": [w_lo, w_hi],
                   "template_type_source": "FITS primary header card COMMENT1 ('spectral type: ...')",
                   "audit_side_normalisation": "none (features.npz rows as stored)",
                   "chi2": "unit-variance mean squared difference over bins finite in both spectra",
                   "seed": SEED},
        "header_type_map": {str(n): header_map[n] for n in sorted(header_map)},
        "readme_ordering_check": readme_check,
        "embedded_map_vs_header": {"n_embedded_entries": len(PICKLES_UVKLIB_MAP), "n_agree": int(n_embedded_match),
                                   "disagreements": embedded_vs_header},
        "per_method": per_method,
        "any_method_passes_gate": any(v["gate"]["status"] == "PASS" for v in per_method.values()),
        "any_deposited_method_passes_gate": any(per_method[m]["gate"]["status"] == "PASS"
                                                for m in VALID_CONTINUUM_METHODS),
    }

    def _nan_to_none(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(v) for v in obj]
        return obj

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(_nan_to_none(payload), f, indent=2)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
