#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Geometry of the ablation line sets on the 696-bin log-uniform grid.

For every line set (the four MK sets in ``src.interpret.lines.LINE_SETS``
and the Fe i / Cr i multiplet set of
``scripts/appendix_a2_polynomial_diagnostics.py``) and every window the
table lists the targeted rest wavelengths, the window limits and width in
Angstrom, the number and indices of grid bins whose centres fall inside the
window (the bins :func:`src.interpret.ablation._window_mask` masks), the
local bin width, and the strong lines of other species that fall inside the
window at the grid resolution. Lines of the set's own species that are not
the nominal target, and strong lines lying within 12 A outside the window,
are listed separately so the window boundaries can be checked.

Rest wavelengths are air wavelengths from the NIST ASD / Moore multiplet
tables; the catalogue below is restricted to lines that are strong in FGK
spectra at R ~ 2000 and is not exhaustive.

Outputs: artifacts/revision/line_sets.csv and artifacts/revision/line_sets_table.tex
Run from the repo root: ``python scripts/revision/line_set_table.py``
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.ablation import _window_mask  # noqa: E402
from src.interpret.lines import LINE_SETS, MK_LINES  # noqa: E402

logger = logging.getLogger(__name__)

SET_SPECIES: dict[str, tuple[str, ...]] = {
    "H_balmer": ("H I",),
    "Mg_b": ("Mg I",),
    "Na_D": ("Na I",),
    "Ca_I": ("Ca I",),
    "Fe_Cr": ("Fe I", "Cr I"),
}
SET_DISPLAY: dict[str, str] = {
    "H_balmer": r"H Balmer", "Mg_b": r"Mg\,b", "Na_D": r"Na\,D", "Ca_I": r"Ca\,\textsc{i}",
    "Fe_Cr": r"Fe\,\textsc{i}/Cr\,\textsc{i}",
}

# Nominal target lines of the Fe i / Cr i set (see appendix_a2 docstring).
FE_CR_TARGET_LINES: list[tuple[str, float, str]] = [
    ("Cr I", 5206.04, "Cr_I_5206"),
    ("Cr I", 5208.42, "Cr_I_5208"),
    ("Fe I", 5269.54, "Fe_I_5270"),
    ("Cr I", 5345.80, "Cr_I_5346"),
]

# Strong lines (air wavelengths, A) of FGK spectra near the ablation windows.
BLEND_CATALOGUE: list[tuple[str, float]] = [
    # near H beta
    ("Fe I", 4859.74), ("Fe I", 4871.32), ("Fe I", 4872.14), ("Fe I", 4878.21),
    ("Fe I", 4890.76), ("Fe I", 4891.49),
    # near Mg b
    ("Fe I", 5167.49), ("Fe I", 5168.90), ("Fe II", 5169.03), ("Fe I", 5171.60),
    ("Ti I", 5173.74), ("Fe I", 5191.45), ("Fe I", 5192.34),
    # Fe/Cr windows
    ("Fe I", 5202.34), ("Cr I", 5204.51), ("Fe I", 5204.58), ("Y II", 5205.73),
    ("Cr I", 5206.04), ("Cr I", 5208.42), ("Fe I", 5208.59), ("Ti I", 5210.39),
    ("Fe I", 5215.18),
    ("Fe II", 5264.81), ("Ca I", 5265.56), ("Fe I", 5266.56), ("Fe I", 5269.54),
    ("Ca I", 5270.27), ("Fe I", 5270.36), ("Fe I", 5273.17), ("Fe II", 5276.00),
    ("Fe I", 5339.93), ("Fe I", 5341.03), ("Co I", 5342.70), ("Cr I", 5345.80),
    ("Cr I", 5348.31), ("Fe I", 5353.37),
    # near Na D
    ("Fe I", 5883.82), ("Ni I", 5892.88), ("Fe I", 5905.67),
    # near Ca I 6162
    ("Na I", 6154.23), ("Na I", 6160.75), ("Ca I", 6161.30), ("Ca I", 6163.75),
    ("Fe I", 6165.36), ("Ca I", 6166.44), ("Ca I", 6169.04), ("Ca I", 6169.56),
    # near Ca I 6439
    ("Fe I", 6430.85), ("Fe II", 6432.68), ("Fe I", 6436.41), ("Eu II", 6437.64),
    ("Ca I", 6449.81),
    # near H alpha
    ("Fe I", 6546.24), ("Ti I", 6554.22), ("Ti I", 6556.06), ("Fe I", 6569.21),
    ("Ca I", 6572.78), ("Fe I", 6574.23), ("Fe I", 6592.91),
]
ADJACENT_TOL_AA = 12.0


def _load_fe_cr() -> list[tuple[float, float]]:
    path = REPO_ROOT / "scripts" / "appendix_a2_polynomial_diagnostics.py"
    spec = importlib.util.spec_from_file_location("appendix_a2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return [tuple(w) for w in mod.FE_CR_LINE_SET]


def _fmt_lines(entries: list[tuple[str, float]]) -> str:
    return "; ".join(f"{sp} {wl:.2f}" for sp, wl in entries) if entries else "none"


def _fmt_lines_tex(entries: list[tuple[str, float]]) -> str:
    def one(sp: str, wl: float) -> str:
        el, ion = sp.split()
        return rf"{el}\,\textsc{{{ion.lower()}}} {wl:.2f}"
    return "; ".join(one(sp, wl) for sp, wl in entries) if entries else "none"


def _bin_ranges(idx: np.ndarray) -> str:
    if len(idx) == 0:
        return ""
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("artifacts/features.npz"))
    p.add_argument("--out-csv", type=Path, default=Path("artifacts/revision/line_sets.csv"))
    p.add_argument("--out-tex", type=Path, default=Path("artifacts/revision/line_sets_table.tex"))
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    fp = np.load(args.features, allow_pickle=False)
    wc = fp["wave_centers"].astype(np.float64)
    gap_mask = fp["gap_mask"].astype(bool)
    dln = float(np.median(np.diff(np.log(wc))))
    bin_width = np.gradient(wc)  # local bin width in A

    sets = {name: [tuple(w) for w in LINE_SETS[name]] for name in ("H_balmer", "Mg_b", "Na_D", "Ca_I")}
    sets["Fe_Cr"] = _load_fe_cr()

    # MK_LINES species are written "HI", "MgI", ...; split element and ionisation stage.
    mk_targets = [(f"{ln.species[:-1]} {ln.species[-1]}", ln.wavelength_aa, ln.name) for ln in MK_LINES]
    all_targets = mk_targets + FE_CR_TARGET_LINES

    rows = []
    for set_name, windows in sets.items():
        own = set(SET_SPECIES[set_name])
        for wi, (lo, hi) in enumerate(windows):
            m = _window_mask(wc, [(lo, hi)])
            idx = np.flatnonzero(m)
            targets = [(sp, wl, nm) for sp, wl, nm in all_targets if lo <= wl <= hi and sp in own]
            inside = [(sp, wl) for sp, wl in BLEND_CATALOGUE if lo <= wl <= hi]
            target_wls = {round(wl, 2) for _, wl, _ in targets}
            other = [(sp, wl) for sp, wl in inside if sp not in own]
            same = [(sp, wl) for sp, wl in inside if sp in own and round(wl, 2) not in target_wls]
            adjacent = [(sp, wl) for sp, wl in BLEND_CATALOGUE
                        if (lo - ADJACENT_TOL_AA <= wl < lo) or (hi < wl <= hi + ADJACENT_TOL_AA)]
            rows.append({
                "line_set": set_name,
                "set_species": "/".join(SET_SPECIES[set_name]),
                "window_index": wi,
                "window_lo_aa": lo,
                "window_hi_aa": hi,
                "width_aa": hi - lo,
                "target_lines": "; ".join(f"{sp} {wl:.2f} ({nm})" for sp, wl, nm in targets),
                "n_bins": int(m.sum()),
                "bin_indices": _bin_ranges(idx),
                "first_bin_center_aa": float(wc[idx[0]]) if len(idx) else np.nan,
                "last_bin_center_aa": float(wc[idx[-1]]) if len(idx) else np.nan,
                "mean_bin_width_aa": float(bin_width[idx].mean()) if len(idx) else np.nan,
                "n_gap_bins_in_window": int((m & gap_mask).sum()),
                "blended_other_species_in_window": _fmt_lines(other),
                "same_species_non_target_in_window": _fmt_lines(same),
                f"strong_lines_within_{ADJACENT_TOL_AA:.0f}aa_outside": _fmt_lines(adjacent),
                "_other": other, "_targets": targets,
            })
    df = pd.DataFrame(rows)
    totals = df.groupby("line_set", sort=False).agg(
        total_width_aa=("width_aa", "sum"), total_bins=("n_bins", "sum"), n_windows=("window_index", "count"),
    )
    df["set_total_width_aa"] = df["line_set"].map(totals["total_width_aa"])
    df["set_total_bins"] = df["line_set"].map(totals["total_bins"])
    df["dln_lambda_per_bin"] = dln
    df["resolving_power_per_bin"] = 1.0 / dln
    df["grid_bin_width_range_aa"] = f"{bin_width.min():.2f}-{bin_width.max():.2f}"

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_other", "_targets"]).to_csv(args.out_csv, index=False)

    # --- LaTeX body (booktabs) --------------------------------------------
    lines = [
        r"% Generated by scripts/revision/line_set_table.py; wavelengths in air (A).",
        rf"% Grid: {len(wc)} log-uniform bins, d ln(lambda) = {dln:.2e} per bin "
        rf"(R_bin = 1/d ln(lambda) = {1.0 / dln:.0f}); bin width {bin_width.min():.2f}-{bin_width.max():.2f} A.",
        r"\begin{tabular}{@{}llrrrp{5.2cm}@{}}",
        r"\toprule",
        r"Line set & Window (\AA) & Target lines (\AA) & Width (\AA) & Bins & Other species inside window (\AA) \\",
        r"\midrule",
    ]
    for set_name in sets:
        sub = df[df["line_set"] == set_name]
        for k, (_, r) in enumerate(sub.iterrows()):
            label = SET_DISPLAY[set_name] if k == 0 else ""
            tgt = ", ".join(f"{wl:.2f}" for _, wl, _ in r["_targets"])
            lines.append(
                rf"{label} & {r['window_lo_aa']:.1f}--{r['window_hi_aa']:.1f} & {tgt} & "
                rf"{r['width_aa']:.0f} & {r['n_bins']} & {_fmt_lines_tex(r['_other'])} \\"
            )
        if len(sub) > 1:
            t = totals.loc[set_name]
            lines.append(rf"\quad total & & & {t['total_width_aa']:.0f} & {int(t['total_bins'])} & \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    args.out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", args.out_csv, args.out_tex)
    for set_name, t in totals.iterrows():
        print(f"{set_name:9s} windows={int(t['n_windows'])} width={t['total_width_aa']:5.1f} A bins={int(t['total_bins'])}")
    print(f"d ln(lambda) = {dln:.3e}; R per bin = {1.0 / dln:.0f}; bin width {bin_width.min():.2f}-{bin_width.max():.2f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
