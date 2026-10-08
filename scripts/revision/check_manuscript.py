#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Static consistency checks for submission/manuscript.tex when no TeX engine is available.

Checks: brace balance and begin/end pairing, every \\ref/\\eqref/\\cite key resolves, every
\\input and \\includegraphics target exists, every macro used is defined (preamble,
revision_numbers.tex or a standard LaTeX/package command), the bibliography has no en or em
dashes and no unverified-reference marker, and which bibliography entries are unused. It also
lists the [TBD-ablation] macros and the [CHECK-ablation] markers that remain in the text.

Run from the repository root: ``python scripts/revision/check_manuscript.py``
Exit status is 1 when a hard error is found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submission"

KNOWN = set(r"""
documentclass usepackage hypersetup geometry title author affil thanks date maketitle begin end
section subsection paragraph label ref eqref cite citep citet bibliographystyle bibliography appendix
input includegraphics caption centering small footnotesize resizebox setlength tabcolsep toprule
midrule bottomrule addlinespace multicolumn textbf emph texttt textsc textwidth linewidth AA
ensuremath newcommand providecommand renewcommand item url href mathrm Delta bar ln lambda alpha
times leq geq le ge approx pm to rightarrow phantom noindent hline cr newtoggle toggletrue
togglefalse iftoggle xspace phantom sim ll gg in mathcal text underline sqrt left right
texorpdfstring noalign omit hspace vspace par quad qquad ldots dots cdot label tag nonumber
paragraph section subsection subsubsection abstract itemize enumerate equation table figure tabular
sloppypar and thefootnote
""".split())


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def main() -> int:
    tex = (SUB / "manuscript.tex").read_text(encoding="utf-8")
    body = strip_comments(tex)
    errors: list[str] = []
    notes: list[str] = []

    # brace balance
    stripped = re.sub(r"\\[{}]", "", body)
    depth = 0
    for i, ch in enumerate(stripped):
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            errors.append(f"unmatched closing brace near char {i}")
            break
    if depth != 0:
        errors.append(f"brace imbalance: {depth}")
    begins = re.findall(r"\\begin\{([^}]+)\}", body)
    ends = re.findall(r"\\end\{([^}]+)\}", body)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            errors.append(f"environment {env}: {begins.count(env)} begin vs {ends.count(env)} end")

    # labels and references
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body)
    for r in sorted(set(refs) - labels):
        errors.append(f"unresolved reference: {r}")
    for lab in sorted(labels - set(refs)):
        notes.append(f"label never referenced: {lab}")

    # citations
    bib = (SUB / "manuscript_citations.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited: set[str] = set()
    for grp in re.findall(r"\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}", body):
        cited.update(k.strip() for k in grp.split(","))
    for k in sorted(cited - keys):
        errors.append(f"citation key not in bibliography: {k}")
    for k in sorted(keys - cited):
        notes.append(f"bibliography entry never cited: {k}")
    if re.search("[\u2013\u2014]", bib):
        errors.append("bibliography contains an en or em dash")
    if re.search("TODO", bib, re.I):
        errors.append("bibliography still carries a TODO marker")
    if re.search("[\u2013\u2014]", tex):
        errors.append("manuscript contains an en or em dash character")

    # files
    for f in re.findall(r"\\input\{([^}]+)\}", body):
        if not (SUB / f).exists():
            errors.append(f"\\input target missing: {f}")
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if not (SUB / f).exists():
            errors.append(f"figure missing: {f}")

    # macro definitions
    nums = (SUB / "revision_numbers.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\(?:newcommand|providecommand|renewcommand)\{\\([A-Za-z]+)\}", tex + nums))
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    unknown = sorted(u for u in used if u not in defined and u not in KNOWN)
    if unknown:
        notes.append("commands not defined in the manuscript or known list (check by eye): " + ", ".join(unknown))
    unused_defs = sorted(d for d in defined if d not in used and not re.match(r"^(abl|str|focus|sd|rt|lum|pk|bd|ew|tri|cv|test|conf|polyConf|med|ks|nClass)", d))
    if unused_defs:
        notes.append("macros defined but not used: " + ", ".join(unused_defs))

    # open slots
    tbd_defs = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{\\textbf\{\[TBD-ablation\]\}\}", nums))
    tbd_used = sorted(u for u in used if u in tbd_defs)
    n_chk = len(re.findall(r"\\chk\b", body))
    for tab in ("ablation_table.tex", "ablation_k_table.tex", "ablation_strata_table.tex"):
        if "[TBD-ablation]" in (SUB / tab).read_text(encoding="utf-8"):
            tbd_used.append(f"<table {tab}>")

    print(f"labels: {len(labels)}, refs: {len(set(refs))}, cite keys used: {len(cited)}, bib entries: {len(keys)}")
    print(f"macros defined: {len(defined)}; macros used in text: {len(used & defined)}")
    print(f"[TBD-ablation] slots in text: {len(tbd_used)}")
    for u in tbd_used:
        print("  TBD   ", u)
    print(f"[CHECK-ablation] markers in text: {n_chk}")
    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
