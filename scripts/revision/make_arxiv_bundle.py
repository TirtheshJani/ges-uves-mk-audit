#!/usr/bin/env python
"""Assemble a clean arXiv upload folder from submission/manuscript.tex.

Writes submission/arxiv/ with:

    manuscript.tex          the master with the leading comment block (everything before
                            \\documentclass) and the %% marker lines removed
    revision_numbers.tex    generated numbers
    *.tex                   every file the master \\input{}s
    *.pdf                   every figure the master \\includegraphics{}es
    manuscript.bbl          fresh, from a latexmk run in a temporary directory

Not copied: manuscript_citations.bib (arXiv wants the .bbl), the venue variants, the
built PDFs and any JSON. The folder is regenerated on every run and is git-ignored.

    python scripts/revision/make_arxiv_bundle.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUB = REPO / "submission"
MASTER = SUB / "manuscript.tex"
BIB = SUB / "manuscript_citations.bib"
OUT = SUB / "arxiv"


def _clean_master(text: str) -> str:
    text = text[text.index("\\documentclass") :]
    lines = [ln for ln in text.splitlines() if not re.match(r"%% [A-Z]+-(START|END)\s*$", ln)]
    return "\n".join(lines) + "\n"


def _dependencies(text: str) -> list[str]:
    inputs = re.findall(r"\\input\{([^}]+)\}", text)
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    names = ["revision_numbers.tex"]
    for name in inputs + figures:
        if "." not in Path(name).name:
            name += ".tex"
        if name not in names:
            names.append(name)
    return names


def _build_bbl(bundle: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for path in bundle.iterdir():
            shutil.copy(path, work / path.name)
        shutil.copy(BIB, work / BIB.name)
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-silent", "manuscript.tex"],
            cwd=work,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return (work / "manuscript.bbl").read_text(encoding="utf-8")


def main() -> None:
    master = MASTER.read_text(encoding="utf-8")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    (OUT / "manuscript.tex").write_text(_clean_master(master), encoding="utf-8", newline="\n")
    for name in _dependencies(master):
        shutil.copy(SUB / name, OUT / name)
    (OUT / "manuscript.bbl").write_text(_build_bbl(OUT), encoding="utf-8", newline="\n")
    print(f"arXiv bundle in {OUT.relative_to(REPO)}/")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
