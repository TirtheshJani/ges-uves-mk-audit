#!/usr/bin/env python
"""Back-fill the Zenodo DOI after the GitHub release has been archived.

Replaces the placeholder "DOI to be minted from the v1.2.0 GitHub release through the
Zenodo integration" (with or without a leading "DOI: ") in the manuscript, README,
CITATION.cff and the Zenodo metadata, then prints what changed. Run it with the concept
DOI that Zenodo shows for the record (it always resolves to the latest version):

    python scripts/revision/set_doi.py 10.5281/zenodo.1234567
    python scripts/revision/build_venue_variants.py
    python scripts/revision/verify_numbers.py
    cd submission && latexmk -pdf manuscript.tex manuscript_elsarticle.tex
    cd submission/rasti_overleaf && latexmk -pdf main.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLACEHOLDER = "to be minted from the v1.2.0 GitHub release through the Zenodo integration"
FILES = ["submission/manuscript.tex", "README.md", "CITATION.cff",
         ".zenodo.json", "submission/zenodo_metadata.json"]


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"10\.5281/zenodo\.\d+", sys.argv[1]):
        raise SystemExit("usage: set_doi.py 10.5281/zenodo.NNNNNNN")
    doi = sys.argv[1]
    for rel in FILES:
        path = REPO / rel
        text = path.read_text(encoding="utf-8")
        if rel == "CITATION.cff":
            new = text.replace(
                '  - type: other\n    value: "DOI: %s"\n    description: "Zenodo DOI of the code and data archive; '
                'no DOI has been minted yet. Replace this entry when the deposit is published."' % PLACEHOLDER,
                '  - type: doi\n    value: "%s"\n    description: "Zenodo concept DOI of the code and data archive '
                '(resolves to the latest version)"' % doi)
        elif rel.endswith(".tex"):
            new = text.replace("DOI " + PLACEHOLDER, "DOI \\href{https://doi.org/%s}{%s}" % (doi, doi))
        else:
            new = text.replace("DOI: " + PLACEHOLDER, "DOI: " + doi).replace(PLACEHOLDER, doi)
        n = text.count(PLACEHOLDER)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
        print(f"{rel}: {n} replaced")


if __name__ == "__main__":
    main()
