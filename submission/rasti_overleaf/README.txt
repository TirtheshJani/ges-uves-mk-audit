RAS Techniques and Instruments submission source
=================================================

Paper: A masked-line ablation audit of a LightGBM F/G/K temperature-class classifier
for Gaia-ESO UVES spectra: what survives, and a train-test scale mismatch
Author: Tirthesh Jani (ORCID 0009-0005-5965-4409)

This folder compiles on its own with pdflatex and BibTeX (Overleaf's default):

  main.tex                  manuscript (rasti class, two-column)
  rasti.cls, rasti.bst      RASTI class and bibliography style (bundle of 28 September 2026,
                            LaTeX Project Public License)
  manuscript_citations.bib  bibliography database
  revision_numbers.tex      every number quoted in the paper, one macro per number
  *_table.tex               generated tables, \input by main.tex
  *.pdf (figures)           ablation_bars, kiel_diagram, continuum_medians,
                            importance_main, confusion_matrix_test
  main.pdf                  compiled manuscript, the file uploaded to ScholarOne

Overleaf: zip this folder, then New Project > Upload Project. Main document: main.tex;
compiler: pdfLaTeX. The project can then be submitted from Overleaf's Submit menu
(RAS Techniques and Instruments), or main.pdf can be uploaded to ScholarOne as the
"Complete manuscript file".

Provenance: main.tex and the support files are generated from submission/manuscript.tex
in the code repository (https://github.com/TirtheshJani/ges-uves-mk-audit) by
scripts/revision/build_venue_variants.py, and the numbers by
scripts/revision/manuscript_numbers.py. Edits made only in Overleaf are overwritten the
next time the folder is regenerated, so carry any change back into the master.

At acceptance (RASTI section 4.3), the source archive for production is this folder as a
.zip, with main.bbl added from a final compile.
