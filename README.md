# ges-uves-mk-audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23482244.svg)](https://doi.org/10.5281/zenodo.23482244)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Code, trained model and feature matrix for the paper

> T. Jani, *A masked-line ablation audit of a LightGBM F/G/K temperature-class classifier for
> Gaia-ESO UVES spectra: what survives, and a train-test scale mismatch*. Submitted to RAS
> Techniques and Instruments, October 2026.

The repository asks one question of a trained stellar classifier: which spectral lines does it
actually depend on? It answers by masking fixed line windows, re-scoring the unmodified model, and
comparing each per-class recall change with a class-matched null built from random windows of the
same width. Every number in the paper is regenerated from the JSON artifacts deposited here.

Archive: [10.5281/zenodo.23482244](https://doi.org/10.5281/zenodo.23482244) (concept DOI, always
resolves to the latest version). Current version: 1.2.1.

## What the paper finds

The classifier is LightGBM, trained on Gaia-ESO FLAMES-UVES U580 spectra (4800 to 6800 Angstrom) and
labelled with three effective-temperature classes named F, G and K. It reaches macro-F1 0.926 on 456
held-out spectra.

Two dependencies survive the audit. Masking the Balmer windows removes almost every F and G
prediction (recall change -0.840 for F, -0.917 for G), and masking Mg b costs 59 of 230 G rows
(recall change -0.257). Both effects move rows the way a temperature estimator would, and the Mg b
effect holds when the masked bins are filled with a constant instead of interpolated.

The K class behaves differently. No line set lowers K recall by more than 3 of 145 rows, and neither
do the 100 K bins with the highest attribution when they are interpolated over. A constant fill over
those same bins does lower K recall (by 0.166), so the audit is able to register a K dependence; it
just does not find one at the catalogued lines. Part of the reason is the sample: 72.8 per cent of K
stars are giants while F and G are almost all dwarfs, so "K versus G" is partly "giant versus dwarf".

Polynomial re-normalisation of the test spectra drops G recall from 0.965 to 0.148. A uniform
rescaling of the unmodified spectra by 1.09 reproduces most of that drop, and retraining on
re-normalised spectra restores macro-F1 to 0.915. The collapse is a train-test scale mismatch rather
than a continuum shortcut.

Three further checks are reported. Field-grouped cross-validation differs from the plain estimate
by 0.006 in macro-F1, inside the 0.010 split-to-split spread, so leakage between spectra from the same
field is not detectable at that resolution. Masking the 6283 Angstrom diffuse-band window as a
sixth line set changes G recall by at most 0.022 (5 of 230 rows). An external cross-check against
the Pickles template library reached an agreement of 0.531 against a pre-specified gate of 0.55 and
is reported as a failed check.

## Install

```bash
git clone https://github.com/TirtheshJani/ges-uves-mk-audit
cd ges-uves-mk-audit
pip install -e ".[dev]"
```

Python 3.10 or newer is required. The deposited model `artifacts/lgbm_mk.pkl` was written with
scikit-learn 1.8.0 (the version is recorded in the pickle). The test suite passes on Python 3.12 and
3.13.

To work from the archived release instead of GitHub, download the zip from the Zenodo record. The
model and feature matrix have these SHA-256 checksums:

```
06336154ac1cbd632a2f9d988a531505f748f4e4dc5bfb3cc7c0491faf717d4a  artifacts/features.npz
eba1865cefcca1c55fea8f3ec8cefa328d9783cfd6712f5a11e7b92485059376  artifacts/lgbm_mk.pkl
```

## Reproduce the paper

Run everything from the repository root. The analyses start from the deposited
`artifacts/features.npz`, `artifacts/lgbm_mk.pkl` and `artifacts/ges_mk_labels.parquet`; no raw
spectra are needed.

```bash
# 1. Tests (212 passed, 5 skipped on 2026-10-22; the 5 Pickles-template tests skip without the templates)
python -m pytest -q -p no:cacheprovider tests

# 2. Analyses (outputs go to artifacts/revision/)
python scripts/revision/scale_decomposition.py        # scale_decomposition.{json,csv}
python scripts/revision/retrain_normalisation.py      # retrain_normalisation.json
python scripts/revision/luminosity_kiel.py            # luminosity_mix.{json,csv}, Kiel and continuum-median figures
python scripts/revision/line_set_table.py             # line_sets.csv, line_sets_table.tex
python scripts/revision/ablation_recalibrated.py      # ablation_recalibrated.{json,csv}, per-row .npz, ablation_bars.pdf (12 to 17 min)
python scripts/revision/ablation_stratified.py        # ablation_stratified.json (needs the per-row .npz above)
python scripts/revision/benchmark_corrected.py --pickles-dir <dir with pickles_uk_1.fits ... pickles_uk_131.fits>
python scripts/revision/cv_field_groups.py all        # field catalogue, groups and cv_field_groups.json (ESO TAP query, about 3 min)
python scripts/revision/k_positive_control.py         # k_positive_control.json (about 45 min)
python scripts/revision/dib_6283_ablation.py          # dib_6283_ablation.json (11 to 15 min)
python scripts/revision/boundary_filtered.py          # boundary_filtered.json

# 3. Numbers, checks and builds
python scripts/revision/manuscript_numbers.py         # submission/revision_numbers.tex and the generated tables
python scripts/revision/verify_numbers.py             # every quoted number traced to its artifact
python scripts/revision/check_manuscript.py           # braces, refs, citations, macros, open slots
python scripts/revision/build_venue_variants.py       # journal builds from the master manuscript
```

Seeds are pinned at 42 across LightGBM, scikit-learn, NumPy and SHAP. Inputs, outputs, seeds and
run times for each step are in `docs/reproducibility_appendix.md`. `cv_field_groups.py` and
`k_positive_control.py` write logs locally; `*.log` files are git-ignored.

`manuscript_numbers.py` writes one LaTeX macro per quoted number, and
`artifacts/revision/manuscript_numbers_index.json` maps each macro to its source file and field. If
you change an analysis and re-run it, the paper picks up the new value on the next build.

## Manuscript builds

`submission/manuscript.tex` is the single master (article class, used for arXiv).
`build_venue_variants.py` generates the journal versions from it:

- `submission/rasti_overleaf/`: a self-contained RAS Techniques and Instruments project with the
  journal's `rasti.cls` and `rasti.bst`. It compiles on Overleaf with the defaults (pdfLaTeX, main
  document `main.tex`).
- `submission/manuscript_elsarticle.tex`: an Elsevier `elsarticle` build.

`make_arxiv_bundle.py` collects the arXiv source into `submission/arxiv/` (git-ignored), and
`set_doi.py` writes a Zenodo DOI into the manuscript, this README and the metadata files. Edit only
the master; the generated files are overwritten on each run.

## Original pipeline

The deposited model and feature matrix were produced by the scripts below. They are listed for
completeness. The fetch and preprocess stages (`src/fetch/`, `src/preprocess/`,
`scripts/fetch_ges_parallel.py`, `scripts/fetch_ges_blue.py`, `scripts/build_hdf5_phase3.py`) need
the raw ESO spectra and are not required to reproduce the paper.

1. `scripts/build_labels.py` resolves Gaia-ESO recommended parameters through the ESO TAP service and
   builds the F/G/K label set.
2. `scripts/build_features.py` extracts the rebinned, continuum-normalised feature matrix from the
   UVES HDF5 store, with the inter-chip-gap mask and per-row sky coordinates.
3. `scripts/train_classifier.py` fits the LightGBM model and writes `metrics.json` (held-out test,
   5-fold StratifiedKFold, StratifiedGroupKFold over spatial groups).
4. `scripts/run_interpret.py` computes permutation importance, TreeSHAP and sliding-window occlusion on
   the validation split.
5. `scripts/ablation.py` and `scripts/ablation_paired.py` are the original ablation drivers (pooled
   null, constant fill). `src/interpret/ablation.py` holds the module they share with the current run.
6. `scripts/measure_ew.py`, `scripts/audit_singletons.py`, `scripts/independent_continuum_check.py`,
   `scripts/appendix_a2_polynomial_diagnostics.py`, `scripts/sensitivity_retrains.py` and
   `scripts/run_baseline.py` produce the equivalent-width, spatial-group, continuum, retraining and
   kNN-baseline artifacts under `artifacts/`.
7. `scripts/run_benchmark.py` is the original Pickles cross-check. Its output in `artifacts/benchmark/`
   used a template-type map that was later corrected; `artifacts/revision/benchmark_corrected.json`
   supersedes it.
8. `scripts/make_figure.py`, `scripts/figure_continuum_medians.py` and
   `scripts/build_manuscript_dossier.py` assemble the original figures and the numerical-claims
   dossier. The manuscript now takes its numbers from `manuscript_numbers.py`.

## Data provenance

- Gaia-ESO Survey recommended parameters: ESO Science Archive TAP service
  (`https://archive.eso.org/tap_cat`, table `safcat."GES_DR5_1_V1"`), Hourihane et al. 2023, A&A 676,
  A129.
- Spectra: UVES U580 setting, 4800 to 6800 Angstrom in air wavelengths, continuum-normalised by the
  Gaia-ESO pipeline (Sacco et al. 2014, A&A 565, A113). FLAMES feeds only the UVES red arm, and the two
  CCDs of its mosaic are stitched; the inter-chip gap is kept and masked downstream (`gap_mask` in
  `features.npz`). The spectra are public through the ESO Science Archive Facility under Programme
  188.B-3002 and are not redistributed here.
- Labels: effective-temperature bins on the dwarf scale of Pecaut and Mamajek (2013), named with
  Morgan-Keenan letters. They are not spectroscopic MK types and carry no luminosity class.
- Benchmark templates: Pickles 1998 UVKLIB library (PASP 110, 863), 131 spectra, from the STScI HLSP
  reference-atlases mirror
  (https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/). Spectral types are
  read from the FITS header (`COMMENT1`). Not redistributed here.
- Atomic line rest wavelengths: NIST Atomic Spectra Database, air wavelengths.

If you use the feature matrix, please also cite the Gaia-ESO Survey (Gilmore et al. 2022, A&A 666,
A120; Randich et al. 2022, A&A 666, A121).

## Repository layout

```
ges-uves-mk-audit/
|-- src/
|   |-- interpret/      classifier, features, labels, ablation, occlusion, line matching,
|   |                   importance, triangulation, plotting, benchmark
|   |-- preprocess/     HDF5 build, continuum normalisation, regridding
|   |-- fetch/          ESO TAP-service client and manifest builders
|   `-- utils/          HDF5 and cross-match helpers
|-- scripts/            original pipeline drivers
|   `-- revision/       paper analyses, number generator, checks, venue builds, DOI back-fill
|-- tests/              pytest suite (unit and CLI smoke tests)
|-- submission/         master manuscript, bibliography, figures, generated macros and tables,
|                       rasti_overleaf/ project, elsarticle build
|-- artifacts/          model, feature matrix, JSON and CSV results, figures
|   `-- revision/       outputs of scripts/revision/
|-- docs/               decision log, folder provenance, reproducibility appendix, earlier drafts
|-- CITATION.cff, .zenodo.json, LICENSE (MIT), pyproject.toml, requirements.txt
```

`docs/decision_log.md` records, with dates, what was fixed before the test set was scored and what
changed afterwards; the paper's deviations-from-plan section is written from it.
`docs/folder_provenance.md` explains how this curated copy relates to the development project and
which large assets (raw spectra, Pickles FITS files, the intermediate HDF5 store) stay out of it.

## Citation

Please cite both the software archive and the paper. GitHub's "Cite this repository" button reads
`CITATION.cff`. In BibTeX:

```bibtex
@software{jani_ges_uves_mk_audit,
  author    = {Jani, Tirthesh},
  title     = {ges-uves-mk-audit: masked-line ablation audit of a LightGBM F/G/K
               temperature-class classifier for Gaia-ESO UVES spectra
               (code, trained model and feature matrix)},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23482244},
  url       = {https://doi.org/10.5281/zenodo.23482244}
}

@unpublished{jani_2026_mk_audit,
  author = {Jani, Tirthesh},
  title  = {A masked-line ablation audit of a LightGBM F/G/K temperature-class classifier
            for Gaia-ESO UVES spectra: what survives, and a train-test scale mismatch},
  note   = {Submitted to RAS Techniques and Instruments},
  year   = {2026}
}
```

The paper entry will be updated with the arXiv identifier and journal reference when they exist.

## License

MIT. See `LICENSE`. The RASTI class and bibliography style in `submission/rasti_overleaf/` are
distributed by the journal under the LaTeX Project Public License.

## Contact

Tirthesh Jani, Department of Computer Science, Lakehead University, Barrie, Ontario.
tirtheshjani@gmail.com, ORCID [0009-0005-5965-4409](https://orcid.org/0009-0005-5965-4409)
