# ges-uves-mk-audit

Masked-line ablation audit of a LightGBM F/G/K (Morgan-Keenan letter) classifier trained on
Gaia-ESO Survey FLAMES-UVES U580 spectra. Version 1.1.0.

Zenodo archive: DOI: to be assigned on Zenodo deposit.

## Abstract

We audit a LightGBM classifier that assigns Gaia-ESO FLAMES-UVES U580 spectra (4800 to 6800
Angstrom) to three effective-temperature classes labelled F, G and K. The classifier reaches
macro-F1 = 0.926 on a held-out test set of 456 spectra. For line sets fixed in advance we replace
the covered bins with a linearly interpolated continuum, score the masked spectra with the unmodified
model, and compare the per-class recall change with a class-matched null built from 5000 bin-matched
random windows; row-level flip counts, an exact McNemar test and the change in the true-class
probability accompany every pair. Masking the Balmer lines lowers F recall by 0.840 and masking Mg b lowers G
recall by 0.257 (0.213 with a constant fill); the Balmer effect size on G depends on the fill (0.917 with
interpolation, 0.387 with a constant fill). K recall is insensitive to every line set: no line set loses more than
3 of 145 K test rows. The full table is in `artifacts/revision/ablation_recalibrated.json` and the manuscript quotes
it through macros.

The K class is mostly giants (72.8 percent) while the F and G classes are almost all dwarfs
(97.9 and 98.2 percent), so a K-versus-G contrast is partly a giant-versus-dwarf contrast. Polynomial
re-normalisation of the test spectra lowers G recall from 0.965 to 0.148. A uniform rescaling of the
unmodified spectra by 1.09 reproduces that drop (G recall 0.217, 127 G rows moving to K), whereas
removing the class-dependent level while keeping the scale costs 0.017 in macro-F1. The collapse is a
train-test scale mismatch, not a class-level continuum shortcut. Retraining on polynomial-normalised
or row-median-normalised spectra gives macro-F1 0.915 and leaves the Mg b effect on K null. An external
cross-check against the Pickles template library, repeated with header-derived template types, reached
an agreement of at most 0.531 against a pre-specified gate of 0.55 and is reported as a failed check.

## Provenance of this repository

This is the curated release copy of a larger development project. `docs/folder_provenance.md`
records the three local copies that existed on 2026-10-01, which one is the release target, what was
ported between them, and which assets stay in the development copy (raw UVES spectra, the Pickles
FITS files and the intermediate HDF5 store, none of which is redistributed here).
`docs/decision_log.md` is the dated decision log that records what was specified before the test
set was scored and what changed afterwards; the manuscript's deviations-from-plan section is
written from it. The manuscript source is `submission/manuscript.tex`; the version that was reviewed
before this revision is kept as `docs/manuscript_pre_revision_2026-10-01.tex`.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10 or newer is required.

## Quick start

Run everything from the repository root.

```bash
# 1. Unit and CLI tests (210 passed, 5 skipped on 2026-10-08; the Pickles-template tests skip when the templates are unreadable)
python -m pytest -q -p no:cacheprovider tests

# 2. Revision analyses (inputs: artifacts/features.npz, artifacts/lgbm_mk.pkl, artifacts/ges_mk_labels.parquet)
python scripts/revision/scale_decomposition.py        # artifacts/revision/scale_decomposition.{json,csv}
python scripts/revision/retrain_normalisation.py      # artifacts/revision/retrain_normalisation.json
python scripts/revision/luminosity_kiel.py            # luminosity_mix.{json,csv}, Kiel and continuum-median figures
python scripts/revision/line_set_table.py             # line_sets.csv, line_sets_table.tex
python scripts/revision/ablation_recalibrated.py      # ablation_recalibrated.{json,csv}, per-row .npz, ablation_bars.pdf (about 12 to 17 min, four runs)
python scripts/revision/ablation_stratified.py        # ablation_stratified.json (needs the per-row .npz above)
python scripts/revision/benchmark_corrected.py --pickles-dir <dir with pickles_uk_1.fits ... pickles_uk_131.fits>

# 3. Manuscript numbers and static checks
python scripts/revision/manuscript_numbers.py         # submission/revision_numbers.tex and the generated tables
python scripts/revision/check_manuscript.py           # braces, refs, citations, macros, open slots
```

`scripts/revision/manuscript_numbers.py` writes one LaTeX macro per quoted number from the JSON
artifacts and `artifacts/revision/manuscript_numbers_index.json` maps every macro to its source file
and field. Macros that wait for the ablation artifacts print as `[TBD-ablation]` until
`ablation_recalibrated.json` and `ablation_stratified.json` exist; both files are now deposited, so the generated
macros and tables contain no placeholders. Seeds, inputs, outputs and
expected run times for each step are in `docs/reproducibility_appendix.md`.

## Original pipeline

The deposited model and feature matrix were produced by the scripts below. They are listed for
completeness; the revision analyses above start from the deposited `artifacts/features.npz` and
`artifacts/lgbm_mk.pkl`.

1. `scripts/build_labels.py` resolves Gaia-ESO recommended parameters through the ESO TAP service and
   builds the F/G/K label set.
2. `scripts/build_features.py` extracts the rebinned, continuum-normalised feature matrix from the
   UVES HDF5 store, with the inter-chip-gap mask and per-row sky coordinates.
3. `scripts/train_classifier.py` fits the LightGBM model and writes `metrics.json` (held-out test,
   5-fold StratifiedKFold, StratifiedGroupKFold over spatial groups).
4. `scripts/run_interpret.py` computes permutation importance, TreeSHAP and sliding-window occlusion on
   the validation split.
5. `scripts/ablation.py` and `scripts/ablation_paired.py` are the original masked-line ablation drivers
   (pooled null, constant fill); `src/interpret/ablation.py` holds the ablation module they and the
   recalibrated run share.
6. `scripts/measure_ew.py`, `scripts/audit_singletons.py`, `scripts/independent_continuum_check.py`,
   `scripts/appendix_a2_polynomial_diagnostics.py`, `scripts/sensitivity_retrains.py` and
   `scripts/run_baseline.py` produce the equivalent-width, spatial-group, continuum, uniform-weight and
   hyperparameter-grid, and kNN-baseline artifacts under `artifacts/`.
7. `scripts/run_benchmark.py` is the original Pickles cross-check. Its deposited output
   (`artifacts/benchmark/`) was produced with a template-type map that has since been corrected and is
   superseded by `artifacts/revision/benchmark_corrected.json`.
8. `scripts/make_figure.py`, `scripts/figure_continuum_medians.py` and `scripts/build_manuscript_dossier.py`
   assemble the original figures and the numerical-claims dossier. The manuscript now draws its numbers
   from `scripts/revision/manuscript_numbers.py` instead of the dossier.

Random seeds are pinned at 42 across LightGBM, scikit-learn, NumPy and SHAP.

## Data provenance

- **Gaia-ESO Survey recommended parameters**: ESO Science Archive TAP service at
  `https://archive.eso.org/tap_cat`, table `safcat."GES_DR5_1_V1"`, citing Hourihane et al. 2023,
  A&A 676, A129.
- **Spectra**: UVES U580 setup, blue and red arms stitched, 4800 to 6800 Angstrom in air wavelengths,
  continuum-normalised per the Gaia-ESO pipeline (Sacco et al. 2014, A&A 565, A113). The inter-chip gap
  is preserved and masked downstream (`gap_mask` in `features.npz`). Public through the ESO Science
  Archive Facility under Programme 188.B-3002. This repository does not redistribute raw spectra; fetch
  them with `scripts/fetch_ges_parallel.py` and `scripts/fetch_ges_blue.py`, then regrid with
  `scripts/build_hdf5_phase3.py`.
- **Labels**: effective-temperature bins on the dwarf scale of Pecaut and Mamajek (2013), named with MK
  letters. They are not spectroscopic MK types and carry no luminosity class.
- **Benchmark templates**: Pickles 1998 UVKLIB library (PASP 110, 863), 131 spectra, from the STScI HLSP
  reference-atlases mirror. The spectral type of each template is read from the FITS header
  (`COMMENT1`). The library is not redistributed here.
- **Atomic line rest wavelengths**: NIST Atomic Spectra Database, air wavelengths.

## Repository layout

```
ges-uves-mk-audit/
|-- src/
|   |-- interpret/      Library modules: classifier, features, labels, ablation, occlusion,
|   |                   line matching, importance, triangulation, plotting, benchmark.
|   |-- preprocess/     HDF5 build, continuum normalisation, regridding.
|   |-- fetch/          ESO TAP-service client and manifest builders.
|   `-- utils/          HDF5 and cross-match helpers.
|-- scripts/            Pipeline drivers (see above).
|   `-- revision/       Revision analyses, manuscript number generator, static manuscript checks.
|-- tests/              pytest suite (unit and CLI smoke tests).
|-- submission/         LaTeX manuscript, bibliography, figures, generated macro and table files.
|-- artifacts/          Model, feature matrix, JSON and CSV results, figures.
|   `-- revision/       Outputs of scripts/revision/.
|-- docs/               Decision log, folder provenance, reproducibility appendix, earlier drafts.
|-- CITATION.cff, .zenodo.json, LICENSE (MIT), pyproject.toml, requirements.txt
```

## License

MIT. See `LICENSE`.

## Citation

If you use this software or the deposited artifacts, please cite both the Zenodo archive
(`CITATION.cff`; DOI: to be assigned on Zenodo deposit) and the accompanying paper, whose reference
will be added to `CITATION.cff` after submission.
