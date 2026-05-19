# ges-uves-mk-audit

Interventional masked-line ablation audit of a LightGBM Morgan-Keenan (MK)
spectral-class classifier trained on Gaia-ESO Survey FLAMES-UVES U580 spectra.

## Abstract

We audit a LightGBM classifier of MK spectral classes F, G, and K trained on
Gaia-ESO Survey FLAMES-UVES U580 spectra in the 4800 to 6800 Angstrom window.
The classifier reaches macro-F1 = 0.926 on a held-out test set of 456
spectra. We then ask whether that performance comes from canonical MK
diagnostics or from something else, using interventional masked-line
ablation. For each pre-specified (line set, class) pair we mask the bins
covering a diagnostic line with a continuum value, score the masked spectra
against the unmodified model, and compare the resulting accuracy drop to
a null distribution of 500 random non-line windows of the same total width.

Two of three pre-specified headline pairs reject the random-window null at
the Bonferroni-corrected per-pair alpha = 0.0033 (family-wise alpha = 0.01).
Masking the Balmer series collapses F-class accuracy to zero; masking the
Mg b triplet drops G-class accuracy by 0.21. The third pair, Mg b on K,
returns a null result under the production Gaia-ESO continuum that a
two-one-sided equivalence test confirms within a +/-0.025 indifference
zone. An independent 5th-order Legendre continuum re-derivation shifts the
(Mg b, K) effect to -0.034 (p = 0.002) and collapses G-class baseline
accuracy from 0.97 to 0.15, revealing that the inherited continuum encodes
a per-class continuum-level signal the model exploits as a non-line
shortcut. The K-class result is therefore a multi-leg shortcut combining
an inherited continuum-level signal with substituted Fe I and Cr I
multiplets in the mid-5200 Angstrom region in place of the canonical Mg b,
Na D, and Ca I diagnostics.

The methodological contribution is that interventional masked-line ablation
is a falsifiable per-class audit instrument, sharper than the correlational
tools (permutation importance, TreeSHAP, sliding-window occlusion) that
dominate stellar machine-learning interpretability so far. The intervention
is defined on the input domain, so the audit is portable to other classifier
architectures with no internal-representation hooks required.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10 or newer is required.

## Reproduce

The end-to-end pipeline runs as a sequence of scripts under `scripts/`:

1. `scripts/build_labels.py` resolves Gaia-ESO recommended parameters via
   the ESO TAP service and constructs the F/G/K label set with
   boundary-distance diagnostics.
2. `scripts/build_features.py` extracts the rebinned, continuum-normalized
   feature matrix from the UVES HDF5 store, including the
   inter-chip-gap mask and per-row sky coordinates.
3. `scripts/train_classifier.py` fits the LightGBM model and produces
   `metrics.json` (held-out test + 5-fold StratifiedKFold +
   StratifiedGroupKFold over spatial groups).
4. `scripts/run_interpret.py` computes permutation importance, TreeSHAP,
   and sliding-window occlusion on the validation split.
5. `scripts/ablation.py` runs the interventional masked-line ablation
   with 500 bootstrap resamples and 500 random-window controls per
   line set, plus continuum-fill sensitivity checks.
6. `scripts/ablation_paired.py` computes the paired bootstrap on the
   substantive between-class difference, the row-level flip-rate
   decomposition, and the TOST equivalence test.
7. `scripts/measure_ew.py` measures equivalent widths on the K-class and
   G-class test rows for the saturation argument.
8. `scripts/audit_singletons.py` quantifies how much of the train+val
   partition is in singleton DBSCAN groups (the lower bound on cluster
   leakage from spatial CV).
9. `scripts/independent_continuum_check.py` re-derives the continuum
   with an iteratively sigma-clipped 5th-order Legendre fit and reruns
   the (Mg b, K) ablation.
10. `scripts/appendix_a2_polynomial_diagnostics.py` runs the per-class
    baseline accuracy, F/G/K confusion, and Fe I / Cr I and Mg b
    ablations under the polynomial continuum.
11. `scripts/sensitivity_retrains.py` retrains under uniform class
    weights and across a 3x3 (max_depth, num_leaves) hyperparameter grid.
12. `scripts/run_baseline.py` trains the kNN-on-spectra baseline.
13. `scripts/run_benchmark.py` performs the Pickles 1998 UVKLIB
    chi-squared cross-check under three continuum-normalisation
    conventions.
14. `scripts/make_figure.py` and `scripts/figure_continuum_medians.py`
    assemble the publication figures.
15. `scripts/build_manuscript_dossier.py` regenerates
    `artifacts/manuscript_numbers.csv` and `.md` after any artifact
    refresh.

Random seeds are pinned at `random_state = 42` across LightGBM,
scikit-learn, NumPy, and SHAP. The pipeline is deterministic under
fixed seed and a fixed input HDF5 store.

## Data provenance

- **Gaia-ESO Survey recommended parameters**: ESO Science Archive TAP
  service at `https://archive.eso.org/tap_cat`, table
  `safcat."GES_DR5_1_V1"`, citing Hourihane et al. 2023, A&A 676, A129
  (bibcode `2023A&A...676A.129H`).
- **Spectra**: UVES U580 setup, blue and red arms stitched, covering 4800
  to 6800 Angstrom in air wavelengths, continuum-normalized per the
  Gaia-ESO pipeline convention (Sacco et al. 2014, A&A 565, A113). The
  inter-chip gap at approximately 5770 to 5832 Angstrom is preserved and
  masked downstream (`gap_mask` key in `features.npz`). Public via the
  ESO Science Archive Facility under Programme 188.B-3002.
- **Benchmark templates**: Pickles 1998 UVKLIB stellar flux library
  (PASP 110, 863), retrieved from the STScI HLSP reference-atlases mirror
  at
  `https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/`.
  UVKLIB indices 109 to 131 are luminosity-class duplicates of 86 to 108
  and are excluded from the comparison.
- **Atomic line rest wavelengths**: NIST Atomic Spectra Database
  (Kramida et al. 2023), air wavelengths.

This repository does not redistribute the raw spectra. Fetch them from
the ESO Science Archive Facility using `scripts/fetch_ges_parallel.py`
and `scripts/fetch_ges_blue.py`, then regrid via
`scripts/build_hdf5_phase3.py` to produce the input HDF5 store.

## Repository layout

```
ges-uves-mk-audit/
├── src/
│   ├── interpret/      Library modules: classifier, features, labels,
│   │                   ablation, line-matching, importance,
│   │                   triangulation, plotting.
│   ├── preprocess/     HDF5 build, continuum normalization, regridding.
│   ├── fetch/          ESO TAP-service client + manifest builders.
│   └── utils/          HDF5 and cross-match helpers.
├── scripts/            End-to-end pipeline drivers (see Reproduce).
├── tests/              pytest suite (unit + CLI smoke tests).
├── submission/         LaTeX manuscript, bibliography, and figures.
├── artifacts/          Trained model, feature matrix, ablation/
│                       interpretability/sensitivity/benchmark JSONs
│                       and CSVs, publication figures, and the
│                       manuscript numerical-claims dossier.
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── LICENSE             MIT
```

## License

MIT. See `LICENSE`.

## Citation

If you use this software or the deposited artifacts, please cite both
this repository (`CITATION.cff`) and the accompanying paper. The Zenodo
DOI for the deposited code and data archive will be added to
`CITATION.cff` after the deposit is minted.
