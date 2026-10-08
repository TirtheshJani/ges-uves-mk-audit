# Reproducibility appendix

Companion to `README.md`. Pointers, inputs, outputs, seeds and run-time expectations for every
step that produces a number in the manuscript. Run all commands from the repository root.

## Environment

The revision analyses were run in a conda environment with Python 3.12.14, numpy 2.5.3, pandas 3.0.6,
scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, shap 0.52.0, statsmodels 0.15.0, astropy 8.0.1,
h5py 3.16.0, matplotlib 3.11.2 and pytest 9.1.1. The minimum versions that the code needs are in
`pyproject.toml` and `requirements.txt` (Python 3.10 or newer). The test suite is
`python -m pytest -q -p no:cacheprovider tests`; 210 tests passed and 5 skipped on 2026-10-08 (the Pickles-template tests skip when the templates are unreadable). The machine used has
16 cores and about 15 GiB of RAM; keep LightGBM `n_jobs` at 4 or below when memory is tight, and do not
run two of the heavy steps below at once.

## Seeds

`random_state = 42` throughout: LightGBM, `StratifiedKFold`, `StratifiedGroupKFold`, permutation
importance, the stratified SHAP subsample, DBSCAN grouping, bootstrap intervals and random-window
draws (`SEED = 42` in each of `scale_decomposition.py`, `retrain_normalisation.py`,
`luminosity_kiel.py` and `ablation_recalibrated.py`; `--seed 42` in `ablation_stratified.py`). The deposited
model is reproduced exactly by the production retrain in `retrain_normalisation.py` (macro-F1 difference
0, identical confusion matrix, best iteration 120).

## Deposited inputs

| File | Content |
| --- | --- |
| `artifacts/features.npz` | feature matrix `X` (3032 spectra, 696 bins), labels `y`, `wave_centers`, train/val/test indices, `gap_mask`, sky coordinates, `dwarf_flag`, `boundary_distance_k` |
| `artifacts/lgbm_mk.pkl` | deposited LightGBM model (fitted on the training partition, early stopping on validation) |
| `artifacts/ges_mk_labels.parquet` | label table with GES Teff, log g and [Fe/H], joined to the features by exact (ra_deg, dec_deg) |
| `artifacts/metrics.json` | held-out metrics, cross-validation summaries, boundary-filtered accuracy, spatial-group statistics |

## Revision steps

Run times marked "estimate" were not timed; they follow from the number of model fits and
predictions each step performs (a LightGBM fit takes 5 to 11 s and 5-fold cross-validation about
50 s on this machine, from `artifacts/sensitivity/*.json` and `artifacts/metrics.json`). The
`runtime_s` field that `ablation_recalibrated.py` writes per run gives the measured figure for that step.

| Step | Command | Inputs | Outputs | Expected run time |
| --- | --- | --- | --- | --- |
| Scale decomposition | `python scripts/revision/scale_decomposition.py` | features, model | `artifacts/revision/scale_decomposition.{json,csv}` | under 5 min (estimate) |
| Retraining under alternative normalisations | `python scripts/revision/retrain_normalisation.py` | features, model | `artifacts/revision/retrain_normalisation.json` | 5 to 15 min (estimate: four fits, one 5-fold CV, four ablations with 500 windows) |
| Luminosity mix, Kiel diagram, continuum medians | `python scripts/revision/luminosity_kiel.py` | features, labels, model, `scale_decomposition.json` | `luminosity_mix.{json,csv}`, `submission/kiel_diagram.pdf`, `submission/continuum_medians.pdf` and the copies in `artifacts/figures/` | under 5 min (estimate) |
| Line-set geometry | `python scripts/revision/line_set_table.py` | features | `line_sets.csv`, `line_sets_table.tex` | seconds |
| Recalibrated ablation (2 continua x 2 fills, 5 line sets, 3 classes, 5000 windows, 2000 bootstrap resamples) | `python scripts/revision/ablation_recalibrated.py` | features, model | `ablation_recalibrated.{json,csv}`, `ablation_recalibrated_per_row.npz`, `submission/ablation_bars.pdf`, `artifacts/figures/ablation_bars.pdf` | about 12 to 17 min for the four runs. The recorded run took 436 s (production, interpolation), 131 s (production, constant), 322 s (polynomial_n5, interpolation) and 127 s (polynomial_n5, constant), 1017 s in total (`runs.*.runtime_s`); the Statistics track estimated about 12 min |
| Stratified ablation | `python scripts/revision/ablation_stratified.py` | `ablation_recalibrated_per_row.npz`, features, labels | `ablation_stratified.json` | under 5 min (estimate) |
| Corrected template cross-check | `python scripts/revision/benchmark_corrected.py --pickles-dir <dir>` | features, model, the 131 Pickles FITS files (not in this repository) | `benchmark_corrected.json` | a few minutes (estimate) |
| Manuscript numbers | `python scripts/revision/manuscript_numbers.py` | the JSON artifacts above plus `metrics.json`, `interpret/triangulation_report.json`, `ablation/mg_b_ew_test.json`, `ablation/gate_eval.json`, `sensitivity/{uniform_weights,hyperparam_grid}_mg_b_k.json` | `submission/revision_numbers.tex`, `ablation_table.tex`, `ablation_k_table.tex`, `ablation_strata_table.tex`, `line_sets_table.tex`, `artifacts/revision/manuscript_numbers_index.json` | seconds |
| Static manuscript checks | `python scripts/revision/check_manuscript.py` | `submission/` | text report; exit status 1 on an unresolved reference, citation, file or brace error | seconds |

Order: `scale_decomposition.py` before `luminosity_kiel.py` (the figure reads its JSON);
`ablation_recalibrated.py` before `ablation_stratified.py`; every analysis before
`manuscript_numbers.py`. The ablation JSON files are deposited, so `manuscript_numbers.py` leaves no `[TBD-ablation]` placeholder; if either file is missing the affected macros fall back to that marker.

## Original pipeline

The commands below produced the deposited model and feature matrix. `<PATH>` is the regridded HDF5
store (not redistributed).

```bash
python scripts/build_labels.py --h5-path <PATH> --cache-dir data/ges/catalogs --out artifacts/ges_mk_labels.parquet
python scripts/build_features.py --h5-path <PATH> --labels artifacts/ges_mk_labels.parquet --out artifacts/features.npz
python scripts/train_classifier.py --features artifacts/features.npz --model-out artifacts/lgbm_mk.pkl --metrics-out artifacts/metrics.json
python scripts/run_interpret.py --features artifacts/features.npz --model artifacts/lgbm_mk.pkl --out-dir artifacts/interpret --shap-max-samples 1000 --top-k 20 --seed 42
```

The legacy ablation driver `scripts/ablation.py` and `scripts/ablation_paired.py` reproduce the
deposited `artifacts/ablation/` files with `--null-mode pooled --match-on angstrom
--continuum-fill empirical-median`; the numbers they produce use the pooled null and the constant fill
described in the decision log and are superseded by the recalibrated run for every statement in the
manuscript except the equivalent-width comparison, the fill-sensitivity record and the two robustness
retrains, which do not depend on the null.

## Known gaps

- The spatial-group identifiers behind the StratifiedGroupKFold pass are not stored in
  `features.npz` (its `groups` key holds one value). They are re-derived by
  `scripts/audit_singletons.py`. `metrics.json` records 1926 unique groups under
  `cv_spatial_n_unique_groups` while the singleton audit that the manuscript quotes records 1966;
  the two derivations have not been reconciled.
- The Pickles FITS files and the regridded HDF5 store live in the development copy of the project
  (`docs/folder_provenance.md`) and are not redistributed.
- `artifacts/benchmark/` holds the original cross-check output, which used a template-type map that was
  later corrected; use `artifacts/revision/benchmark_corrected.json`.
