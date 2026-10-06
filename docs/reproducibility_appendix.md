# Reproducibility appendix: stellar-mk-audit

Pointers only. No narrative. Companion to `docs/audit_dossier.md`.

## Repo state

- HEAD commit: `75680f0 phase3: gate accepted via pivot pathway (Decision 30)`
- Branch: `main`
- Recent commits per phase (most recent first):
  - `75680f0` phase3: gate accepted via pivot pathway (Decision 30)
  - `206fbd6` phase3: triangulation (jaccard perm-shap=0.4815, physicist=PIVOT)
  - `6b56249` phase2: gate accepted (macro-f1=0.9262; cv_spatial_mean=0.9028)
  - `b792d53` phase2: spatial-CV sensitivity (Decision 29; cv_spatial_mean=0.9028)
  - `e6c2954` phase2: classifier (macro-f1=0.9262, recall-ok)
  - `b817185` phase1: gate accepted (n=3032, n_bins=696, classes=FGK; Decisions 27 and 28 logged)
  - `8099c0e` phase1: add gap_mask to features.npz (Decision 28; Physicist-post remediation)
  - `42abbf7` phase1: features and labels (n=3032, n_bins=696, classes=FGK)
  - `e98bbab` phase0: backfill commit hash in acceptance_log
  - `d4b4090` phase0: probes complete (n_covered=3224, F:691 G:2031 K:1233; A dropped)
  - `2730c67` phase0-recovery: add blue chip fetcher and Phase 3 HDF5 builder (Decision 24)
  - `f41a2f5` phase0-prep: parallel fetch wrapper with ADQL/URL fixes (Decision 23)
  - `69f16dd` phase0a: hotfix VRAD column name and async TAP for full row return
  - `3d929a8` phase0a: gate pass (reviewer PASS, physicist-post PASS, advisories addressed)
  - `42c3c8c` phase0a: replace VizieR catalog with ESO TAP GES_DR5_1_V1
  - `7f67b11` phase-1: repo health (pyproject, readme, decisions log, vizier verified)

## Environment

- Python 3.11.6 in `.venv`
- Core dependencies (`pyproject.toml`): `numpy>=1.24`, `pandas>=2.0`,
  `requests>=2.31`, `astropy>=5.2`, `pyarrow>=14.0`, `h5py>=3.10`,
  `pyyaml>=6.0`, `matplotlib>=3.8`, `tqdm>=4.65`, `scipy>=1.10`,
  `lightgbm>=4.0`, `shap>=0.42`, `scikit-learn>=1.3`, `astroquery>=0.4.7`
- Dev extras: `pytest>=7.4`, `pytest-cov>=4.1`, `ruff>=0.4`, `black>=24.3`,
  `mypy>=1.8`

## Seeds

`random_state = 42` throughout: `LGBMClassifier`, `StratifiedKFold(n_splits=5,
random_state=42)`, `StratifiedGroupKFold(n_splits=5, random_state=42)`,
`permutation_importance(scoring="accuracy", n_repeats=10, random_state=42)`,
`stratified_subsample(max_n=1000, seed=42)`, `DBSCAN(eps=0.1, min_samples=5,
metric='haversine')`. Bootstrap CIs use `seed=42`.

## Gate verdicts (one line per phase, from `artifacts/acceptance_log.json`)

- Phase minus-1 (2026-04-19, `7f67b11`): Reviewer PASS, Physicist PASS;
  VizieR probe DEFERRED.
- Phase 0a (2026-04-29, `42c3c8c`): Reviewer PASS, Physicist PASS; 93 tests
  pass.
- Phase 0 (2026-04-30, `d4b4090`): Reviewer PASS, Physicist PASS; n_covered =
  3224, F=691, G=2031, K=1233, A dropped per Decision 26.
- Phase 1 (2026-04-30, `8099c0e`): Reviewer PASS (iter-2), Physicist PASS
  (iter-2 after Decision 28 gap_mask).
- Phase 2 (2026-04-30, `b792d53`): Reviewer PASS (iter-2), Physicist PASS
  (iter-2 after Decision 29 spatial CV); macro_f1 = 0.9262.
- Phase 3 (2026-04-30, `206fbd6`): Reviewer PASS, Physicist PASS-WITH-PIVOT;
  Jaccard perm-vs-shap = 0.4815; Decision 30 logged.

## Log files (path; size)

- `artifacts/build_labels.log`; 1018 bytes
- `artifacts/build_features.log`; 3094 bytes
- `artifacts/build_hdf5.log`; 1762 bytes
- `artifacts/coverage_probe.log`; 107 bytes
- `artifacts/class_count_probe.log`; 536 bytes
- `artifacts/train_classifier.log`; 8776 bytes
- `artifacts/run_interpret.log`; 1662193 bytes (1.6 MB; use Grep, not full
  Read)
- `artifacts/fetch_ges.log`; 3750 bytes
- `artifacts/fetch_ges_blue.log`; 3496 bytes
- `artifacts/fetch_ges_resume.log`; 41 bytes
- `artifacts/ges_catalog_fetch.log`; 1105 bytes
- `artifacts/starlist_build.log`; 345 bytes
- `artifacts/eso_tap_probe.json.log`; 3579 bytes
- `artifacts/vizier_probe_phase0.log`; 43809 bytes
- `artifacts/augment_radec.log`; 663 bytes

## Artefact paths

- `artifacts/ges_mk_labels.parquet`; 176680 bytes; 3955 rows, 10 columns;
  per-class F=691, G=2031, K=1233
- `artifacts/ges_mk_labels.stats.json`; ghost audit, n_dropped_e_teff
- `artifacts/features.npz`; 7103429 bytes; 13 keys including `gap_mask`,
  `ra_deg`, `dec_deg`
- `artifacts/lgbm_mk.pkl`; 1768094 bytes; final LightGBM fit on train+val
  per Decision 21
- `artifacts/metrics.json`; held-out test, 5-fold StratifiedKFold,
  StratifiedGroupKFold spatial-CV blocks, boundary-filtered accuracy
- `artifacts/interpret/perm_importance.npz`; global 1D ranking, shape (696,)
- `artifacts/interpret/shap_values.npz`; per-class, shape (3, 454, 696)
- `artifacts/interpret/occlusion.npz`; sliding-window occlusion on val
- `artifacts/interpret/shap_stability.json`; per-class stability scores
- `artifacts/interpret/triangulation_report.json`; pairwise / three-way
  Jaccard, per-class top-20 post gap-mask, pre-mask diagnostic, red flags
- `artifacts/figures/importance_overlay_per_class.pdf`; 31272 bytes; 3 panels
  (F/G/K), 300 DPI, MK_LINES annotated, gap-mask shaded
- `artifacts/acceptance_log.json`; reproducibility contract, 6 entries

## Reproduction commands

Verbatim from auditplan sections 5.3, 5.4, 5.5; HDF5 path placeholder is
`<PATH>` (project default `data/common/processed/regridded_spectra.h5`).

```bash
mkdir -p artifacts

python scripts/build_labels.py \
    --h5-path <PATH> \
    --cache-dir data/ges/catalogs \
    --out artifacts/ges_mk_labels.parquet \
    2>&1 | tee artifacts/build_labels.log

python scripts/build_features.py \
    --h5-path <PATH> \
    --labels artifacts/ges_mk_labels.parquet \
    --out artifacts/features.npz \
    2>&1 | tee artifacts/build_features.log

python scripts/train_classifier.py \
    --features artifacts/features.npz \
    --model-out artifacts/lgbm_mk.pkl \
    --metrics-out artifacts/metrics.json \
    2>&1 | tee artifacts/train_classifier.log

python scripts/run_interpret.py \
    --features artifacts/features.npz \
    --model artifacts/lgbm_mk.pkl \
    --out-dir artifacts/interpret \
    --shap-max-samples 1000 \
    --top-k 20 \
    --seed 42 \
    -v \
    2>&1 | tee artifacts/run_interpret.log
```

Test suite: `pytest tests/ -x -v` (current state 80 to 93 passing across
phases; latest Phase 3 ran `pytest tests/test_triangulation.py
tests/test_plotting.py -v`, 17 passed in 0.67 s).
