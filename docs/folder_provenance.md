# Folder provenance and release target

Three local copies of this project existed on 2026-10-01. This note records
what each one is, which is the release target, and what was ported between
them. It exists so that nobody has to re-derive the comparison.

## The three folders

| Folder | Role | Last content change | Size | Git |
| --- | --- | --- | --- | --- |
| `GitHub/ges-uves-mk-audit` | **Release target.** Curated public repository: three-class (F/G/K) pipeline, ESO TAP catalogue source, deposited model and feature matrix, 183-test suite. | 2026-05-18 (initial public commit) | 14 MB | single commit `Initial public release` |
| `GitHub/stellar-mk-audit` | Development repository. Full working tree including raw data (7208 UVES FITS, 131 Pickles FITS, regridded HDF5), `decisions.md`, `release/` bundles (arXiv, Zenodo, reviewer responses). | 2026-05-19 00:12 | 7.9 GB | rewritten history; tags `pre-history-scrub`, `v1.0-submission` |
| `Documents/stellar-mk-audit` | Older clone of the development repository (README still describes the A/F/G/K, VizieR-era pipeline). Contains agent configuration files and personal manuscript drafts (`TJs manuscripts/`). | 2026-05-07 | 7.9 GB | clone of origin/main, 31 reflog entries |

The two `stellar-mk-audit` copies share the same raw data (byte-identical
FITS counts and sizes). `GitHub/stellar-mk-audit` is strictly newer than
`Documents/stellar-mk-audit` on every differing file, so the latter is
superseded and is not referenced further.

## What the development copy had that the release copy lacked

Files in `GitHub/stellar-mk-audit` that were newer than their counterpart in
`ges-uves-mk-audit` (all dated 2026-05-18 23:52 to 2026-05-19 00:12, after the
public release was cut at 21:54 on 2026-05-18):

- `submission/manuscript.tex` and `manuscript_citations.bib`: the version the
  peer review read (Section 4.4 "The continuum-level shortcut on K and G",
  Section 5.2 "Future work: retraining ..."). The release copy was an older
  draft, as the review noted. The superseded release draft is kept at
  `docs/manuscript_release_2026-05-18_superseded.tex` for reference.
- `src/interpret/plotting.py`, `scripts/make_figure.py`,
  `scripts/figure_continuum_medians.py`: display-name tables and the
  regenerated figure set.
- All six `submission/*.pdf` figures and the eight `artifacts/figures/*.pdf`.
- `docs/reproducibility_appendix.md` (cited in the manuscript's Data and code
  availability section but absent from the release copy).
- `decisions.md`: the dated decision log (42 entries). Ported to
  `docs/decision_log.md`.

All of the above were copied into `ges-uves-mk-audit` on 2026-10-01. After
porting, the test suite passes (183 tests).

## Pre-specification evidence

The manuscript cites commits `b021595`, `fa5576f` and `0682aa5`. They do not
exist in the public repository's history (one commit), but all three exist in
the object store of `GitHub/stellar-mk-audit`, reachable from the tag
`pre-history-scrub`:

| Commit | Author date (UTC) | Message |
| --- | --- | --- |
| `b021595ff37f46c3a18889f492849c2cb8b3ac11` | 2026-04-30 00:47 (2026-04-29 20:47 EDT) | `phase0-prep: log Phase 2 CV strategy and Phase 4 Ca I gate decisions` (Decisions 21 and 22) |
| `fa5576f7b05d3ef84eb6f959b787f5fb9d1c5c2a` | 2026-04-30 15:53 (11:53 EDT) | `phase2: classifier (macro-f1=0.9262, recall-ok)` |
| `0682aa5fcc0e4dc99c40def6597264df4e3aeb07` | 2026-04-30 23:52 (19:52 EDT) | `phase4: ablation (Decision 31; gap_mask + 500 controls + continuum-fill sensitivity)` |

The dates support the manuscript's timeline (line sets logged the evening
before the test set was first scored; pair assignments locked eight hours
after). To make this verifiable by readers, the author should run, from
`GitHub/stellar-mk-audit`:

```
git bundle create ges-uves-mk-audit-history.bundle --all
```

and deposit the bundle alongside the Zenodo archive (see
`RELEASE_CHECKLIST.md`). Until then the manuscript cites the dated decision
log (`docs/decision_log.md`) rather than the hashes.

Note on content: Decision 22 (2026-04-29) named `H_balmer`, `Mg_b`, `Na_D` as
the three headline *line sets* with Ca I secondary. Decision 31 (2026-04-30)
froze the headline *pairs* as (H Balmer, F), (Mg b, G), (Mg b, K) and moved
(Na D, K) and (Ca I, K) to "pivot" status after the Phase 3 triangulation
(validation split) predicted a null on them. The manuscript's
deviations-from-plan paragraph states this.

## Assets that stay in the development copy

- `data/pickles/pickles_uk_1.fits` ... `pickles_uk_131.fits`: the full STScI
  UVKLIB set. Used by `scripts/run_benchmark.py --pickles-dir`; not
  redistributed in the release repository.
- `data/common/processed/regridded_spectra.h5` (90 MB): the intermediate HDF5
  store. Its wavelength sampling is reported in the manuscript's Section 2.4.
- `data/ges/uves/*.fits` (7208 files, 7.8 GB): raw UVES Phase 3 products from
  the ESO archive. Fetchable with `scripts/fetch_ges_parallel.py`.
- `release/zenodo/`: the bundle staged for the first (unminted) Zenodo
  deposit. It is superseded by the `artifacts/` tree of the release
  repository after this revision.
