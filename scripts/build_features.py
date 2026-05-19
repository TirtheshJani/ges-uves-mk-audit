#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build the LightGBM feature matrix from the regridded HDF5."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.interpret.features import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_MAX_NAN_FRAC,
    DEFAULT_MIN_COVERED_SPECTRA,
    DEFAULT_MIN_SNR,
    DEFAULT_REBIN,
    DEFAULT_WAVE_MAX,
    DEFAULT_WAVE_MIN,
    build_features,
    coverage_probe,
)

logger = logging.getLogger(__name__)


def _assert_na_d_resolved(wave_centers: np.ndarray) -> None:
    """reversal check: Na D1 (5895.92) and Na D2 (5889.95) must
    fall into different rebinned bins.

    Raises RuntimeError if the bin index containing 5889.95 A equals the
    bin index containing 5895.92 A.
    """
    idx_d2 = int(np.argmin(np.abs(wave_centers - 5889.95)))
    idx_d1 = int(np.argmin(np.abs(wave_centers - 5895.92)))
    if idx_d1 == idx_d2:
        raise RuntimeError(
            "reversal triggered: Na D1 and Na D2 collapsed "
            f"into the same bin (idx={idx_d1}, "
            f"wave_centers[{idx_d1}]={float(wave_centers[idx_d1]):.3f} A)"
        )
    logger.info(
        "Na D resolved: D2(5889.95) -> bin %d (%.3f A); "
        "D1(5895.92) -> bin %d (%.3f A)",
        idx_d2, float(wave_centers[idx_d2]),
        idx_d1, float(wave_centers[idx_d1]),
    )


def _log_gap_region_medians(
    wave_centers: np.ndarray,
    median_imputer: np.ndarray,
    gap_min: float = 5769.0,
    gap_max: float = 5834.0,
) -> None:
    """Log per-bin train-median for the UVES inter-chip gap region.

    Physicist red-line 2: ablation must not interpret gap-imputed
    bins as physics signal, so the per-bin median for bins inside the gap
    is logged here for downstream sanity-checking.
    """
    in_gap = (wave_centers >= gap_min) & (wave_centers <= gap_max)
    n_gap = int(in_gap.sum())
    if n_gap == 0:
        logger.info(
            "no rebinned bins inside UVES inter-chip gap [%.0f, %.0f] A",
            gap_min, gap_max,
        )
        return
    logger.info(
        "UVES inter-chip gap [%.0f, %.0f] A covers %d rebinned bins; "
        "per-bin train-medians:",
        gap_min, gap_max, n_gap,
    )
    for i in np.flatnonzero(in_gap):
        logger.info(
            "  bin %d  wave_center=%.3f A  median=%.4f",
            int(i), float(wave_centers[i]), float(median_imputer[i]),
        )


def compute_gap_mask(
    wave_centers: np.ndarray,
    gap_min: float = 5769.0,
    gap_max: float = 5834.0,
) -> np.ndarray:
    """Return a boolean array marking the UVES U580 inter-chip gap.

     features.npz schema is extended with a boolean ``gap_mask``
    of shape ``(n_bins)`` that is True where ``wave_centers`` lies inside
    the UVES U580 inter-chip dead zone [5769, 5834] A (Sacco et al. 2014,
    2014A&A...565A.113S; Dekker et al. 2000). and consumers
    are required to read and respect this mask in their own phases. only produces it; the classifier in still sees the full feature
    matrix because the mask is metadata, not a transformation.
    """
    mask = (wave_centers >= gap_min) & (wave_centers <= gap_max)
    return np.asarray(mask, dtype=bool)


def _augment_npz_with_gap_mask(
    out_path: Path,
    payload: dict[str, np.ndarray],
    gap_mask: np.ndarray,
) -> None:
    """Re-save the features npz so it carries the new gap_mask key.

    ``build_features()`` already wrote ``out_path`` with the legacy 10-key
    schema. extends the schema to 11 keys; rather than modify
    the (read-only) library function, we re-save the same payload plus
    ``gap_mask`` here.
    """
    augmented: dict[str, np.ndarray] = dict(payload)
    augmented["gap_mask"] = gap_mask
    np.savez_compressed(out_path, **augmented)
    logger.info(
        "augmented %s with gap_mask (n_bins=%d, sum(gap_mask)=%d)",
        out_path, gap_mask.shape[0], int(gap_mask.sum()),
    )


def derive_aligned_radec(
    h5_path: Path,
    labels_parquet: Path,
    payload: dict[str, np.ndarray],
    min_snr: float,
    max_nan_frac: float,
    max_spectra: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the build_features row filter to produce ra_deg/dec_deg
    aligned to the rows in ``payload['X']``.

     features.npz schema is extended with ``ra_deg`` and
    ``dec_deg`` so a SECOND CV using spatial groups can be run at.
    The library function ``build_features`` is read-only per, so this script-level helper replays the same filter chain
    deterministically (same seed, same order of operations) and returns the
    surviving (ra, dec) per row in payload-row order.

    Sanity guard: shape must match payload['X'].shape[0]; ValueError if not.
    """
    import h5py

    labels = pd.read_parquet(labels_parquet)
    if "ra_deg" not in labels.columns or "dec_deg" not in labels.columns:
        raise RuntimeError(
            f"labels parquet missing ra_deg/dec_deg columns: {labels_parquet}"
        )

    with h5py.File(h5_path, "r") as h5:
        source_files = np.array(h5["metadata/source_file"][:], dtype=object)
        source_files = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in source_files],
            dtype=object,
        )
        snr = np.array(h5["metadata/snr_median"][:], dtype=float)

    sf_to_row = {sf: i for i, sf in enumerate(source_files)}
    rows = np.array(
        [sf_to_row.get(sf, -1) for sf in labels["source_file"].to_numpy()],
        dtype=int,
    )
    keep_match = rows >= 0
    labels = labels.loc[keep_match].reset_index(drop=True)
    rows = rows[keep_match]

    snr_ok = snr[rows] >= min_snr
    labels = labels.loc[snr_ok].reset_index(drop=True)
    rows = rows[snr_ok]

    # Replay rebin on the same flux rows to compute the NaN-frac mask
    # exactly as build_features does.
    from src.interpret.features import (
        DEFAULT_REBIN, DEFAULT_WAVE_MAX, DEFAULT_WAVE_MIN, rebin_flux,
    )
    with h5py.File(h5_path, "r") as h5:
        wave = h5["spectra/wavelength"][:]
        flux = h5["spectra/flux"]
        flux_rows = np.empty((len(rows), wave.shape[0]), dtype=np.float32)
        for k, r in enumerate(rows):
            flux_rows[k, :] = flux[r, :]
    X_replay, _ = rebin_flux(
        flux_rows, wave,
        wave_min=DEFAULT_WAVE_MIN, wave_max=DEFAULT_WAVE_MAX,
        rebin_factor=DEFAULT_REBIN,
    )
    nf = np.mean(~np.isfinite(X_replay), axis=1)
    keep_nf = nf <= max_nan_frac
    labels = labels.loc[keep_nf].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if max_spectra is not None and len(labels) > max_spectra:
        n_classes = labels["mk_class"].nunique()
        per_class = max_spectra // max(n_classes, 1)
        take: list[int] = []
        for _, grp in labels.groupby("mk_class"):
            ids = grp.index.to_numpy()
            if len(ids) > per_class:
                take.extend(rng.choice(ids, size=per_class, replace=False).tolist())
            else:
                take.extend(ids.tolist())
        take_arr = np.array(sorted(take))
        labels = labels.loc[take_arr].reset_index(drop=True)

    if len(labels) != payload["X"].shape[0]:
        raise RuntimeError(
            f"derive_aligned_radec replay produced {len(labels)} rows but "
            f"payload['X'] has {payload['X'].shape[0]} rows; row alignment "
            "broken (deterministic replay diverged from build_features)"
        )

    ra_deg = labels["ra_deg"].to_numpy().astype(np.float64)
    dec_deg = labels["dec_deg"].to_numpy().astype(np.float64)
    if not np.all(np.isfinite(ra_deg)) or not np.all(np.isfinite(dec_deg)):
        raise RuntimeError("ra_deg/dec_deg contains non-finite values")
    if ra_deg.min() < 0.0 or ra_deg.max() >= 360.0:
        raise RuntimeError(
            f"ra_deg out of [0, 360): min={ra_deg.min()}, max={ra_deg.max()}"
        )
    if dec_deg.min() < -90.0 or dec_deg.max() > 90.0:
        raise RuntimeError(
            f"dec_deg out of [-90, 90]: min={dec_deg.min()}, max={dec_deg.max()}"
        )
    logger.info(
        "derive_aligned_radec: ra_deg in [%.3f, %.3f], dec_deg in [%.3f, %.3f] "
        "(n=%d)",
        float(ra_deg.min()), float(ra_deg.max()),
        float(dec_deg.min()), float(dec_deg.max()), len(ra_deg),
    )
    return ra_deg, dec_deg


def _augment_npz_with_radec(
    out_path: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> None:
    """Re-save the features npz with ra_deg and dec_deg appended.

     schema extends from 11 keys (gap_mask added in )
    to 13 keys. Loaded back, then re-saved with the new arrays so prior keys
    are preserved verbatim.
    """
    with np.load(out_path, allow_pickle=False) as d:
        existing = {k: d[k] for k in d.files}
    existing["ra_deg"] = ra_deg.astype(np.float64)
    existing["dec_deg"] = dec_deg.astype(np.float64)
    np.savez_compressed(out_path, **existing)
    logger.info(
        "augmented %s with ra_deg (n=%d) and dec_deg (n=%d) per ",
        out_path, ra_deg.shape[0], dec_deg.shape[0],
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--wave-min", type=float, default=DEFAULT_WAVE_MIN)
    p.add_argument("--wave-max", type=float, default=DEFAULT_WAVE_MAX)
    p.add_argument("--rebin", type=int, default=DEFAULT_REBIN)
    p.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR)
    p.add_argument("--max-nan-frac", type=float, default=DEFAULT_MAX_NAN_FRAC)
    p.add_argument("--max-spectra", type=int, default=5000)
    p.add_argument("--group-col", default=None,
                   help="labels column to use for group-stratified split")
    p.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    p.add_argument("--min-covered-spectra", type=int, default=DEFAULT_MIN_COVERED_SPECTRA)
    p.add_argument("--skip-coverage-probe", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not args.skip_coverage_probe:
        coverage_probe(
            h5_path=args.h5,
            wave_min=args.wave_min,
            wave_max=args.wave_max,
            coverage_threshold=args.coverage_threshold,
            min_spectra=args.min_covered_spectra,
        )

    payload = build_features(
        h5_path=args.h5,
        labels_parquet=args.labels,
        out_path=args.out,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        rebin=args.rebin,
        min_snr=args.min_snr,
        max_nan_frac=args.max_nan_frac,
        max_spectra=args.max_spectra,
        group_col=args.group_col,
        seed=args.seed,
    )
    X = payload["X"]
    wc = payload["wave_centers"]
    med = payload["median_imputer"]
    _assert_na_d_resolved(wc)
    _log_gap_region_medians(wc, med)
    gap_mask = compute_gap_mask(wc)
    _augment_npz_with_gap_mask(args.out, payload, gap_mask)

    # append ra_deg, dec_deg to the npz so can derive
    # spatial groups for a second leakage-aware CV.
    ra_deg, dec_deg = derive_aligned_radec(
        h5_path=args.h5,
        labels_parquet=args.labels,
        payload=payload,
        min_snr=args.min_snr,
        max_nan_frac=args.max_nan_frac,
        max_spectra=args.max_spectra,
        seed=args.seed,
    )
    if ra_deg.shape[0] != X.shape[0] or dec_deg.shape[0] != X.shape[0]:
        raise RuntimeError(
            f"ra/dec alignment broken: ra={ra_deg.shape[0]}, "
            f"dec={dec_deg.shape[0]}, X rows={X.shape[0]}"
        )
    _augment_npz_with_radec(args.out, ra_deg, dec_deg)
    print(f"wrote {args.out}  shape={X.shape}  "
          f"train/val/test={len(payload['train_idx'])}/"
          f"{len(payload['val_idx'])}/{len(payload['test_idx'])} "
          f"gap_mask=bool[{gap_mask.shape[0]}], sum={int(gap_mask.sum())}  "
          f"ra/dec=float64[{ra_deg.shape[0]}]")


if __name__ == "__main__":
    main()
