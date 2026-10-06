# SPDX-License-Identifier: MIT
"""Pickles (1998) template-matching benchmark for the MK classifier.

Per test spectrum we resample each Pickles UVKLIB template onto the feature
grid and pick the template with the minimum chi-squared to the (continuum-
normalized) spectrum. The template's MK type is read from the STScI FITS
header ("spectral type" card) when present, otherwise from the embedded
index -> type lookup; its coarse class is the "Pickles-MK" label that we
compare against the LightGBM prediction on the same spectrum.

This is a reality check against an external library, not a training signal.
Pickles spans O-M; we collapse everything outside A/F/G/K to "OTHER".

Catalog: Pickles 1998, PASP 110, 863 (VizieR J/PASP/110/863).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

# Index -> MK type for the 131 STScI/CDBS "pickles_uk_<N>.fits" files.
# Transcribed from the STScI FITS primary headers: the COMMENT1 card of each
# file reads "spectral type: <MK>" (COMMENT2 gives the metallicity class and
# COMMENT3 the original UVKLIB filename, e.g. "ukwf5v.dat"). The STScI
# sequence interleaves the metal-weak ("w") and metal-rich ("r") UVKLIB
# variants with the solar-metallicity templates, so the same MK type can
# appear at consecutive indices; those indices are
# 17 (weak F5V), 19 (rich F6V), 21 (weak F8V), 22 (rich F8V), 24 (weak G0V), 25 (rich G0V), 28 (weak G5V), 29 (rich G5V), 32 (rich K0V), 74 (weak G5III), 75 (rich G5III), 77 (weak G8III), 79 (weak K0III), 80 (rich K0III), 82 (weak K1III), 83 (rich K1III), 85 (weak K2III), 86 (rich K2III), 88 (weak K3III), 89 (rich K3III), 91 (weak K4III), 92 (rich K4III), 94 (rich K5III).
# The library is Pickles 1998, PASP 110, 863 (VizieR J/PASP/110/863).
PICKLES_UVKLIB_MAP: Final[dict[int, str]] = {
    1: 'O5V', 2: 'O9V', 3: 'B0V', 4: 'B1V', 5: 'B3V',
    6: 'B57V', 7: 'B8V', 8: 'B9V', 9: 'A0V', 10: 'A2V',
    11: 'A3V', 12: 'A5V', 13: 'A7V', 14: 'F0V', 15: 'F2V',
    16: 'F5V', 17: 'F5V', 18: 'F6V', 19: 'F6V', 20: 'F8V',
    21: 'F8V', 22: 'F8V', 23: 'G0V', 24: 'G0V', 25: 'G0V',
    26: 'G2V', 27: 'G5V', 28: 'G5V', 29: 'G5V', 30: 'G8V',
    31: 'K0V', 32: 'K0V', 33: 'K2V', 34: 'K3V', 35: 'K4V',
    36: 'K5V', 37: 'K7V', 38: 'M0V', 39: 'M1V', 40: 'M2V',
    41: 'M2.5V', 42: 'M3V', 43: 'M4V', 44: 'M5V', 45: 'M6V',
    46: 'B2IV', 47: 'B6IV', 48: 'A0IV', 49: 'A47IV', 50: 'F02IV',
    51: 'F5IV', 52: 'F8IV', 53: 'G0IV', 54: 'G2IV', 55: 'G5IV',
    56: 'G8IV', 57: 'K0IV', 58: 'K1IV', 59: 'K3IV', 60: 'O8III',
    61: 'B12III', 62: 'B3III', 63: 'B5III', 64: 'B9III', 65: 'A0III',
    66: 'A3III', 67: 'A5III', 68: 'A7III', 69: 'F0III', 70: 'F2III',
    71: 'F5III', 72: 'G0III', 73: 'G5III', 74: 'G5III', 75: 'G5III',
    76: 'G8III', 77: 'G8III', 78: 'K0III', 79: 'K0III', 80: 'K0III',
    81: 'K1III', 82: 'K1III', 83: 'K1III', 84: 'K2III', 85: 'K2III',
    86: 'K2III', 87: 'K3III', 88: 'K3III', 89: 'K3III', 90: 'K4III',
    91: 'K4III', 92: 'K4III', 93: 'K5III', 94: 'K5III', 95: 'M0III',
    96: 'M1III', 97: 'M2III', 98: 'M3III', 99: 'M4III', 100: 'M5III',
    101: 'M6III', 102: 'M7III', 103: 'M8III', 104: 'M9III', 105: 'M10III',
    106: 'B2II', 107: 'B5II', 108: 'F0II', 109: 'F2II', 110: 'G5II',
    111: 'K01II', 112: 'K34II', 113: 'M3II', 114: 'B0I', 115: 'B1I',
    116: 'B3I', 117: 'B5I', 118: 'B8I', 119: 'A0I', 120: 'A2I',
    121: 'F0I', 122: 'F5I', 123: 'F8I', 124: 'G0I', 125: 'G2I',
    126: 'G5I', 127: 'G8I', 128: 'K2I', 129: 'K3I', 130: 'K4I',
    131: 'M2I',
}

_FN_RE = re.compile(r"(?:pickles_)?uk_?(?P<n>\d+)\.(fits|dat|txt)", re.IGNORECASE)


def parse_pickles_filename(filename: str) -> str:
    """Return the MK type string for a Pickles UVKLIB filename via ``PICKLES_UVKLIB_MAP``.

    Raises ``ValueError`` when the name does not follow the ``uk<N>.fits`` /
    ``pickles_uk_<N>.fits`` convention and ``KeyError`` when the index is
    outside 1..131. ``load_pickles_library`` counts both cases and logs them
    once per benchmark run.
    """
    name = Path(filename).name
    m = _FN_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"{name!r} does not match Pickles UVKLIB naming")
    n = int(m.group("n"))
    if n not in PICKLES_UVKLIB_MAP:
        raise KeyError(f"Pickles index {n} not in UVKLIB map (1..131)")
    return PICKLES_UVKLIB_MAP[n]


_HEADER_TYPE_RE = re.compile(r"spectral\s*type\s*:\s*(?P<t>[A-Za-z0-9.]+)", re.IGNORECASE)


def read_pickles_header_type(path: Path) -> str | None:
    """Return the MK type recorded in a Pickles FITS primary header, or ``None``.

    The STScI/CDBS distribution of the UVKLIB library writes the type as a
    ``COMMENT1 = 'spectral type: G2V'`` card in the primary header. Every
    header card whose value is a string is scanned for the pattern
    ``spectral type: <type>`` so the result does not depend on the card
    name. Returns ``None`` when no such card exists (e.g. for templates
    written by other tools) or the file has no readable primary header.
    """
    from astropy.io import fits

    try:
        header = fits.getheader(path, 0)
    except Exception:  # pragma: no cover - unreadable file
        return None
    for key in header:
        try:
            value = header[key]
        except Exception:  # pragma: no cover - malformed card
            continue
        if not isinstance(value, str):
            continue
        m = _HEADER_TYPE_RE.search(value)
        if m:
            return m.group("t")
    return None


def collapse_to_mk(type_str: str) -> str:
    """Map a full MK type (e.g. 'G2V', 'K3III') to the A/F/G/K/OTHER coarse label."""
    if not type_str:
        return "OTHER"
    letter = type_str[0].upper()
    if letter in {"A", "F", "G", "K"}:
        return letter
    return "OTHER"


@dataclass
class Template:
    filename: str
    mk_type: str
    mk_class: str
    flux_on_grid: np.ndarray # continuum-normalized, shape (n_bins)


def load_template_fits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Pickles FITS template and return (wave_aa, flux)."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
        if data.dtype.names is not None:
            names = [n.upper() for n in data.dtype.names]
            wave = np.asarray(data[data.dtype.names[names.index("WAVELENGTH")]], dtype=float)
            flux = np.asarray(data[data.dtype.names[names.index("FLUX")]], dtype=float)
        else:
            # linear WCS fallback
            h = hdul[0].header
            crval1 = float(h["CRVAL1"])
            cdelt1 = float(h.get("CDELT1", h.get("CD1_1", 1.0)))
            naxis1 = int(h["NAXIS1"])
            wave = crval1 + cdelt1 * np.arange(naxis1)
            flux = np.asarray(data, dtype=float)
    return wave, flux


VALID_CONTINUUM_METHODS = ("median_filter_200", "median_filter_50", "polynomial_n5")


def continuum_normalize(
    wave: np.ndarray,
    flux: np.ndarray,
    method: str = "median_filter_200",
    window_aa: float = 200.0,
    polynomial_order: int = 5,
    sigma_clip: float = 3.0,
    sigma_clip_iters: int = 5,
) -> np.ndarray:
    """Continuum-normalise a 1-D spectrum.

    Three methods:

      ``median_filter_200`` (default; current production):
        median filter with a 200 A window. Wide enough that M-class TiO
        bandheads (200 to 300 A spacing per Pickles 1998) get flattened,
        which biases the template match against M-type templates.

      ``median_filter_50``: median filter with a 50 A window. Narrow enough
        to preserve M-class bandhead structure; tighter on FGK features but
        still robust to local noise.

      ``polynomial_n5``: standard iteratively sigma-clipped low-order
        polynomial continuum fit (5th-order Legendre, 3 sigma lower clip,
        up to 5 iterations). The fit converges to the continuum on spectra
        with discrete absorption features by progressively masking the
        deepest residuals.
    """
    if method not in VALID_CONTINUUM_METHODS:
        raise ValueError(
            f"unknown continuum method {method!r}, must be one of {VALID_CONTINUUM_METHODS}"
        )
    if method == "median_filter_200":
        return _continuum_median_filter(wave, flux, window_aa=200.0)
    if method == "median_filter_50":
        return _continuum_median_filter(wave, flux, window_aa=50.0)
    return _continuum_polynomial_clipped(
        wave, flux, order=polynomial_order,
        sigma_clip=sigma_clip, max_iters=sigma_clip_iters,
    )


def _continuum_median_filter(
    wave: np.ndarray, flux: np.ndarray, window_aa: float = 200.0
) -> np.ndarray:
    """Cheap polynomial-free continuum normalization by median-filtering."""
    from scipy.ndimage import median_filter

    dx = float(np.median(np.diff(wave)))
    size = max(3, int(round(window_aa / max(dx, 1e-6))))
    if size % 2 == 0:
        size += 1
    cont = median_filter(flux, size=size)
    cont = np.where(cont > 0, cont, np.nan)
    return (flux / cont).astype(np.float32)


def _continuum_polynomial_clipped(
    wave: np.ndarray,
    flux: np.ndarray,
    order: int = 5,
    sigma_clip: float = 3.0,
    max_iters: int = 5,
) -> np.ndarray:
    """Iteratively sigma-clipped Legendre polynomial continuum fit.

    Standard procedure: fit a low-order polynomial to the spectrum, mask
    points below (continuum - sigma_clip * residual_std), refit, and repeat
    until the mask converges or ``max_iters`` is reached.

    Polynomial order 5 over the 4800 to 6800 A window does not over-fit any
    feature wider than approximately 400 A; the dominant TiO bandheads
    (200-300 A) survive the fit because they get masked by sigma-clipping
    rather than absorbed into the continuum model. The Mg b triplet
    (~21 A), Na D doublet (~6 A), and individual Fe / Ca lines all fall
    well below the polynomial's effective resolution.
    """
    finite = np.isfinite(flux)
    if not finite.any():
        return flux.astype(np.float32)
    # Map wave to [-1, 1] for numerical stability of the Legendre basis.
    w_min = float(wave[finite].min())
    w_max = float(wave[finite].max())
    span = max(w_max - w_min, 1e-9)
    x = 2.0 * (wave - w_min) / span - 1.0
    mask = finite.copy()
    for _ in range(max_iters):
        if mask.sum() <= order + 1:
            break
        coeffs = np.polynomial.legendre.legfit(x[mask], flux[mask], order)
        cont = np.polynomial.legendre.legval(x, coeffs)
        resid = flux - cont
        sigma = float(np.std(resid[mask], ddof=1)) if mask.sum() > 1 else 0.0
        if sigma <= 0:
            break
        new_mask = finite & (resid > -sigma_clip * sigma)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    coeffs = np.polynomial.legendre.legfit(x[mask], flux[mask], order)
    cont = np.polynomial.legendre.legval(x, coeffs)
    cont = np.where(cont > 0, cont, np.nan)
    return (flux / cont).astype(np.float32)


def load_pickles_library(
    pickles_dir: Path,
    wave_centers: np.ndarray,
    continuum_method: str = "median_filter_200",
) -> tuple[list[Template], dict]:
    """Load all UVKLIB FITS templates from ``pickles_dir`` and resample to ``wave_centers``.

    The MK type of each template is taken from the FITS header
    (:func:`read_pickles_header_type`) when the header carries a
    ``spectral type`` card; otherwise it falls back to
    ``PICKLES_UVKLIB_MAP`` keyed on the filename index. Files that carry no
    header type and whose index is outside the map are skipped and counted.

    Returns ``(templates, stats)`` where ``stats`` is a dict carrying
    ``n_files_seen``, ``n_loaded``, ``n_type_from_header``,
    ``n_type_from_map``, ``n_unmapped_skipped`` and ``n_invalid_filename``
    (files matching neither filename pattern). The tuple return is preferred
    over a mutated argument so the caller can log or persist the stats
    explicitly.
    """
    pickles_dir = Path(pickles_dir)
    files = sorted(pickles_dir.glob("uk*.fits")) + sorted(pickles_dir.glob("pickles_uk*.fits"))
    if not files:
        raise RuntimeError(f"no Pickles FITS templates found in {pickles_dir}")
    templates: list[Template] = []
    n_unmapped_skipped = 0
    n_invalid_filename = 0
    n_from_header = 0
    n_from_map = 0
    for path in files:
        mk_type = read_pickles_header_type(path)
        if mk_type is not None:
            n_from_header += 1
        else:
            try:
                mk_type = parse_pickles_filename(path.name)
            except KeyError:
                n_unmapped_skipped += 1
                continue
            except ValueError as exc:
                n_invalid_filename += 1
                logger.warning("skipping %s: %s", path.name, exc)
                continue
            n_from_map += 1
        wave, flux = load_template_fits(path)
        flux_norm = continuum_normalize(wave, flux, method=continuum_method)
        flux_on_grid = np.interp(
            wave_centers, wave, flux_norm, left=np.nan, right=np.nan,
        ).astype(np.float32)
        templates.append(Template(
            filename=path.name,
            mk_type=mk_type,
            mk_class=collapse_to_mk(mk_type),
            flux_on_grid=flux_on_grid,
        ))
    if n_unmapped_skipped > 0:
        logger.info(
            "Pickles loader: skipped %d templates with neither a header type "
            "nor a map entry.",
            n_unmapped_skipped,
        )
    logger.info(
        "loaded %d Pickles templates on %d-bin grid (n_files_seen=%d, "
        "n_type_from_header=%d, n_type_from_map=%d, n_unmapped_skipped=%d, "
        "n_invalid_filename=%d).",
        len(templates), len(wave_centers), len(files), n_from_header,
        n_from_map, n_unmapped_skipped, n_invalid_filename,
    )
    stats = {
        "n_files_seen": len(files),
        "n_loaded": len(templates),
        "n_type_from_header": n_from_header,
        "n_type_from_map": n_from_map,
        "n_unmapped_skipped": n_unmapped_skipped,
        "n_invalid_filename": n_invalid_filename,
    }
    return templates, stats


def best_template_per_spectrum(
    X_test: np.ndarray, templates: list[Template],
) -> np.ndarray:
    """Return an array of length ``len(X_test)`` with the best template index.

    Uses chi-squared assuming unit variance (features are continuum-normalized
    and rebinned, so per-bin variance ~ 1% in the noise regime we care about).
    Templates are restricted to bins where the template is finite for each
    individual spectrum (no global mask).
    """
    T = np.stack([t.flux_on_grid for t in templates], axis=0)
    finite_mask = np.isfinite(T) & np.isfinite(X_test)[:, None, :]
    chi2 = np.full((X_test.shape[0], T.shape[0]), np.inf, dtype=np.float32)
    for i in range(X_test.shape[0]):
        m = finite_mask[i]
        diffs = T - X_test[i][None, :]
        diffs[~m] = 0.0
        counts = m.sum(axis=1)
        sq = (diffs ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2[i] = np.where(counts > 0, sq / counts, np.inf)
    return np.argmin(chi2, axis=1).astype(np.int32)


def benchmark_report(
    y_pred_model: np.ndarray,
    y_pickles_mk: np.ndarray,
    class_labels: list[str],
) -> dict:
    """Confusion + agreement between classifier and Pickles-MK labels.

    Both inputs must be A/F/G/K/OTHER strings of the same length.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    all_labels = list(class_labels) + ["OTHER"]
    cm = confusion_matrix(y_pickles_mk, y_pred_model, labels=all_labels).tolist()
    p, r, f, _ = precision_recall_fscore_support(
        y_pickles_mk, y_pred_model, labels=all_labels, average=None, zero_division=0,
    )
    macro_f1 = float(f1_score(
        y_pickles_mk, y_pred_model, labels=class_labels, average="macro", zero_division=0,
    ))
    agreement = float(np.mean(np.asarray(y_pickles_mk) == np.asarray(y_pred_model)))
    return {
        "labels": all_labels,
        "confusion_matrix": cm,
        "per_class_precision": {all_labels[i]: float(p[i]) for i in range(len(all_labels))},
        "per_class_recall": {all_labels[i]: float(r[i]) for i in range(len(all_labels))},
        "per_class_f1": {all_labels[i]: float(f[i]) for i in range(len(all_labels))},
        "macro_f1_fgk": macro_f1,
        "agreement_rate": agreement,
        "n_compared": int(len(y_pickles_mk)),
    }


def save_report(out_dir: Path, report: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    confusion = report["confusion_matrix"]
    labels = report["labels"]
    csv_lines = ["," + ",".join(labels)]
    for label, row in zip(labels, confusion):
        csv_lines.append(f"{label}," + ",".join(str(v) for v in row))
    (out_dir / "benchmark_confusion.csv").write_text("\n".join(csv_lines) + "\n")
    with (out_dir / "benchmark_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote benchmark confusion + report -> %s", out_dir)
