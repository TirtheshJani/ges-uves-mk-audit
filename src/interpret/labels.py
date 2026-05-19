# SPDX-License-Identifier: MIT
"""Build MK-class labels for Gaia-ESO UVES spectra.

Labels come from the public Gaia-ESO DR5.1 recommended-parameters catalog
served by the ESO Science Archive at the public TAP endpoint
``https://archive.eso.org/tap_cat`` as table ``safcat."GES_DR5_1_V1"``
(Hourihane et al. 2023, A&A 676, A129; bibcode 2023A&A...676A.129H).
The catalog is fetched on first use via ``src.interpret.eso_catalog``
and cached locally as parquet (cache filename and on-disk schema unchanged
relative to the prior VizieR-backed implementation; see ).

MK Teff bin edges follow Pecaut & Mamajek (2013, Table 5), rounded to the
nearest 100 K:

  A: [7300, 10000)
  F: [6000, 7300)
  G: [5300, 6000)
  K: [3900, 5300)

O/B/M are excluded at label-construction time per the interpretability plan:
GES UVES is FGK-targeted, and the classifier is trained only on A/F/G/K.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import pandas as pd

from src.interpret.lines import ALLOWED_MK_CLASSES

logger = logging.getLogger(__name__)

GES_DR5_BIBCODE: Final[str] = "2023A&A...676A.129H"

MK_BIN_EDGES: Final[dict[str, tuple[float, float]]] = {
    "A": (7300.0, 10000.0),
    "F": (6000.0, 7300.0),
    "G": (5300.0, 6000.0),
    "K": (3900.0, 5300.0),
}

MK_INT: Final[dict[str, int]] = {c: i for i, c in enumerate(ALLOWED_MK_CLASSES)}

_FILENAME_RE = re.compile(
    r"ges_uves_(?P<ra>-?\d+\.\d+)_(?P<dec>-?\d+\.\d+)\.fits",
    re.IGNORECASE,
)


def parse_ra_dec_from_filename(source_file: str | Path) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) parsed from the GES UVES filename convention.

    The spectrum fetcher (src/fetch/fetch_ges.py) writes files as
    ``ges_uves_{RA:.6f}_{Dec:.6f}.fits``, so the pointing RA/Dec is
    recoverable from the filename. Raises ValueError if the filename
    does not match.
    """
    name = Path(source_file).name
    m = _FILENAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"filename {name!r} does not match GES UVES convention")
    return float(m.group("ra")), float(m.group("dec"))


def bin_teff_to_mk(teff: np.ndarray) -> np.ndarray:
    """Bin Teff (K) to MK class strings. Values outside FGK range -> 'OTHER'."""
    teff = np.asarray(teff, dtype=float)
    out = np.full(teff.shape, "OTHER", dtype=object)
    for cls, (lo, hi) in MK_BIN_EDGES.items():
        mask = (teff >= lo) & (teff < hi)
        out[mask] = cls
    return out


def compute_boundary_distance_k(
    teff: np.ndarray, mk_class: np.ndarray
) -> np.ndarray:
    """Distance in K from each Teff to the nearest edge of its own MK bin.

    Returns NaN for rows labelled 'OTHER' (no native bin).
    """
    teff = np.asarray(teff, dtype=float)
    mk_class = np.asarray(mk_class, dtype=object)
    dist = np.full(teff.shape, np.nan, dtype=float)
    for cls, (lo, hi) in MK_BIN_EDGES.items():
        mask = mk_class == cls
        if not np.any(mask):
            continue
        dist[mask] = np.minimum(np.abs(teff[mask] - lo), np.abs(teff[mask] - hi))
    return dist


def fetch_ges_params_catalog(
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch (and cache) the GES DR5.1 recommended-parameters catalog.

    On a cache hit (``cache_dir/ges_dr5_params.parquet`` exists and
    ``refresh=False``), the parquet is read and returned. Otherwise the
    catalog is fetched from the ESO TAP endpoint via
    ``src.interpret.eso_catalog.fetch_ges_dr5_params``, written to the cache
    parquet, and returned.

    The returned DataFrame has 15 columns:
    ``cname, ra_deg, dec_deg, teff_k, e_teff_k, logg, e_logg, feh, e_feh,
    snr, rv, e_rv, rec_setup, setup, ges_type``.

    Parameters
    ----------
    cache_dir: Path
        Directory holding the parquet cache file. Created if absent.
    refresh: bool, optional
        If True, ignore the cache and re-fetch from ESO TAP. Default False.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ges_dr5_params.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        logger.info(
            "loaded cached GES params (%d rows) from %s", len(df), cache_path,
        )
        return df

    from src.interpret import eso_catalog  # lazy import for test hygiene

    logger.info(
        "cache miss: fetching GES params from ESO TAP "
        "(refresh=%s, bibcode=%s)",
        refresh, GES_DR5_BIBCODE,
    )
    df = eso_catalog.fetch_ges_dr5_params()
    df.to_parquet(cache_path, index=False)
    logger.info(
        "cached GES params (%d rows) -> %s [bibcode %s]",
        len(df), cache_path, GES_DR5_BIBCODE,
    )
    return df


@dataclass
class GhostAudit:
    """Counts of catalog rows whose CName has no matched spectrum in the HDF5.

    admits mixed-setup catalog rows (REC_SETUP LIKE '%580%') in
    addition to pure ``U580``. The audit reports how many catalog rows have
    no positional match within the cross-match radius, broken out by pure
    vs mixed setup, so the reversal trigger
    (``mixed_setup_ghost_rate > 0.05``) can be evaluated.

    Attributes
    ----------
    n_catalog_rows: int
        Number of rows in the input catalog after the catalog-side fetch.
    n_ghost: int
        Catalog rows with no spectrum match within ``match_radius_arcsec``.
    n_ghost_pure_u580: int
        Subset of ghosts whose ``rec_setup`` equals exactly ``"U580"``.
    n_ghost_mixed_setup: int
        Subset of ghosts whose ``rec_setup`` contains ``"|"`` or ``":"``
        (i.e. multiple setups).
    n_mixed_setup_total: int
        Total catalog rows whose ``rec_setup`` is mixed (denominator for
        the reversal trigger).
    mixed_setup_ghost_rate: float
        ``n_ghost_mixed_setup / max(n_mixed_setup_total, 1)``.
    sample_ghost_cnames: list[str]
        Up to 20 ghost CNames for human inspection.
    """

    n_catalog_rows: int
    n_ghost: int
    n_ghost_pure_u580: int
    n_ghost_mixed_setup: int
    n_mixed_setup_total: int
    mixed_setup_ghost_rate: float
    sample_ghost_cnames: list[str]


def audit_ghost_cnames(
    catalog_df: pd.DataFrame,
    matched_cnames: set[str],
) -> GhostAudit:
    """Count catalog rows whose CName has no matched spectrum.

    Implements the reversal trigger: mixed-setup catalog rows
    (``rec_setup`` containing ``|`` or ``:``) admitted by the
    ``REC_SETUP LIKE '%580%'`` filter that have no corresponding U580
    spectrum in the HDF5. The HDF5 schema does not carry CName, so
    ``matched_cnames`` is the set of CNames assigned via positional
    cross-match in:func:`build_labels`.

    Parameters
    ----------
    catalog_df: pd.DataFrame
        Catalog frame with at least ``cname`` and ``rec_setup`` columns.
    matched_cnames: set[str]
        CNames assigned to a spectrum via positional cross-match.

    Returns
    -------
    GhostAudit
        Counts and a sample of ghost CNames (truncated to 20).
    """
    cnames = catalog_df["cname"].astype(object).to_numpy()
    rec_setups = catalog_df["rec_setup"].astype(object).to_numpy()

    is_mixed = np.array(
        [("|" in str(rs)) or (":" in str(rs)) for rs in rec_setups],
        dtype=bool,
    )
    is_pure_u580 = np.array(
        [str(rs).strip() == "U580" for rs in rec_setups],
        dtype=bool,
    )

    is_ghost = np.array(
        [str(c) not in matched_cnames for c in cnames],
        dtype=bool,
    )

    n_mixed_total = int(is_mixed.sum())
    n_ghost_mixed = int((is_ghost & is_mixed).sum())
    rate = n_ghost_mixed / max(n_mixed_total, 1)

    ghost_cnames = [str(c) for c in cnames[is_ghost]]
    sample = ghost_cnames[:20]

    return GhostAudit(
        n_catalog_rows=int(len(catalog_df)),
        n_ghost=int(is_ghost.sum()),
        n_ghost_pure_u580=int((is_ghost & is_pure_u580).sum()),
        n_ghost_mixed_setup=n_ghost_mixed,
        n_mixed_setup_total=n_mixed_total,
        mixed_setup_ghost_rate=float(rate),
        sample_ghost_cnames=sample,
    )


@dataclass
class LabelStats:
    n_input: int
    n_unmatched: int
    n_other: int
    n_final: int
    per_class: dict[str, int]
    n_dropped_e_teff: int = 0
    per_class_dropped_e_teff: dict[str, int] | None = None
    ghost_audit: GhostAudit | None = None


def build_labels(
    h5_path: Path,
    cache_dir: Path,
    match_radius_arcsec: float = 0.5,
    ambiguity_radius_arcsec: float = 2.0,
    min_per_class: int = 50,
    warn_per_class: int = 200,
    allow_drop_underfilled: bool = False,
    e_teff_max_k: float = 200.0,
) -> tuple[pd.DataFrame, LabelStats]:
    """Build MK-class labels for the GES UVES spectra in ``h5_path``.

    Parameters
    ----------
    h5_path: regridded HDF5 from build_hdf5.py.
    cache_dir: directory for the ESO TAP catalog parquet cache
        (``cache_dir/ges_dr5_params.parquet``).
    match_radius_arcsec: position cross-match tolerance (default 0.5 arcsec
        per Gaia DR3 astrometric accuracy; see physicist review).
    ambiguity_radius_arcsec: second-nearest-neighbour distance below which a
        match is treated as ambiguous and dropped.
    min_per_class: hard RuntimeError floor per surviving MK class.
    warn_per_class: emit logger.warning below this count.
    allow_drop_underfilled: if True and a class falls below min_per_class,
        drop the class rather than raising (use for A in FGK-heavy samples).
    e_teff_max_k: reject catalog rows whose recommended Teff uncertainty
        ``e_teff_k`` exceeds this value. 200 K is well below
        the narrowest MK Teff bin spacing (700 K, G/K boundary). NaN
        ``e_teff_k`` values fail the cut and are dropped. The cut is
        applied after positional cross-match and before MK binning so
        per-class drop counts reflect the binning decision.

    Returns
    -------
    (labels_df, stats)
        labels_df columns: source_file, ra_deg, dec_deg, teff_k, logg, feh,
        mk_class, mk_int, boundary_distance_k, dwarf_flag.

    Notes
    -----
    ghost-CNAME audit: catalog rows admitted by
    ``REC_SETUP LIKE '%580%'`` that have no matching spectrum in the HDF5
    are counted (split by pure-U580 vs mixed-setup) and reported in
    ``stats.ghost_audit``. If
    ``stats.ghost_audit.mixed_setup_ghost_rate > 0.05`` the reversal trigger is met and the catalog should be re-fetched with
    ``REC_SETUP = 'U580'`` exact equality.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    h5_path = Path(h5_path)
    cache_dir = Path(cache_dir)

    with h5py.File(h5_path, "r") as h5:
        source_files = np.array(h5["metadata/source_file"][:], dtype=object)
        surveys = np.array(h5["metadata/survey"][:], dtype=object)
    source_files = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in source_files],
        dtype=object,
    )
    surveys = np.array(
        [s.decode() if isinstance(s, bytes) else s for s in surveys],
        dtype=object,
    )

    ges_mask = surveys == "ges"
    ges_files = source_files[ges_mask]
    if len(ges_files) == 0:
        raise RuntimeError(f"no GES spectra found in {h5_path}")

    ras = np.empty(len(ges_files), dtype=float)
    decs = np.empty(len(ges_files), dtype=float)
    for i, fname in enumerate(ges_files):
        ras[i], decs[i] = parse_ra_dec_from_filename(fname)

    cat = fetch_ges_params_catalog(cache_dir)
    spec_coord = SkyCoord(ras * u.deg, decs * u.deg)
    cat_coord = SkyCoord(
        cat["ra_deg"].to_numpy() * u.deg,
        cat["dec_deg"].to_numpy() * u.deg,
    )

    idx, d2d, _ = spec_coord.match_to_catalog_sky(cat_coord)
    sep_arcsec = d2d.to(u.arcsec).value
    matched = sep_arcsec <= match_radius_arcsec

    _, d2d2, _ = spec_coord.match_to_catalog_sky(cat_coord, nthneighbor=2)
    amb = d2d2.to(u.arcsec).value < ambiguity_radius_arcsec
    matched = matched & ~amb

    n_unmatched = int((~matched).sum())
    if n_unmatched:
        logger.info(
            "dropped %d/%d GES spectra with no unambiguous match within %.2f arcsec",
            n_unmatched, len(ges_files), match_radius_arcsec,
        )

    teff_raw = cat["teff_k"].to_numpy()[idx]
    e_teff_raw = cat["e_teff_k"].to_numpy()[idx]
    logg_raw = cat["logg"].to_numpy()[idx]
    feh_raw = cat["feh"].to_numpy()[idx]
    cname_raw = cat["cname"].astype(object).to_numpy()[idx]

    df_full = pd.DataFrame({
        "source_file": ges_files,
        "ra_deg": ras,
        "dec_deg": decs,
        "teff_k": teff_raw,
        "e_teff_k": e_teff_raw,
        "logg": logg_raw,
        "feh": feh_raw,
        "cname": cname_raw,
    })
    df = df_full.loc[matched].reset_index(drop=True)

    matched_cnames: set[str] = set(df["cname"].astype(object).tolist())
    ghost_audit = audit_ghost_cnames(cat, matched_cnames)
    logger.info(
        "ghost-CNAME audit: catalog_rows=%d ghosts=%d "
        "(pure_u580=%d, mixed_setup=%d/%d, rate=%.4f)",
        ghost_audit.n_catalog_rows, ghost_audit.n_ghost,
        ghost_audit.n_ghost_pure_u580, ghost_audit.n_ghost_mixed_setup,
        ghost_audit.n_mixed_setup_total, ghost_audit.mixed_setup_ghost_rate,
    )
    if ghost_audit.mixed_setup_ghost_rate > 0.05:
        logger.warning(
            "reversal triggered: mixed_setup_ghost_rate=%.4f > 0.05; "
            "consider re-fetching with REC_SETUP = 'U580' exact equality",
            ghost_audit.mixed_setup_ghost_rate,
        )

    e_teff_arr = df["e_teff_k"].to_numpy()
    e_teff_ok = np.isfinite(e_teff_arr) & (e_teff_arr <= e_teff_max_k)
    n_dropped_e_teff = int((~e_teff_ok).sum())

    teff_pre_cut = df["teff_k"].to_numpy()
    mk_pre_cut = bin_teff_to_mk(teff_pre_cut)
    per_class_dropped_e_teff: dict[str, int] = {}
    for cls in ALLOWED_MK_CLASSES:
        per_class_dropped_e_teff[cls] = int(
            ((~e_teff_ok) & (mk_pre_cut == cls)).sum()
        )

    if n_dropped_e_teff:
        logger.info(
            "E_TEFF cut: dropped %d/%d rows with e_teff_k > %.1f K (per-class: %s)",
            n_dropped_e_teff, len(df), e_teff_max_k, per_class_dropped_e_teff,
        )
    df = df.loc[e_teff_ok].reset_index(drop=True)

    mk_class = bin_teff_to_mk(df["teff_k"].to_numpy())
    n_other = int((mk_class == "OTHER").sum())
    keep = mk_class != "OTHER"
    df = df.loc[keep].reset_index(drop=True)
    df["mk_class"] = mk_class[keep]
    df["boundary_distance_k"] = compute_boundary_distance_k(
        df["teff_k"].to_numpy(), df["mk_class"].to_numpy()
    )
    df["dwarf_flag"] = df["logg"] > 3.5
    df["mk_int"] = df["mk_class"].map(MK_INT).astype("int8")

    counts = df["mk_class"].value_counts().to_dict()
    to_drop: list[str] = []
    for cls in ALLOWED_MK_CLASSES:
        n = int(counts.get(cls, 0))
        if n < min_per_class:
            if allow_drop_underfilled:
                logger.warning(
                    "class %s under-filled (%d < %d) - dropping",
                    cls, n, min_per_class,
                )
                to_drop.append(cls)
            else:
                raise RuntimeError(
                    f"class {cls!r} has only {n} spectra (< {min_per_class}); "
                    "pass allow_drop_underfilled=True to proceed with fewer classes."
                )
        elif n < warn_per_class:
            logger.warning(
                "class %s has only %d spectra (< warn threshold %d)",
                cls, n, warn_per_class,
            )

    if to_drop:
        df = df[~df["mk_class"].isin(to_drop)].reset_index(drop=True)

    per_class = df["mk_class"].value_counts().to_dict()
    stats = LabelStats(
        n_input=int(len(ges_files)),
        n_unmatched=n_unmatched,
        n_other=n_other,
        n_final=int(len(df)),
        per_class={k: int(v) for k, v in per_class.items()},
        n_dropped_e_teff=n_dropped_e_teff,
        per_class_dropped_e_teff=per_class_dropped_e_teff,
        ghost_audit=ghost_audit,
    )
    return (
        df[[
            "source_file", "ra_deg", "dec_deg", "teff_k", "logg", "feh",
            "mk_class", "mk_int", "boundary_distance_k", "dwarf_flag",
        ]],
        stats,
    )
