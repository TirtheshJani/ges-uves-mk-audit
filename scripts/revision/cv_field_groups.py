#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Cross-validation grouped by Gaia-ESO survey field (GES_FLD).

The deposited run reports a plain 5-fold StratifiedKFold macro-F1 on the
2,576 train+val rows and a StratifiedGroupKFold "spatial" pass whose groups
come from DBSCAN on sky coordinates.  Because 95.8 percent of those DBSCAN
groups are singletons, the spatial pass bounds cluster leakage only loosely.
The Gaia-ESO DR5.1 catalogue carries a survey field identifier, GES_FLD (one
open cluster or calibration field per value), which is the natural leakage
group.  This script

  1. fetches OBJECT, GES_FLD, GES_TYPE, RA, DECLINATION, TEFF, LOGG, FEH for
     every REC_SETUP LIKE '%580%' row of safcat."GES_DR5_1_V1" from the ESO
     TAP service (asynchronous job; same row filter as
     src.interpret.eso_catalog.build_ges_dr5_adql),
  2. matches every feature row of artifacts/features.npz to the catalogue by
     sky position (nearest neighbour within 0.5 arcsec, the radius used by
     src.interpret.labels.build_labels),
  3. summarises the GES_FLD groups on train+val and on all rows and relates
     them to the DBSCAN groups of the deposited spatial pass,
  4. re-runs the deposited StratifiedKFold control through
     src.interpret.classifier.train_cv and checks it against
     artifacts/metrics.json, and
  5. runs train_cv with the GES_FLD groups (StratifiedGroupKFold, n_splits=5,
     random_state=42, production hyperparameters, early stopping on the
     held-out fold exactly as the deposited CV does).

Nothing in src/ and none of the deposited artifacts are modified.  Outputs:
  artifacts/revision/ges_fld_catalog.csv
  artifacts/revision/ges_fld_groups.csv
  artifacts/revision/cv_field_groups.json
  artifacts/revision/cv_field_groups.log

Run from the repo root, one stage per call (each stage takes well under
three minutes on a 2-core machine):
  python scripts/revision/cv_field_groups.py fetch
  python scripts/revision/cv_field_groups.py match
  python scripts/revision/cv_field_groups.py cv --which control
  python scripts/revision/cv_field_groups.py cv --which field
  python scripts/revision/cv_field_groups.py assemble
``cv --which X --fold K`` runs one fold (same splitter, same train and
evaluate calls as train_cv) and ``cv --which X --combine`` assembles the
stored folds, for machines where a full 5-fold pass does not fit in one call.
Every stage reads and rewrites cv_field_groups.json, so stages can be re-run
individually.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.interpret.classifier import (  # noqa: E402
    DEFAULT_EARLY_STOPPING,
    DEFAULT_HPARAMS,
    derive_spatial_groups,
    evaluate,
    singleton_group_stats,
    train,
    train_cv,
)
from src.interpret.eso_catalog import ESO_TAP_URL, GES_DR5_TABLE  # noqa: E402
from src.interpret.lines import ALLOWED_MK_CLASSES  # noqa: E402

FEATURES = REPO_ROOT / "artifacts" / "features.npz"
METRICS = REPO_ROOT / "artifacts" / "metrics.json"
OUT_DIR = REPO_ROOT / "artifacts" / "revision"
CATALOG_CSV = OUT_DIR / "ges_fld_catalog.csv"
GROUPS_CSV = OUT_DIR / "ges_fld_groups.csv"
RESULT_JSON = OUT_DIR / "cv_field_groups.json"
LOG_FILE = OUT_DIR / "cv_field_groups.log"

ADQL = (
    "SELECT OBJECT, GES_FLD, GES_TYPE, RA, DECLINATION, TEFF, LOGG, FEH "
    f"FROM {GES_DR5_TABLE} "
    "WHERE REC_SETUP LIKE '%580%' "
    "AND TEFF IS NOT NULL AND LOGG IS NOT NULL AND FEH IS NOT NULL"
)
COLUMN_ALIASES = {
    "OBJECT": "cname",
    "GES_FLD": "ges_fld",
    "GES_TYPE": "ges_type",
    "RA": "ra_deg",
    "DECLINATION": "dec_deg",
    "TEFF": "teff_k",
    "LOGG": "logg",
    "FEH": "feh",
}
STRING_COLS = ("cname", "ges_fld", "ges_type")
PARAM_COLS = ("teff_k", "logg", "feh")
MATCH_RADIUS_ARCSEC = 0.5
AMBIGUITY_RADIUS_ARCSEC = 2.0
SEED = 42
N_SPLITS = 5
PER_FOLD_TIMEOUT_S = 3600.0
DBSCAN_EPS_DEG = 0.1
DBSCAN_MIN_SAMPLES = 5
TOP_N_GROUPS = 5
KEY_FOR = {"control": "cv_control", "field": "cv_field_groups"}

logger = logging.getLogger("cv_field_groups")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr),
                    logging.FileHandler(LOG_FILE, mode="a")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def to_plain(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    if RESULT_JSON.exists():
        with RESULT_JSON.open() as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    state["last_updated_utc"] = utc_now()
    with RESULT_JSON.open("w") as f:
        json.dump(to_plain(state), f, indent=2)
    logger.info("wrote %s", RESULT_JSON)


def software_versions() -> dict:
    import astropy
    import astroquery
    import lightgbm
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "astropy": astropy.__version__,
        "astroquery": astroquery.__version__,
        "platform": platform.platform(),
        "n_cpu": os.cpu_count(),
    }


def inputs_block(num_class: int, class_labels: list[str]) -> dict:
    hp = dict(DEFAULT_HPARAMS)
    hp["num_class"] = int(num_class)
    return {
        "features": rel(FEATURES),
        "rows_used_for_cv": (
            "train_idx followed by val_idx (tv_idx), exactly as "
            "scripts/train_classifier.py builds X_tv/y_tv"
        ),
        "tap_url": ESO_TAP_URL,
        "table": GES_DR5_TABLE,
        "adql": ADQL,
        "tap_mode": "asynchronous (TapPlus.launch_job_async)",
        "match_radius_arcsec": MATCH_RADIUS_ARCSEC,
        "ambiguity_radius_arcsec_reported": AMBIGUITY_RADIUS_ARCSEC,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "cv_function": "src.interpret.classifier.train_cv (unmodified)",
        "splitter_field_groups": (
            "sklearn.model_selection.StratifiedGroupKFold(n_splits=5, "
            "shuffle=True, random_state=42) with groups = integer-encoded GES_FLD"
        ),
        "splitter_control": (
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), reached "
            "through the train_cv fallback on the zero-filled groups array of "
            "features.npz, as in the deposited run"
        ),
        "hyperparameters": hp,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING,
        "early_stopping_eval_set": (
            "the held-out fold itself (train_cv convention, unchanged)"
        ),
        "per_fold_timeout_s": PER_FOLD_TIMEOUT_S,
        "class_labels": list(class_labels),
        "group_encoding": (
            "pandas.factorize over GES_FLD of all 3032 feature rows; one integer "
            "per distinct field; rows without a GES_FLD (if any) get unique codes"
        ),
        "dbscan_reference": {
            "function": "src.interpret.classifier.derive_spatial_groups on all rows",
            "eps_deg": DBSCAN_EPS_DEG,
            "min_samples": DBSCAN_MIN_SAMPLES,
        },
        "software": software_versions(),
    }


def load_features():
    payload = np.load(FEATURES, allow_pickle=False)
    X = payload["X"]
    y = payload["y"].astype(np.int64)
    train_idx = payload["train_idx"]
    val_idx = payload["val_idx"]
    test_idx = payload["test_idx"]
    tv_idx = np.concatenate([train_idx, val_idx])
    present = sorted(np.unique(y).tolist())
    class_labels = [ALLOWED_MK_CLASSES[i] for i in present]
    return payload, X, y, train_idx, val_idx, test_idx, tv_idx, len(present), class_labels


def encode_groups(fld: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer-encode GES_FLD labels; empty labels become unique codes."""
    fld = np.asarray(fld, dtype=object)
    codes, _ = pd.factorize(fld)
    codes = codes.astype(np.int64)
    empty = np.array([(not isinstance(v, str)) or v == "" for v in fld])
    n_empty = int(empty.sum())
    if n_empty:
        codes[empty] = codes.max() + 1 + np.arange(n_empty, dtype=np.int64)
    return codes, n_empty


# --------------------------------------------------------------------------
# stage 1: fetch
# --------------------------------------------------------------------------
def stage_fetch(state: dict, sync: bool = False) -> None:
    from astroquery.utils.tap.core import TapPlus

    mode = "sync" if sync else "async"
    logger.info("ESO TAP request: url=%s mode=%s adql=%s", ESO_TAP_URL, mode, ADQL)
    t0 = time.perf_counter()
    tap = TapPlus(url=ESO_TAP_URL)
    job = tap.launch_job(ADQL) if sync else tap.launch_job_async(ADQL)
    table = job.get_results()
    runtime = float(time.perf_counter() - t0)
    logger.info("ESO TAP returned rows=%d in %.1fs", len(table), runtime)

    upper = {c.upper(): c for c in table.colnames}
    missing = [c for c in COLUMN_ALIASES if c not in upper]
    if missing:
        raise KeyError(f"TAP response missing columns {missing}; got {table.colnames}")
    df = table.to_pandas().rename(columns={upper[k]: v for k, v in COLUMN_ALIASES.items()})
    df = df.loc[:, list(COLUMN_ALIASES.values())]
    for col in STRING_COLS:
        df[col] = df[col].map(
            lambda v: v.decode() if isinstance(v, bytes) else v
        ).map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].where(df[col].notna(), None).astype(object)
    for col in ("ra_deg", "dec_deg"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in PARAM_COLS:
        # catalogue values are float32 with at most 2-3 decimals; rounding at
        # 4 dp removes float32 noise without losing information
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").round(4)
    df.to_csv(CATALOG_CSV, index=False)
    logger.info("wrote %s (rows=%d)", CATALOG_CSV, len(df))

    fld = df["ges_fld"]
    fld_ok = fld.map(lambda v: isinstance(v, str) and v != "")
    pos_nfld = df.loc[fld_ok].groupby(["ra_deg", "dec_deg"])["ges_fld"].nunique()
    counts = fld[fld_ok].value_counts()
    state["catalogue"] = {
        "file": rel(CATALOG_CSV),
        "fetched_utc": utc_now(),
        "tap_mode": mode,
        "fetch_runtime_s": round(runtime, 2),
        "n_rows": int(len(df)),
        "n_distinct_ges_fld": int(counts.size),
        "n_rows_null_or_empty_ges_fld": int((~fld_ok).sum()),
        "n_distinct_cname": int(df["cname"].nunique()),
        "n_rows_with_repeated_cname": int(df["cname"].duplicated(keep=False).sum()),
        "n_distinct_positions": int(len(df.drop_duplicates(["ra_deg", "dec_deg"]))),
        "n_positions_shared_by_rows_with_different_ges_fld": int((pos_nfld > 1).sum()),
        "n_rows_null_ra_or_dec": int(df[["ra_deg", "dec_deg"]].isna().any(axis=1).sum()),
        "largest_fields": [
            {"ges_fld": str(k), "n_rows": int(v)} for k, v in counts.head(TOP_N_GROUPS).items()
        ],
        "ges_type_counts": {
            str(k): int(v) for k, v in df["ges_type"].fillna("").value_counts().items()
        },
    }
    logger.info("catalogue summary: %s", json.dumps(to_plain(state["catalogue"])))


# --------------------------------------------------------------------------
# stage 2: match + group statistics + DBSCAN relation
# --------------------------------------------------------------------------
def group_stats(fld: np.ndarray) -> dict:
    fld = np.asarray(fld, dtype=object)
    codes, n_empty = encode_groups(fld)
    s = pd.Series(fld[[isinstance(v, str) and v != "" for v in fld]])
    counts = s.value_counts()
    sizes = counts.to_numpy()
    base = singleton_group_stats(codes)
    out = {
        "n_rows": int(len(fld)),
        "n_rows_without_ges_fld": int(n_empty),
        "n_groups": int(len(np.unique(codes))),
        "n_singleton_groups": int((sizes == 1).sum()) + int(n_empty),
        "n_rows_in_singleton_groups": int((sizes == 1).sum()) + int(n_empty),
        "n_groups_size_ge2": int((sizes >= 2).sum()),
        "n_rows_in_groups_size_ge2": int(sizes[sizes >= 2].sum()),
        "largest_group_size": int(sizes.max()) if len(sizes) else 0,
        "group_size_median": float(np.median(sizes)) if len(sizes) else float("nan"),
        "largest_groups": [
            {"ges_fld": str(k), "n_rows": int(v)} for k, v in counts.head(TOP_N_GROUPS).items()
        ],
        "singleton_group_stats": base,
    }
    out["singleton_group_fraction"] = out["n_singleton_groups"] / out["n_groups"]
    out["singleton_row_fraction"] = out["n_rows_in_singleton_groups"] / out["n_rows"]
    out["note"] = (
        "singleton_group_stats is src.interpret.classifier.singleton_group_stats "
        "applied to the integer-encoded GES_FLD groups; its n_dbscan_noise_rows "
        "key counts negative ids and is 0 by construction here"
    )
    return out


def dbscan_relation(fld: np.ndarray, sp: np.ndarray) -> dict:
    codes, _ = encode_groups(fld)
    df = pd.DataFrame({"fld": codes, "sp": np.asarray(sp, dtype=np.int64)})
    fld_sizes = df.groupby("fld")["sp"].transform("size")
    per_fld_n_sp = df.groupby("fld")["sp"].nunique()
    clustered = df[df["sp"] >= 0]
    per_fld_n_clusters = clustered.groupby("fld")["sp"].nunique()
    per_cluster_n_fld = clustered.groupby("sp")["fld"].nunique()
    noise = df["sp"] < 0
    nonsingleton_fld = fld_sizes >= 2
    all_noise_per_fld = df.groupby("fld")["sp"].max() < 0
    fld_size_by_group = df.groupby("fld")["sp"].size()
    return {
        "n_rows": int(len(df)),
        "n_ges_fld_groups": int(per_fld_n_sp.size),
        "n_dbscan_groups": int(df["sp"].nunique()),
        "n_dbscan_clusters": int(per_cluster_n_fld.size),
        "n_dbscan_noise_rows": int(noise.sum()),
        "n_ges_fld_groups_spanning_gt1_dbscan_group": int((per_fld_n_sp > 1).sum()),
        "n_ges_fld_groups_spanning_gt1_dbscan_cluster": int((per_fld_n_clusters > 1).sum()),
        "n_dbscan_clusters_spanning_gt1_ges_fld": int((per_cluster_n_fld > 1).sum()),
        "n_dbscan_clusters_within_one_ges_fld": int((per_cluster_n_fld == 1).sum()),
        "n_rows_in_ges_fld_groups_size_ge2": int(nonsingleton_fld.sum()),
        "n_rows_dbscan_noise_with_ges_fld_mates": int((noise & nonsingleton_fld).sum()),
        "n_ges_fld_groups_entirely_dbscan_noise": int(all_noise_per_fld.sum()),
        "n_ges_fld_groups_size_ge2_entirely_dbscan_noise": int(
            (all_noise_per_fld & (fld_size_by_group >= 2)).sum()
        ),
        "note": (
            "DBSCAN groups are derive_spatial_groups(eps_deg=0.1, min_samples=5) "
            "on all 3032 rows, as in scripts/train_classifier.py; noise rows carry "
            "unique negative ids and are counted as distinct groups. "
            "n_rows_dbscan_noise_with_ges_fld_mates counts rows the deposited "
            "spatial pass treated as singletons although another row of the same "
            "partition shares their GES_FLD."
        ),
    }


def stage_match(state: dict) -> None:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    payload, X, y, train_idx, val_idx, test_idx, tv_idx, num_class, class_labels = load_features()
    ra = payload["ra_deg"].astype(np.float64)
    dec = payload["dec_deg"].astype(np.float64)
    n = len(ra)
    split = np.array([""] * n, dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    if (split == "").any():
        raise RuntimeError("feature rows not covered by train/val/test indices")

    cat = pd.read_csv(CATALOG_CSV, keep_default_na=False, dtype={c: str for c in STRING_COLS})
    for col in ("ra_deg", "dec_deg", *PARAM_COLS):
        cat[col] = pd.to_numeric(cat[col], errors="coerce").astype("float64")
    cat_ra = cat["ra_deg"].to_numpy()
    cat_dec = cat["dec_deg"].to_numpy()
    cat_fld = cat["ges_fld"].to_numpy(dtype=object)

    spec = SkyCoord(ra * u.deg, dec * u.deg)
    catc = SkyCoord(cat_ra * u.deg, cat_dec * u.deg)
    idx, d2d, _ = spec.match_to_catalog_sky(catc)
    sep = d2d.to(u.arcsec).value
    idx2, d2d2, _ = spec.match_to_catalog_sky(catc, nthneighbor=2)
    sep2 = d2d2.to(u.arcsec).value

    exact = (cat_ra[idx] == ra) & (cat_dec[idx] == dec)
    exact6 = (np.round(cat_ra[idx], 6) == np.round(ra, 6)) & (
        np.round(cat_dec[idx], 6) == np.round(dec, 6)
    )
    within = sep <= MATCH_RADIUS_ARCSEC
    unmatched = ~within
    second_within_amb = sep2 < AMBIGUITY_RADIUS_ARCSEC
    second_within_match = sep2 <= MATCH_RADIUS_ARCSEC
    conflict_near = second_within_match & (cat_fld[idx] != cat_fld[idx2])
    pos_nfld = cat.groupby(["ra_deg", "dec_deg"])["ges_fld"].nunique()
    conflict_same_pos = np.array([
        pos_nfld.get((cat_ra[i], cat_dec[i]), 1) > 1 for i in idx
    ])
    used, used_counts = np.unique(idx[within], return_counts=True)

    cname = np.where(within, cat["cname"].to_numpy(dtype=object)[idx], "")
    fld = np.where(within, cat_fld[idx], "")
    gtype = np.where(within, cat["ges_type"].to_numpy(dtype=object)[idx], "")
    groups_df = pd.DataFrame({
        "row_index": np.arange(n, dtype=np.int64),
        "ra_deg": ra,
        "dec_deg": dec,
        "cname": cname,
        "ges_fld": fld,
        "ges_type": gtype,
        "split": split,
        "sep_arcsec": sep,
    })
    groups_df.to_csv(GROUPS_CSV, index=False)
    logger.info("wrote %s (rows=%d)", GROUPS_CSV, n)

    matched_fld_ok = within & np.array([isinstance(v, str) and v != "" for v in fld])
    state["matching"] = {
        "file": rel(GROUPS_CSV),
        "n_feature_rows": int(n),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_exact_float_equal": int(exact.sum()),
        "n_equal_after_rounding_catalogue_to_6dp": int(exact6.sum()),
        "n_within_match_radius": int(within.sum()),
        "n_unmatched": int(unmatched.sum()),
        "unmatched_row_indices": [int(i) for i in np.flatnonzero(unmatched)],
        "n_matched_without_ges_fld": int((within & ~matched_fld_ok).sum()),
        "sep_arcsec_max_matched": float(sep[within].max()) if within.any() else float("nan"),
        "sep_arcsec_median_matched": float(np.median(sep[within])) if within.any() else float("nan"),
        "n_second_neighbour_within_ambiguity_radius": int(second_within_amb.sum()),
        "n_second_neighbour_within_match_radius": int(second_within_match.sum()),
        "n_second_neighbour_within_match_radius_different_ges_fld": int(conflict_near.sum()),
        "n_matched_positions_shared_by_rows_with_different_ges_fld": int(
            (conflict_same_pos & within).sum()
        ),
        "n_catalogue_rows_matched_by_multiple_feature_rows": int((used_counts > 1).sum()),
        "ges_type_counts_matched_full": {
            str(k): int(v) for k, v in pd.Series(gtype[within]).value_counts().items()
        },
        "ges_type_counts_matched_trainval": {
            str(k): int(v)
            for k, v in pd.Series(gtype[tv_idx][within[tv_idx]]).value_counts().items()
        },
    }
    logger.info("matching summary: %s", json.dumps(to_plain(state["matching"])))

    state["group_statistics"] = {
        "trainval": group_stats(fld[tv_idx]),
        "full": group_stats(fld),
    }
    logger.info("group statistics: %s", json.dumps(to_plain(state["group_statistics"])))

    t0 = time.perf_counter()
    sp = derive_spatial_groups(
        ra_deg=ra, dec_deg=dec, eps_deg=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES,
    )
    dt = float(time.perf_counter() - t0)
    state["dbscan_relation"] = {
        "derive_spatial_groups_runtime_s": round(dt, 3),
        "trainval": dbscan_relation(fld[tv_idx], sp[tv_idx]),
        "full": dbscan_relation(fld, sp),
    }
    logger.info("dbscan relation: %s", json.dumps(to_plain(state["dbscan_relation"])))
    state.setdefault("inputs", inputs_block(num_class, class_labels))


# --------------------------------------------------------------------------
# stage 3: cross-validation
# --------------------------------------------------------------------------
def make_split_iter(X, y, groups):
    """Replicate train_cv's splitter choice exactly."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n_unique = int(len(np.unique(groups)))
    if n_unique < N_SPLITS:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        return "StratifiedKFold", splitter.split(X, y)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return "StratifiedGroupKFold", splitter.split(X, y, groups=groups)


def fold_composition(X, y, groups, class_labels) -> tuple[str, list[dict]]:
    name, it = make_split_iter(X, y, groups)
    present = sorted(np.unique(y).tolist())
    out = []
    for k, (tr, va) in enumerate(it):
        tr_g = set(np.unique(groups[tr]).tolist())
        va_g = set(np.unique(groups[va]).tolist())
        out.append({
            "fold": int(k),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_groups_train": int(len(tr_g)),
            "n_groups_val": int(len(va_g)),
            "groups_disjoint": bool(not (tr_g & va_g)),
            "val_class_counts": {
                class_labels[i]: int((y[va] == c).sum()) for i, c in enumerate(present)
            },
        })
    return name, out


def cv_groups(which: str, payload, tv_idx):
    if which == "control":
        groups = payload["groups"][tv_idx]
        extra = {"groups_source": "features.npz groups array (zero-filled in the deposited run)"}
    else:
        g = pd.read_csv(GROUPS_CSV, keep_default_na=False, dtype={"ges_fld": str})
        if not np.array_equal(g["row_index"].to_numpy(), np.arange(len(g))):
            raise RuntimeError("ges_fld_groups.csv row_index is not 0..n-1")
        codes, n_empty = encode_groups(g["ges_fld"].to_numpy(dtype=object))
        groups = codes[tv_idx]
        extra = {
            "groups_source": rel(GROUPS_CSV),
            "n_rows_without_ges_fld_trainval": int(
                sum(1 for v in g["ges_fld"].to_numpy(dtype=object)[tv_idx] if v == "")
            ),
        }
    return np.asarray(groups), extra


def summarise(res: dict, groups, X, y, class_labels, wall_s: float, how: str) -> dict:
    per_fold = list(res["per_fold"])
    f1s = [float(f["macro_f1"]) for f in per_fold]
    accs = [float(f["accuracy"]) for f in per_fold]
    recall_mean = {}
    if per_fold:
        for lab in class_labels:
            recall_mean[lab] = float(np.mean([f["per_class_recall"][lab] for f in per_fold]))
    splitter_name, comp = fold_composition(X, y, groups, class_labels)
    return {
        "how": how,
        "splitter": res["splitter"],
        "splitter_check": splitter_name,
        "n_splits": int(res["n_splits"]),
        "n_folds_completed": len(per_fold),
        "fallback_triggered": bool(res["cv_fallback_triggered"]),
        "fallback_reason": str(res["cv_fallback_reason"]),
        "n_rows": int(len(y)),
        "n_unique_groups": int(len(np.unique(groups))),
        "macro_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_f1_sd_population_ddof0": float(np.std(f1s, ddof=0)) if f1s else float("nan"),
        "macro_f1_sd_sample_ddof1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else float("nan"),
        "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
        "accuracy_sd_population_ddof0": float(np.std(accs, ddof=0)) if accs else float("nan"),
        "accuracy_sd_sample_ddof1": float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan"),
        "train_cv_cv_macro_f1_mean": float(res["cv_macro_f1_mean"]),
        "train_cv_cv_macro_f1_std": float(res["cv_macro_f1_std"]),
        "per_fold_macro_f1": f1s,
        "per_fold_accuracy": accs,
        "per_fold_best_iteration": [int(f["best_iteration"]) for f in per_fold],
        "per_class_recall_mean": recall_mean,
        "per_fold": per_fold,
        "fold_composition": comp,
        "total_fit_runtime_s": float(res["total_runtime_s"]),
        "wall_runtime_s": float(wall_s),
        "finished_utc": utc_now(),
    }


def reproduction_block(record: dict) -> dict:
    with METRICS.open() as f:
        dep = json.load(f)
    dep_f1 = [float(f["macro_f1"]) for f in dep["cv_per_fold"]]
    dep_acc = [float(f["accuracy"]) for f in dep["cv_per_fold"]]
    dep_it = [int(f["best_iteration"]) for f in dep["cv_per_fold"]]
    new_f1 = record["per_fold_macro_f1"]
    same_len = len(dep_f1) == len(new_f1)
    diffs = [abs(a - b) for a, b in zip(dep_f1, new_f1)] if same_len else []
    return {
        "deposited_file": rel(METRICS),
        "deposited_cv_macro_f1_mean": float(dep["cv_macro_f1_mean"]),
        "deposited_cv_macro_f1_std": float(dep["cv_macro_f1_std"]),
        "deposited_cv_splitter": str(dep["cv_splitter"]),
        "deposited_cv_fallback_reason": str(dep["cv_fallback_reason"]),
        "deposited_per_fold_macro_f1": dep_f1,
        "deposited_per_fold_accuracy": dep_acc,
        "deposited_per_fold_best_iteration": dep_it,
        "rerun_cv_macro_f1_mean": float(record["macro_f1_mean"]),
        "rerun_cv_macro_f1_std": float(record["macro_f1_sd_population_ddof0"]),
        "abs_difference_mean": abs(float(dep["cv_macro_f1_mean"]) - float(record["macro_f1_mean"])),
        "max_abs_difference_per_fold_macro_f1": max(diffs) if diffs else float("nan"),
        "per_fold_macro_f1_identical": bool(same_len and all(a == b for a, b in zip(dep_f1, new_f1))),
        "per_fold_accuracy_identical": bool(
            same_len and all(a == b for a, b in zip(dep_acc, record["per_fold_accuracy"]))
        ),
        "per_fold_best_iteration_identical": bool(
            same_len and dep_it == record["per_fold_best_iteration"]
        ),
        "splitter_identical": str(dep["cv_splitter"]) == str(record["splitter"]),
        "reproduced_exactly": bool(
            same_len
            and float(dep["cv_macro_f1_mean"]) == float(record["macro_f1_mean"])
            and all(a == b for a, b in zip(dep_f1, new_f1))
        ),
        "reproduced_within_1e-6": bool(
            same_len and abs(float(dep["cv_macro_f1_mean"]) - float(record["macro_f1_mean"])) < 1e-6
        ),
    }


def run_single_fold(X, y, groups, num_class, class_labels, fold: int) -> tuple[str, dict]:
    """One fold with the same splitter and the same train/evaluate calls as train_cv."""
    name, it = make_split_iter(X, y, groups)
    for k, (tr, va) in enumerate(it):
        if k != fold:
            continue
        if name == "StratifiedGroupKFold":
            assert not (set(np.unique(groups[tr]).tolist()) & set(np.unique(groups[va]).tolist()))
        logger.info("single fold %d/%d: n_train=%d n_val=%d", k, N_SPLITS, len(tr), len(va))
        t0 = time.perf_counter()
        model = train(
            X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va],
            num_class=num_class, hparams=None, early_stopping_rounds=DEFAULT_EARLY_STOPPING,
        )
        runtime_s = float(time.perf_counter() - t0)
        m = evaluate(model, X[va], y[va], class_labels=class_labels,
                     n_train=int(len(tr)), n_val=int(len(va)))
        rec = {
            "fold": int(k),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "macro_f1": float(m.macro_f1),
            "accuracy": float(m.accuracy),
            "per_class_recall": dict(m.per_class_recall),
            "confusion_matrix": list(m.confusion_matrix),
            "runtime_s": runtime_s,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        }
        logger.info("single fold %d done: macro_f1=%.4f acc=%.4f runtime=%.1fs",
                    k, m.macro_f1, m.accuracy, runtime_s)
        return name, rec
    raise ValueError(f"fold {fold} out of range 0..{N_SPLITS - 1}")


def stage_cv(state: dict, which: str, fold: int | None, combine: bool) -> None:
    payload, X, y, train_idx, val_idx, test_idx, tv_idx, num_class, class_labels = load_features()
    X_tv, y_tv = X[tv_idx], y[tv_idx]
    groups, extra = cv_groups(which, payload, tv_idx)
    key = KEY_FOR[which]
    state.setdefault("inputs", inputs_block(num_class, class_labels))
    logger.info("cv stage: which=%s key=%s n=%d n_unique_groups=%d fold=%s combine=%s",
                which, key, len(y_tv), len(np.unique(groups)), fold, combine)

    if fold is not None:
        name, rec = run_single_fold(X_tv, y_tv, groups, num_class, class_labels, fold)
        block = state.setdefault(key, {})
        block.setdefault("per_fold_partial", {})[str(fold)] = rec
        block["partial_splitter"] = name
        block.update(extra)
        return

    if combine:
        block = state.get(key, {})
        partial = block.get("per_fold_partial", {})
        missing = [k for k in range(N_SPLITS) if str(k) not in partial]
        if missing:
            raise RuntimeError(f"folds {missing} missing for {which}; run them first")
        per_fold = [partial[str(k)] for k in range(N_SPLITS)]
        f1s = [f["macro_f1"] for f in per_fold]
        res = {
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s, ddof=0)),
            "per_fold": per_fold,
            "cv_fallback_triggered": False,
            "cv_fallback_reason": "",
            "splitter": block.get("partial_splitter", ""),
            "n_splits": N_SPLITS,
            "total_runtime_s": float(sum(f["runtime_s"] for f in per_fold)),
        }
        record = summarise(res, groups, X_tv, y_tv, class_labels,
                           wall_s=res["total_runtime_s"],
                           how="single-fold calls (--fold) combined")
        record["per_fold_partial"] = partial
    else:
        t0 = time.perf_counter()
        res = train_cv(
            X=X_tv,
            y=y_tv,
            groups=groups,
            num_class=num_class,
            n_splits=N_SPLITS,
            random_state=SEED,
            per_fold_timeout_s=PER_FOLD_TIMEOUT_S,
            class_labels=class_labels,
        )
        wall = float(time.perf_counter() - t0)
        record = summarise(res, groups, X_tv, y_tv, class_labels, wall_s=wall,
                           how="src.interpret.classifier.train_cv, all folds in one call")

    record.update(extra)
    if which == "control":
        record["reproduction"] = reproduction_block(record)
        logger.info("control reproduction: %s", json.dumps(to_plain(record["reproduction"])))
    state[key] = record
    logger.info("%s: mean=%.6f sd0=%.6f sd1=%.6f folds=%s", key, record["macro_f1_mean"],
                record["macro_f1_sd_population_ddof0"], record["macro_f1_sd_sample_ddof1"],
                record["per_fold_macro_f1"])


# --------------------------------------------------------------------------
# stage 4: assemble
# --------------------------------------------------------------------------
def stage_assemble(state: dict) -> dict:
    required = ("inputs", "catalogue", "matching", "group_statistics",
                "dbscan_relation", "cv_control", "cv_field_groups")
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(f"sections missing from {RESULT_JSON}: {missing}")
    for k in ("cv_control", "cv_field_groups"):
        if "macro_f1_mean" not in state[k]:
            raise RuntimeError(f"{k} holds only partial folds; run `cv --which ... --combine`")
    with METRICS.open() as f:
        dep = json.load(f)
    ctrl = state["cv_control"]
    fld = state["cv_field_groups"]
    comparison = {
        "deposited_plain_cv_macro_f1_mean": float(dep["cv_macro_f1_mean"]),
        "deposited_plain_cv_macro_f1_std": float(dep["cv_macro_f1_std"]),
        "deposited_spatial_cv_macro_f1_mean": float(dep["cv_spatial_macro_f1_mean"]),
        "deposited_spatial_cv_macro_f1_std": float(dep["cv_spatial_macro_f1_std"]),
        "deposited_spatial_singleton_group_fraction": float(
            dep["cv_spatial_singleton_stats"]["singleton_group_fraction"]
        ),
        "rerun_control_macro_f1_mean": float(ctrl["macro_f1_mean"]),
        "control_reproduced_exactly": bool(ctrl["reproduction"]["reproduced_exactly"]),
        "field_grouped_macro_f1_mean": float(fld["macro_f1_mean"]),
        "field_grouped_macro_f1_sd_population_ddof0": float(fld["macro_f1_sd_population_ddof0"]),
        "field_grouped_macro_f1_sd_sample_ddof1": float(fld["macro_f1_sd_sample_ddof1"]),
        "field_grouped_minus_plain": float(fld["macro_f1_mean"]) - float(dep["cv_macro_f1_mean"]),
        "field_grouped_minus_spatial": (
            float(fld["macro_f1_mean"]) - float(dep["cv_spatial_macro_f1_mean"])
        ),
        "field_singleton_group_fraction_trainval": float(
            state["group_statistics"]["trainval"]["singleton_group_fraction"]
        ),
    }
    final = {
        "description": (
            "Leakage-aware cross-validation of the production LightGBM MK classifier "
            "with groups defined by the Gaia-ESO survey field identifier GES_FLD "
            "(one open cluster or calibration field per value), run on the 2576 "
            "train+val rows with the unmodified src.interpret.classifier.train_cv "
            "(StratifiedGroupKFold, n_splits=5, random_state=42, production "
            "hyperparameters, early stopping on the held-out fold). GES_FLD is "
            "assigned to every feature row by a 0.5 arcsec sky match against "
            "safcat.\"GES_DR5_1_V1\" from the ESO TAP service. The deposited plain "
            "StratifiedKFold control is re-run through the same function and checked "
            "against artifacts/metrics.json. Produced by "
            "scripts/revision/cv_field_groups.py; no deposited file was modified."
        ),
        "script": "scripts/revision/cv_field_groups.py",
        "generated_utc": utc_now(),
        "inputs": state["inputs"],
        "catalogue": state["catalogue"],
        "matching": state["matching"],
        "group_statistics": state["group_statistics"],
        "dbscan_relation": state["dbscan_relation"],
        "cv_control": ctrl,
        "cv_field_groups": fld,
        "comparison": comparison,
    }
    logger.info("comparison: %s", json.dumps(to_plain(comparison)))
    return final


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["fetch", "match", "cv", "assemble", "all"])
    p.add_argument("--which", choices=["control", "field"],
                   help="cv stage: which grouping to run")
    p.add_argument("--fold", type=int, default=None,
                   help="cv stage: run only this fold (0-based) and store it")
    p.add_argument("--combine", action="store_true",
                   help="cv stage: assemble folds stored with --fold")
    p.add_argument("--sync", action="store_true",
                   help="fetch stage: use a synchronous TAP job (ESO caps these at 2000 rows)")
    args = p.parse_args(argv)

    setup_logging()
    logger.info("stage=%s args=%s", args.stage, vars(args))
    t0 = time.perf_counter()
    state = load_state()

    if args.stage in ("fetch", "all"):
        stage_fetch(state, sync=args.sync)
        save_state(state)
    if args.stage in ("match", "all"):
        stage_match(state)
        save_state(state)
    if args.stage == "cv":
        if args.which is None:
            p.error("cv requires --which control|field")
        stage_cv(state, args.which, args.fold, args.combine)
        save_state(state)
    if args.stage == "all":
        for which in ("control", "field"):
            stage_cv(state, which, None, False)
            save_state(state)
    if args.stage in ("assemble", "all"):
        final = stage_assemble(state)
        save_state(final)
    logger.info("stage %s done in %.1fs", args.stage, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
