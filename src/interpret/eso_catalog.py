# SPDX-License-Identifier: MIT
"""ESO TAP client for the Gaia-ESO DR5.1 recommended-parameters catalog.

The catalog is served by the ESO Science Archive at the public TAP endpoint
``https://archive.eso.org/tap_cat`` as table ``safcat."GES_DR5_1_V1"``. This
module is the network boundary for that fetch; ``src.interpret.labels``
imports it but contains no TAP-specific code itself.

Wavelength convention is irrelevant here (this is a parameter catalog, not a
spectrum), but the parameters returned (Teff, logg, [Fe/H], RV, SNR) are the
official GES DR5.1 recommended values per Hourihane et al. 2023, A&A 676, A129
(bibcode 2023A&A...676A.129H).
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

ESO_TAP_URL: Final[str] = "https://archive.eso.org/tap_cat"
GES_DR5_TABLE: Final[str] = 'safcat."GES_DR5_1_V1"'
DEFAULT_REC_SETUP_LIKE: Final[str] = "%580%"

_TAP_ALIASES: Final[dict[str, str]] = {
    "OBJECT": "cname",
    "RA": "ra_deg",
    "DECLINATION": "dec_deg",
    "TEFF": "teff_k",
    "E_TEFF": "e_teff_k",
    "LOGG": "logg",
    "E_LOGG": "e_logg",
    "FEH": "feh",
    "E_FEH": "e_feh",
    "SNR": "snr",
    "VRAD": "rv",
    "E_VRAD": "e_rv",
    "REC_SETUP": "rec_setup",
    "SETUP": "setup",
    "GES_TYPE": "ges_type",
}

_NUMERIC_TARGETS: Final[tuple[str, ...]] = (
    "ra_deg",
    "dec_deg",
    "teff_k",
    "e_teff_k",
    "logg",
    "e_logg",
    "feh",
    "e_feh",
    "snr",
    "rv",
    "e_rv",
)

_OBJECT_TARGETS: Final[tuple[str, ...]] = (
    "cname",
    "rec_setup",
    "setup",
    "ges_type",
)


def build_ges_dr5_adql(rec_setup_like: str = DEFAULT_REC_SETUP_LIKE) -> str:
    """Return the ADQL query for the GES DR5.1 parameters fetch.

    The query selects the 15 source columns enumerated in ``_TAP_ALIASES``
    in key order, requires Teff, logg, and [Fe/H] to be non-null, and filters
    on REC_SETUP via SQL ``LIKE`` so mixed-setup spectra including U580 are
    captured (e.g. REC_SETUP values such as ``HR10|HR21:U580`` match
    ``%580%``).

    Parameters
    ----------
    rec_setup_like: str
        SQL LIKE pattern applied to the REC_SETUP column. Default ``%580%``
        captures pure U580 plus mixed setups containing U580.
    """
    select_cols = ", ".join(_TAP_ALIASES.keys())
    query = (
        f"SELECT {select_cols} "
        f"FROM {GES_DR5_TABLE} "
        f"WHERE TEFF IS NOT NULL "
        f"AND LOGG IS NOT NULL "
        f"AND FEH IS NOT NULL "
        f"AND REC_SETUP LIKE '{rec_setup_like}'"
    )
    return query


def _rename_columns_case_insensitive(
    table_columns: list[str],
) -> dict[str, str]:
    """Map source column names to canonical lowercase names, case-insensitive."""
    cols_upper = {c.upper(): c for c in table_columns}
    mapping: dict[str, str] = {}
    for src_upper, canonical in _TAP_ALIASES.items():
        if src_upper not in cols_upper:
            raise KeyError(
                f"ESO TAP response missing expected column {src_upper!r}; "
                f"got columns: {sorted(table_columns)}"
            )
        mapping[cols_upper[src_upper]] = canonical
    return mapping


def fetch_ges_dr5_params(
    rec_setup_like: str = DEFAULT_REC_SETUP_LIKE,
    tap_url: str = ESO_TAP_URL,
    timeout_s: float = 600.0,
) -> pd.DataFrame:
    """Fetch the GES DR5.1 parameters catalog from the ESO TAP endpoint.

    Returns a DataFrame with exactly the 15 columns enumerated in
    ``_TAP_ALIASES.values()``, in that order. Numeric columns are cast to
    ``float64``; string columns to ``object``.

    Parameters
    ----------
    rec_setup_like: str
        SQL LIKE pattern for the REC_SETUP filter.
    tap_url: str
        ESO TAP endpoint URL.
    timeout_s: float
        Maximum time for the asynchronous TAP job. Currently informational;
        ``astroquery.utils.tap.core.TapPlus.launch_job_async`` does not expose
        a per-call timeout, but the parameter is retained for forward
        compatibility and retries if needed.

    Notes
    -----
    Async mode (``launch_job_async``) is used because ESO TAP applies a 2000-row
    cap on synchronous queries; the GES DR5.1 catalog has roughly 6,000 UVES
    rows after the REC_SETUP filter, so sync mode silently truncates. Async
    submits the query as a job, polls for completion, and returns the full
    result set.
    """
    from astroquery.utils.tap.core import TapPlus  # lazy import for test hygiene

    query = build_ges_dr5_adql(rec_setup_like=rec_setup_like)
    logger.info(
        "ESO TAP request: url=%s table=%s rec_setup_like=%s timeout_s=%.1f",
        tap_url, GES_DR5_TABLE, rec_setup_like, timeout_s,
    )

    tap = TapPlus(url=tap_url)
    job = tap.launch_job_async(query)
    table = job.get_results()

    rename = _rename_columns_case_insensitive(list(table.colnames))
    df = table.to_pandas().rename(columns=rename)

    for col in _NUMERIC_TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in _OBJECT_TARGETS:
        df[col] = df[col].astype(object)

    ordered = list(_TAP_ALIASES.values())
    df = df.loc[:, ordered]

    logger.info("ESO TAP returned rows=%d", len(df))
    return df
