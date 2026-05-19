"""Unit tests for src/interpret/eso_catalog.py (no real network calls)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from src.interpret import eso_catalog
from src.interpret.eso_catalog import (
    GES_DR5_TABLE,
    _TAP_ALIASES,
    build_ges_dr5_adql,
    fetch_ges_dr5_params,
)


_SOURCE_COLUMNS: tuple[str, ...] = tuple(_TAP_ALIASES.keys())

_EM_DASH = chr(0x2014)
_EN_DASH = chr(0x2013)


def _fixture_table(n_rows: int = 3) -> Table:
    """Build an astropy Table with all 15 source column names and n_rows rows."""
    rng = np.random.default_rng(seed=0)
    data = {
        "OBJECT": np.array([f"cn{i:03d}" for i in range(n_rows)], dtype=object),
        "RA": rng.uniform(0.0, 360.0, size=n_rows),
        "DECLINATION": rng.uniform(-90.0, 90.0, size=n_rows),
        "TEFF": rng.uniform(4000.0, 8000.0, size=n_rows),
        "E_TEFF": rng.uniform(50.0, 200.0, size=n_rows),
        "LOGG": rng.uniform(2.0, 5.0, size=n_rows),
        "E_LOGG": rng.uniform(0.05, 0.2, size=n_rows),
        "FEH": rng.uniform(-1.0, 0.5, size=n_rows),
        "E_FEH": rng.uniform(0.05, 0.2, size=n_rows),
        "SNR": rng.uniform(20.0, 200.0, size=n_rows),
        "VRAD": rng.uniform(-50.0, 50.0, size=n_rows),
        "E_VRAD": rng.uniform(0.1, 2.0, size=n_rows),
        "REC_SETUP": np.array(["U580"] * n_rows, dtype=object),
        "SETUP": np.array(["U580"] * n_rows, dtype=object),
        "GES_TYPE": np.array(["GE_SD"] * n_rows, dtype=object),
    }
    return Table(data)


class _FakeJob:
    def __init__(self, table: Table) -> None:
        self._table = table

    def get_results(self) -> Table:
        return self._table


class _FakeTapPlus:
    """Stand-in for astroquery.utils.tap.core.TapPlus used by monkeypatching."""

    last_url: str | None = None
    last_query: str | None = None

    def __init__(self, url: str | None = None, **_: object) -> None:
        type(self).last_url = url
        self._table: Table = _fixture_table(n_rows=3)

    def launch_job_async(self, query: str, **_: object) -> _FakeJob:
        type(self).last_query = query
        return _FakeJob(self._table)


def _install_fake_tap(monkeypatch: pytest.MonkeyPatch, n_rows: int = 3) -> None:
    """Install a FakeTapPlus on the astroquery module that fetch imports lazily."""
    fake_table = _fixture_table(n_rows=n_rows)

    class _Bound(_FakeTapPlus):
        def __init__(self, url: str | None = None, **_: object) -> None:
            type(self).last_url = url
            self._table = fake_table

    import astroquery.utils.tap.core as tap_core
    monkeypatch.setattr(tap_core, "TapPlus", _Bound)


class TestBuildAdql:
    def test_build_ges_dr5_adql_contains_required_filters(self) -> None:
        q = build_ges_dr5_adql()
        assert GES_DR5_TABLE in q
        assert "TEFF IS NOT NULL" in q
        assert "LOGG IS NOT NULL" in q
        assert "FEH IS NOT NULL" in q
        assert "REC_SETUP LIKE '%580%'" in q
        for src_col in _SOURCE_COLUMNS:
            assert src_col in q, f"ADQL missing source column {src_col!r}"

    def test_build_ges_dr5_adql_respects_rec_setup_pattern(self) -> None:
        q = build_ges_dr5_adql(rec_setup_like="U580")
        assert "REC_SETUP LIKE 'U580'" in q


class TestFetchGesDr5Params:
    def test_fetch_ges_dr5_params_renames_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tap(monkeypatch, n_rows=3)
        df = fetch_ges_dr5_params()
        assert tuple(df.columns) == tuple(_TAP_ALIASES.values())
        assert len(df) == 3
        assert df["ra_deg"].dtype == np.float64
        assert df["cname"].dtype == object

    def test_fetch_ges_dr5_params_logs_rowcount(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _install_fake_tap(monkeypatch, n_rows=3)
        with caplog.at_level(logging.INFO, logger=eso_catalog.logger.name):
            fetch_ges_dr5_params()
        assert any("rows=3" in rec.getMessage() for rec in caplog.records), (
            f"no INFO log matching rows=3; saw: {[r.getMessage() for r in caplog.records]}"
        )

    def test_fetch_ges_dr5_params_returns_15_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tap(monkeypatch, n_rows=5)
        df = fetch_ges_dr5_params()
        assert df.shape == (5, 15)
        assert list(df.columns) == list(_TAP_ALIASES.values())

    def test_fetch_ges_dr5_params_numeric_dtypes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tap(monkeypatch, n_rows=3)
        df = fetch_ges_dr5_params()
        for col in (
            "ra_deg", "dec_deg", "teff_k", "e_teff_k", "logg", "e_logg",
            "feh", "e_feh", "snr", "rv", "e_rv",
        ):
            assert df[col].dtype == np.float64, f"{col} dtype is {df[col].dtype}"
        for col in ("cname", "rec_setup", "setup", "ges_type"):
            assert df[col].dtype == object, f"{col} dtype is {df[col].dtype}"


class TestFetchGesDr5MissingColumn:
    def test_missing_required_column_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad_table = _fixture_table(n_rows=2)
        bad_table.remove_column("FEH")

        class _BrokenTap:
            def __init__(self, url: str | None = None, **_: object) -> None:
                pass

            def launch_job_async(self, query: str, **_: object) -> _FakeJob:
                return _FakeJob(bad_table)

        import astroquery.utils.tap.core as tap_core
        monkeypatch.setattr(tap_core, "TapPlus", _BrokenTap)

        with pytest.raises(KeyError):
            fetch_ges_dr5_params()


class TestFixtureSanity:
    def test_fixture_has_all_source_columns(self) -> None:
        t = _fixture_table(n_rows=2)
        assert set(t.colnames) == set(_SOURCE_COLUMNS)

    def test_fixture_to_pandas_roundtrip(self) -> None:
        t = _fixture_table(n_rows=4)
        df = t.to_pandas()
        assert len(df) == 4
        assert set(df.columns) == set(_SOURCE_COLUMNS)


def test_no_em_dashes_in_source_module() -> None:
    """Hard rule from."""
    import pathlib
    text = pathlib.Path(eso_catalog.__file__).read_text(encoding="utf-8")
    assert _EM_DASH not in text
    assert _EN_DASH not in text
