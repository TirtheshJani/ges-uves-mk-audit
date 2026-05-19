"""Unit tests for src/interpret/labels.py (no network calls)."""
from __future__ import annotations

import inspect
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.interpret import labels as labels_module
from src.interpret.eso_catalog import _TAP_ALIASES
from src.interpret.labels import (
    MK_BIN_EDGES,
    MK_INT,
    audit_ghost_cnames,
    bin_teff_to_mk,
    build_labels,
    compute_boundary_distance_k,
    fetch_ges_params_catalog,
    parse_ra_dec_from_filename,
)
from src.interpret.lines import ALLOWED_MK_CLASSES


_CACHE_COLUMNS: tuple[str, ...] = tuple(_TAP_ALIASES.values())


def _fixture_catalog_df(n_rows: int = 5) -> pd.DataFrame:
    """Build a 15-column GES DR5 fixture DataFrame matching the cache schema."""
    rng = np.random.default_rng(seed=1)
    data = {
        "cname": np.array([f"cn{i:03d}" for i in range(n_rows)], dtype=object),
        "ra_deg": rng.uniform(0.0, 360.0, size=n_rows).astype("float64"),
        "dec_deg": rng.uniform(-90.0, 90.0, size=n_rows).astype("float64"),
        "teff_k": rng.uniform(4000.0, 8000.0, size=n_rows).astype("float64"),
        "e_teff_k": rng.uniform(50.0, 200.0, size=n_rows).astype("float64"),
        "logg": rng.uniform(2.0, 5.0, size=n_rows).astype("float64"),
        "e_logg": rng.uniform(0.05, 0.2, size=n_rows).astype("float64"),
        "feh": rng.uniform(-1.0, 0.5, size=n_rows).astype("float64"),
        "e_feh": rng.uniform(0.05, 0.2, size=n_rows).astype("float64"),
        "snr": rng.uniform(20.0, 200.0, size=n_rows).astype("float64"),
        "rv": rng.uniform(-50.0, 50.0, size=n_rows).astype("float64"),
        "e_rv": rng.uniform(0.1, 2.0, size=n_rows).astype("float64"),
        "rec_setup": np.array(["U580"] * n_rows, dtype=object),
        "setup": np.array(["U580"] * n_rows, dtype=object),
        "ges_type": np.array(["GE_SD"] * n_rows, dtype=object),
    }
    df = pd.DataFrame(data)
    return df.loc[:, list(_CACHE_COLUMNS)]


class TestBinTeffToMk:
    def test_canonical_cases(self):
        teff = np.array([8000.0, 6500.0, 5800.0, 4500.0, 3000.0, 11000.0])
        expected = np.array(["A", "F", "G", "K", "OTHER", "OTHER"], dtype=object)
        np.testing.assert_array_equal(bin_teff_to_mk(teff), expected)

    def test_edges_inclusive_lower_exclusive_upper(self):
        assert bin_teff_to_mk(np.array([7300.0]))[0] == "A"
        assert bin_teff_to_mk(np.array([9999.9]))[0] == "A"
        assert bin_teff_to_mk(np.array([10000.0]))[0] == "OTHER"
        assert bin_teff_to_mk(np.array([6000.0]))[0] == "F"
        assert bin_teff_to_mk(np.array([7299.9]))[0] == "F"
        assert bin_teff_to_mk(np.array([5300.0]))[0] == "G"
        assert bin_teff_to_mk(np.array([5999.9]))[0] == "G"
        assert bin_teff_to_mk(np.array([3900.0]))[0] == "K"
        assert bin_teff_to_mk(np.array([5299.9]))[0] == "K"

    def test_no_obm_labels_ever(self):
        rng = np.random.default_rng(42)
        teff = rng.uniform(2000.0, 40000.0, size=1000)
        labels = bin_teff_to_mk(teff)
        assert not np.isin(labels.astype(str), ["O", "B", "M"]).any()


class TestComputeBoundaryDistance:
    def test_midpoint_of_bin(self):
        teff = np.array([8650.0])  # midway in A
        mk = np.array(["A"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert d[0] == pytest.approx(1350.0)

    def test_near_lower_edge(self):
        teff = np.array([7400.0])  # 100 K above 7300
        mk = np.array(["A"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert d[0] == pytest.approx(100.0)

    def test_other_gives_nan(self):
        teff = np.array([2500.0])
        mk = np.array(["OTHER"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        assert np.isnan(d[0])

    def test_vectorises_correctly(self):
        teff = np.array([7400.0, 6100.0, 5900.0, 4000.0])
        mk = np.array(["A", "F", "G", "K"], dtype=object)
        d = compute_boundary_distance_k(teff, mk)
        np.testing.assert_allclose(d, [100.0, 100.0, 100.0, 100.0])


class TestParseRaDecFromFilename:
    def test_canonical_name(self):
        ra, dec = parse_ra_dec_from_filename("ges_uves_123.456789_-45.678901.fits")
        assert ra == pytest.approx(123.456789)
        assert dec == pytest.approx(-45.678901)

    def test_with_path_prefix(self):
        ra, dec = parse_ra_dec_from_filename(
            "data/ges/uves/ges_uves_10.123456_-20.654321.fits"
        )
        assert ra == pytest.approx(10.123456)
        assert dec == pytest.approx(-20.654321)

    def test_rejects_non_ges_filename(self):
        with pytest.raises(ValueError):
            parse_ra_dec_from_filename("apogee_123.fits")


class TestMkIntContract:
    def test_integer_codes_match_allowed_classes(self):
        assert list(MK_INT.keys()) == list(ALLOWED_MK_CLASSES)
        assert list(MK_INT.values()) == list(range(len(ALLOWED_MK_CLASSES)))


class TestMkBinEdges:
    def test_no_gaps_between_bins(self):
        ordered = sorted(MK_BIN_EDGES.items(), key=lambda kv: kv[1][1])
        prev_hi = None
        for _, (lo, hi) in ordered:
            if prev_hi is not None:
                assert lo == prev_hi, "MK bins must be contiguous"
            prev_hi = hi

    def test_all_allowed_classes_have_edges(self):
        assert set(MK_BIN_EDGES.keys()) == set(ALLOWED_MK_CLASSES)


class TestFetchGesParamsCatalog:
    def test_fetch_ges_params_catalog_cache_hit_skips_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "ges_dr5_params.parquet"
        fixture = _fixture_catalog_df(n_rows=5)
        fixture.to_parquet(cache_path, index=False)

        def _boom(*args: object, **kwargs: object) -> pd.DataFrame:
            raise RuntimeError("network must not be called on cache hit")

        from src.interpret import eso_catalog as eso_catalog_module
        monkeypatch.setattr(eso_catalog_module, "fetch_ges_dr5_params", _boom)

        df = fetch_ges_params_catalog(tmp_path)
        assert len(df) == 5
        assert tuple(df.columns) == _CACHE_COLUMNS

    def test_fetch_ges_params_catalog_cache_miss_writes_parquet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture = _fixture_catalog_df(n_rows=5)

        def _fake_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
            return fixture.copy()

        from src.interpret import eso_catalog as eso_catalog_module
        monkeypatch.setattr(
            eso_catalog_module, "fetch_ges_dr5_params", _fake_fetch
        )

        cache_path = tmp_path / "ges_dr5_params.parquet"
        assert not cache_path.exists()

        df = fetch_ges_params_catalog(tmp_path)
        assert cache_path.exists(), "cache parquet must be written on miss"
        assert len(df) == 5
        assert tuple(df.columns) == _CACHE_COLUMNS

        # Re-read the cache directly to confirm the schema persisted.
        on_disk = pd.read_parquet(cache_path)
        assert tuple(on_disk.columns) == _CACHE_COLUMNS
        assert len(on_disk) == 5

    def test_fetch_ges_params_catalog_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture = _fixture_catalog_df(n_rows=3)

        def _fake_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
            return fixture.copy()

        from src.interpret import eso_catalog as eso_catalog_module
        monkeypatch.setattr(
            eso_catalog_module, "fetch_ges_dr5_params", _fake_fetch
        )

        df = fetch_ges_params_catalog(tmp_path)
        assert tuple(df.columns) == _CACHE_COLUMNS
        assert len(df.columns) == 15

    def test_fetch_ges_params_catalog_refresh_bypasses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-populate cache with one row.
        old = _fixture_catalog_df(n_rows=1)
        old.to_parquet(tmp_path / "ges_dr5_params.parquet", index=False)

        # New fetch returns three rows; refresh=True must use it.
        new = _fixture_catalog_df(n_rows=3)

        def _fake_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
            return new.copy()

        from src.interpret import eso_catalog as eso_catalog_module
        monkeypatch.setattr(
            eso_catalog_module, "fetch_ges_dr5_params", _fake_fetch
        )

        df = fetch_ges_params_catalog(tmp_path, refresh=True)
        assert len(df) == 3


def _make_synthetic_hdf5(
    h5_path: Path,
    ras: np.ndarray,
    decs: np.ndarray,
) -> None:
    """Write a minimal regridded HDF5 with metadata and zero-flux placeholders."""
    n = len(ras)
    source_files = np.array(
        [f"ges_uves_{ra:.6f}_{dec:.6f}.fits".encode() for ra, dec in zip(ras, decs)]
    )
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("metadata/source_file", data=source_files)
        h5.create_dataset(
            "metadata/survey",
            data=np.array(["ges".encode()] * n, dtype="|S3"),
        )
        h5.create_dataset(
            "metadata/snr_median",
            data=np.full(n, 100.0, dtype=np.float32),
        )
        h5.create_dataset(
            "spectra/wavelength",
            data=np.linspace(4800.0, 6800.0, 100, dtype=np.float64),
        )
        h5.create_dataset(
            "spectra/flux",
            data=np.ones((n, 100), dtype=np.float32),
        )


class TestETeffCut:
    """ drop catalog rows with E_TEFF > 200 K (default)."""

    def test_drops_only_above_threshold(self, tmp_path, monkeypatch):
        # Three rows: e_teff = 100 (keep), 250 (drop), NaN (drop).
        ras = np.array([10.0, 20.0, 30.0], dtype=float)
        decs = np.array([-10.0, -20.0, -30.0], dtype=float)
        h5_path = tmp_path / "spec.h5"
        _make_synthetic_hdf5(h5_path, ras, decs)

        cat = pd.DataFrame({
            "cname": ["c0", "c1", "c2"],
            "ra_deg": ras,
            "dec_deg": decs,
            "teff_k": np.array([5500.0, 5500.0, 5500.0]),
            "e_teff_k": np.array([100.0, 250.0, np.nan]),
            "logg": np.array([4.5, 4.5, 4.5]),
            "e_logg": np.array([0.1, 0.1, 0.1]),
            "feh": np.array([0.0, 0.0, 0.0]),
            "e_feh": np.array([0.05, 0.05, 0.05]),
            "snr": np.array([100.0, 100.0, 100.0]),
            "rv": np.array([0.0, 0.0, 0.0]),
            "e_rv": np.array([1.0, 1.0, 1.0]),
            "rec_setup": ["U580", "U580", "U580"],
            "setup": ["U580", "U580", "U580"],
            "ges_type": ["GE_SD"] * 3,
        })

        def _fake_fetch(*args, **kwargs):
            return cat.copy()

        monkeypatch.setattr(
            labels_module, "fetch_ges_params_catalog",
            lambda *a, **k: cat.copy(),
        )

        df, stats = build_labels(
            h5_path=h5_path,
            cache_dir=tmp_path,
            min_per_class=1,
            allow_drop_underfilled=True,
            e_teff_max_k=200.0,
        )
        # Only the e_teff=100 row should survive (G class, 5500 K).
        assert len(df) == 1
        assert stats.n_dropped_e_teff == 2

    def test_default_threshold_is_200(self):
        sig = inspect.signature(build_labels)
        assert sig.parameters["e_teff_max_k"].default == 200.0

    def test_stats_records_dropped_count(self, tmp_path, monkeypatch):
        # All three rows above threshold -> all dropped, n_dropped_e_teff=3.
        ras = np.array([10.0, 20.0, 30.0], dtype=float)
        decs = np.array([-10.0, -20.0, -30.0], dtype=float)
        h5_path = tmp_path / "spec.h5"
        _make_synthetic_hdf5(h5_path, ras, decs)

        cat = pd.DataFrame({
            "cname": ["c0", "c1", "c2"],
            "ra_deg": ras,
            "dec_deg": decs,
            "teff_k": np.array([5500.0, 5500.0, 5500.0]),
            "e_teff_k": np.array([300.0, 400.0, 500.0]),
            "logg": np.array([4.5] * 3),
            "e_logg": np.array([0.1] * 3),
            "feh": np.array([0.0] * 3),
            "e_feh": np.array([0.05] * 3),
            "snr": np.array([100.0] * 3),
            "rv": np.array([0.0] * 3),
            "e_rv": np.array([1.0] * 3),
            "rec_setup": ["U580"] * 3,
            "setup": ["U580"] * 3,
            "ges_type": ["GE_SD"] * 3,
        })
        monkeypatch.setattr(
            labels_module, "fetch_ges_params_catalog",
            lambda *a, **k: cat.copy(),
        )

        df, stats = build_labels(
            h5_path=h5_path,
            cache_dir=tmp_path,
            min_per_class=1,
            allow_drop_underfilled=True,
            e_teff_max_k=200.0,
        )
        assert len(df) == 0
        assert stats.n_dropped_e_teff == 3
        assert stats.per_class_dropped_e_teff is not None
        assert stats.per_class_dropped_e_teff["G"] == 3


class TestGhostAudit:
    """reversal trigger: mixed-setup ghost rate > 0.05."""

    def test_no_ghosts_when_all_match(self):
        cat = pd.DataFrame({
            "cname": ["a", "b", "c"],
            "rec_setup": ["U580", "U580", "U580"],
        })
        matched = {"a", "b", "c"}
        audit = audit_ghost_cnames(cat, matched)
        assert audit.n_ghost == 0
        assert audit.n_ghost_pure_u580 == 0
        assert audit.n_ghost_mixed_setup == 0
        assert audit.mixed_setup_ghost_rate == 0.0
        assert audit.sample_ghost_cnames == []

    def test_counts_pure_and_mixed_separately(self):
        cat = pd.DataFrame({
            "cname": ["a", "b", "c", "d", "e"],
            "rec_setup": [
                "U580", "U580", # pure U580
                "HR10|HR21:U580", # mixed
                "U520|U580", # mixed
                "U580|U520", # mixed
            ],
        })
        # b is a ghost (pure U580). c and d are ghosts (mixed). a and e match.
        matched = {"a", "e"}
        audit = audit_ghost_cnames(cat, matched)
        assert audit.n_catalog_rows == 5
        assert audit.n_ghost == 3  # b, c, d
        assert audit.n_ghost_pure_u580 == 1  # b
        assert audit.n_ghost_mixed_setup == 2  # c, d
        assert audit.n_mixed_setup_total == 3  # c, d, e
        assert audit.mixed_setup_ghost_rate == pytest.approx(2.0 / 3.0)

    def test_sample_ghost_cnames_truncated_to_20(self):
        n = 50
        cat = pd.DataFrame({
            "cname": [f"cn{i:03d}" for i in range(n)],
            "rec_setup": ["U580"] * n,
        })
        matched: set[str] = set()  # all are ghosts
        audit = audit_ghost_cnames(cat, matched)
        assert audit.n_ghost == n
        assert len(audit.sample_ghost_cnames) == 20
        assert audit.sample_ghost_cnames == [f"cn{i:03d}" for i in range(20)]


class TestBuildLabelsETeffCut:
    """Verify cut order: cross-match -> E_TEFF cut -> MK binning."""

    def test_e_teff_cut_applied_after_match_before_binning(
        self, tmp_path, monkeypatch
    ):
        # 4 spectra. Catalog has 4 matching rows; 2 with bad E_TEFF.
        # 2 of the 4 catalog Teff land in OTHER (Teff=11000) before any cut.
        # We want to confirm n_dropped_e_teff counts BAD E_TEFF rows
        # regardless of MK class (so per_class_dropped includes OTHER-Teff
        # rows as the class they would have binned to, or 0 since OTHER
        # is not an allowed class).
        ras = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        decs = np.array([-10.0, -20.0, -30.0, -40.0], dtype=float)
        h5_path = tmp_path / "spec.h5"
        _make_synthetic_hdf5(h5_path, ras, decs)

        cat = pd.DataFrame({
            "cname": ["c0", "c1", "c2", "c3"],
            "ra_deg": ras,
            "dec_deg": decs,
            "teff_k": np.array([5500.0, 5500.0, 4500.0, 4500.0]),
            "e_teff_k": np.array([50.0, 250.0, 50.0, 250.0]), # 2 above 200
            "logg": np.array([4.5] * 4),
            "e_logg": np.array([0.1] * 4),
            "feh": np.array([0.0] * 4),
            "e_feh": np.array([0.05] * 4),
            "snr": np.array([100.0] * 4),
            "rv": np.array([0.0] * 4),
            "e_rv": np.array([1.0] * 4),
            "rec_setup": ["U580"] * 4,
            "setup": ["U580"] * 4,
            "ges_type": ["GE_SD"] * 4,
        })
        monkeypatch.setattr(
            labels_module, "fetch_ges_params_catalog",
            lambda *a, **k: cat.copy(),
        )

        df, stats = build_labels(
            h5_path=h5_path,
            cache_dir=tmp_path,
            min_per_class=1,
            allow_drop_underfilled=True,
            e_teff_max_k=200.0,
        )
        # Two rows survive (one G at 5500, one K at 4500), both with e_teff=50.
        assert len(df) == 2
        assert set(df["mk_class"]) == {"G", "K"}
        assert stats.n_dropped_e_teff == 2
        # per_class_dropped_e_teff is keyed on MK class assigned BEFORE the cut.
        assert stats.per_class_dropped_e_teff["G"] == 1
        assert stats.per_class_dropped_e_teff["K"] == 1
        # ghost audit: all 4 catalog rows matched (we built HDF5 from them)
        assert stats.ghost_audit is not None
        assert stats.ghost_audit.n_ghost == 0
