"""Unit tests for src/interpret/benchmark.py (no FITS downloads except synth)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.interpret.benchmark import (
    PICKLES_UVKLIB_MAP,
    benchmark_report,
    best_template_per_spectrum,
    collapse_to_mk,
    load_pickles_library,
    parse_pickles_filename,
    read_pickles_header_type,
)


class TestParsePicklesFilename:
    @pytest.mark.parametrize("n,expected_start", [
        (9, "A0"), (14, "F0"), (23, "G0"), (31, "K0"),
        (65, "A0"), (69, "F0"), (72, "G0"), (78, "K0"),
    ])
    def test_known_dwarfs_and_giants(self, n, expected_start):
        assert parse_pickles_filename(f"uk{n}.fits").startswith(expected_start)

    def test_pickles_prefix_variant(self):
        assert parse_pickles_filename("pickles_uk_31.fits") == "K0V"

    def test_rejects_non_pickles(self):
        with pytest.raises(ValueError):
            parse_pickles_filename("random.fits")

    def test_out_of_range_number_raises(self):
        with pytest.raises(KeyError):
            parse_pickles_filename("uk999.fits")


class TestCollapseToMk:
    @pytest.mark.parametrize("full,coarse", [
        ("A0V", "A"), ("F5III", "F"), ("G8IV", "G"), ("K3V", "K"),
        ("M2V", "OTHER"), ("B8I", "OTHER"), ("O5V", "OTHER"), ("", "OTHER"),
    ])
    def test_mapping(self, full, coarse):
        assert collapse_to_mk(full) == coarse


class TestUvklibMapCoverage:
    def test_all_fgk_dwarfs_represented(self):
        coarse = {collapse_to_mk(v) for v in PICKLES_UVKLIB_MAP.values()}
        assert {"A", "F", "G", "K"} <= coarse

    def test_keys_are_positive_ints(self):
        assert all(isinstance(k, int) and k > 0 for k in PICKLES_UVKLIB_MAP)

    def test_covers_all_131_stsci_files(self):
        assert sorted(PICKLES_UVKLIB_MAP) == list(range(1, 132))

    @pytest.mark.parametrize("n,expected", [
        (1, "O5V"), (16, "F5V"), (17, "F5V"), (20, "F8V"), (26, "G2V"),
        (46, "B2IV"), (86, "K2III"), (108, "F0II"), (109, "F2II"),
        (114, "B0I"), (131, "M2I"),
    ])
    def test_header_transcribed_types(self, n, expected):
        """Spot checks against the COMMENT1 'spectral type' cards of the STScI files."""
        assert PICKLES_UVKLIB_MAP[n] == expected


class TestBestTemplatePerSpectrum:
    def _make_templates(self, n_bins):
        from src.interpret.benchmark import Template

        rng = np.random.default_rng(0)
        flux_a = 1.0 + 0.01 * rng.standard_normal(n_bins).astype(np.float32)
        flux_g = 0.9 + 0.02 * rng.standard_normal(n_bins).astype(np.float32)
        return [
            Template("uk9.fits", "A0V", "A", flux_a),
            Template("uk31.fits", "K0V", "K", flux_g),
        ]

    def test_picks_correct_template(self):
        templates = self._make_templates(n_bins=100)
        X = np.stack([
            templates[0].flux_on_grid, # should match template 0
            templates[1].flux_on_grid, # should match template 1
        ])
        idx = best_template_per_spectrum(X, templates)
        assert idx.tolist() == [0, 1]

    def test_handles_nan_in_template(self):
        templates = self._make_templates(n_bins=100)
        templates[0].flux_on_grid[:5] = np.nan
        X = np.stack([templates[0].flux_on_grid, templates[1].flux_on_grid])
        X = np.where(np.isnan(X), 1.0, X)  # spectra have no NaN after imputation
        idx = best_template_per_spectrum(X, templates)
        assert idx.shape == (2,)


class TestBenchmarkReport:
    def test_perfect_agreement(self):
        y = np.array(["A", "F", "G", "K", "A", "OTHER"])
        rpt = benchmark_report(y, y, class_labels=["A", "F", "G", "K"])
        assert rpt["agreement_rate"] == 1.0
        assert rpt["macro_f1_fgk"] == pytest.approx(1.0)
        assert rpt["n_compared"] == 6

    def test_all_disagree(self):
        y_true = np.array(["A", "F", "G", "K"])
        y_pred = np.array(["F", "G", "K", "A"])
        rpt = benchmark_report(y_pred, y_true, class_labels=["A", "F", "G", "K"])
        assert rpt["agreement_rate"] == 0.0


def _write_synthetic_pickles_fits(path: Path, n_bins: int = 200) -> None:
    """Write a minimal Pickles-shaped FITS template at ``path``."""
    from astropy.io import fits

    wave = np.linspace(4500.0, 7000.0, n_bins).astype(np.float32)
    flux = np.ones(n_bins, dtype=np.float32)
    col_w = fits.Column(name="WAVELENGTH", format="E", array=wave)
    col_f = fits.Column(name="FLUX", format="E", array=flux)
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.BinTableHDU.from_columns([col_w, col_f])
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


class TestLoadPicklesLibrary:
    def test_load_pickles_uses_map_for_all_131_and_skips_out_of_range(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Headerless synthetic templates fall back to the map; indices 100..131
        all load, and indices outside 1..131 are skipped with one INFO log."""
        for n in list(range(100, 132)) + [132, 140]:
            _write_synthetic_pickles_fits(tmp_path / f"pickles_uk_{n}.fits", n_bins=128)

        wc = np.linspace(4800.0, 6800.0, 64).astype(np.float32)
        with caplog.at_level(logging.INFO, logger="src.interpret.benchmark"):
            templates, stats = load_pickles_library(tmp_path, wc)

        assert stats["n_files_seen"] == 34
        assert stats["n_loaded"] == 32
        assert stats["n_type_from_header"] == 0
        assert stats["n_type_from_map"] == 32
        assert stats["n_unmapped_skipped"] == 2
        assert stats["n_invalid_filename"] == 0
        assert len(templates) == 32
        types = {t.filename: t.mk_type for t in templates}
        assert types["pickles_uk_109.fits"] == PICKLES_UVKLIB_MAP[109] == "F2II"
        assert types["pickles_uk_131.fits"] == "M2I"
        msgs = " ".join(rec.getMessage() for rec in caplog.records)
        assert "skipped 2" in msgs

    def test_load_pickles_prefers_header_type(self, tmp_path: Path):
        """A 'spectral type' header card overrides the filename map."""
        from astropy.io import fits

        path = tmp_path / "pickles_uk_9.fits"  # map says A0V
        _write_synthetic_pickles_fits(path, n_bins=128)
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["COMMENT1"] = "spectral type: K3III"
            hdul.flush()
        assert read_pickles_header_type(path) == "K3III"
        wc = np.linspace(4800.0, 6800.0, 64).astype(np.float32)
        templates, stats = load_pickles_library(tmp_path, wc)
        assert stats["n_type_from_header"] == 1
        assert stats["n_type_from_map"] == 0
        assert templates[0].mk_type == "K3III"
        assert templates[0].mk_class == "K"

    def test_read_header_type_none_without_card(self, tmp_path: Path):
        path = tmp_path / "pickles_uk_9.fits"
        _write_synthetic_pickles_fits(path, n_bins=128)
        assert read_pickles_header_type(path) is None

    def test_benchmark_report_fgk_only(self):
        """class_labels=['F','G','K'] returns macro_f1 over FGK only."""
        y_true = np.array(["F", "G", "K", "F", "G", "K"])
        y_pred = np.array(["F", "G", "K", "F", "G", "K"])
        rpt = benchmark_report(
            y_pred_model=y_pred, y_pickles_mk=y_true,
            class_labels=["F", "G", "K"],
        )
        assert rpt["agreement_rate"] == 1.0
        assert rpt["macro_f1_fgk"] == pytest.approx(1.0)
        assert rpt["n_compared"] == 6
        # OTHER is in the labels list (extra), but should have zero counts.
        assert "OTHER" in rpt["labels"]
        assert rpt["per_class_recall"]["OTHER"] == 0.0
