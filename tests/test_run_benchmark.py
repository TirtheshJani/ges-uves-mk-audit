"""CLI-level tests for scripts/run_benchmark.py.

These tests exercise the helpers and module-level logic without invoking the
full pipeline (no model load, no FITS templates).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make scripts/ importable as a package-less module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

run_benchmark: Any = importlib.import_module("run_benchmark")


class TestFilterToFgk:
    def test_cli_filter_to_fgk_drops_other_rows(self) -> None:
        """ synthetic arrays. After FGK filter,
        n_compared = input_total - count(y_pickles == 'OTHER')."""
        y_pickles = np.array(["F", "G", "K", "OTHER", "OTHER", "G", "F"])
        y_model = np.array(["F", "G", "K", "F", "G", "G", "K"])
        y_p_fgk, y_m_fgk, n_dropped = run_benchmark._filter_to_fgk(y_pickles, y_model)

        assert n_dropped == 2
        assert len(y_p_fgk) == len(y_pickles) - n_dropped
        assert len(y_m_fgk) == len(y_pickles) - n_dropped
        assert "OTHER" not in y_p_fgk.tolist()
        # Order of surviving rows must be preserved.
        expected_p = ["F", "G", "K", "G", "F"]
        expected_m = ["F", "G", "K", "G", "K"]
        assert y_p_fgk.tolist() == expected_p
        assert y_m_fgk.tolist() == expected_m

    def test_filter_to_fgk_all_other(self) -> None:
        y_pickles = np.array(["OTHER", "OTHER"])
        y_model = np.array(["F", "G"])
        y_p, y_m, n = run_benchmark._filter_to_fgk(y_pickles, y_model)
        assert n == 2
        assert len(y_p) == 0
        assert len(y_m) == 0

    def test_filter_to_fgk_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            run_benchmark._filter_to_fgk(
                np.array(["F", "G"]),
                np.array(["F", "G", "K"]),
            )


class TestDecision33Gate:
    def test_pass_when_above_floor_and_n_above_min(self) -> None:
        gate = run_benchmark._decision_33_gate(
            agreement_rate=0.60, n_compared=200, floor=0.55, min_n=100,
        )
        assert gate["status"] == "PASS"
        assert gate["required"] == 0.55
        assert gate["observed"] == 0.60
        assert gate["n_compared"] == 200

    def test_fail_when_below_floor(self) -> None:
        gate = run_benchmark._decision_33_gate(
            agreement_rate=0.50, n_compared=200, floor=0.55, min_n=100,
        )
        assert gate["status"] == "FAIL"

    def test_fail_when_n_too_small(self) -> None:
        gate = run_benchmark._decision_33_gate(
            agreement_rate=0.99, n_compared=50, floor=0.55, min_n=100,
        )
        assert gate["status"] == "FAIL"
