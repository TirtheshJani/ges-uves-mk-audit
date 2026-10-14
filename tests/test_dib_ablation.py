"""Line-set construction of ``scripts/revision/dib_6283_ablation.py``.

Checks the ``DIB_6283`` window and its bin count against the deposited
``artifacts/features.npz`` grid without running the ablation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURES = _REPO_ROOT / "artifacts" / "features.npz"


def _load_dib_module():
    path = _REPO_ROOT / "scripts" / "revision" / "dib_6283_ablation.py"
    spec = importlib.util.spec_from_file_location("dib_6283_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dib_line_set_follows_the_five_deposited_sets() -> None:
    dib = _load_dib_module()
    sets = dib.dib_line_sets()
    assert list(sets) == ["H_balmer", "Mg_b", "Na_D", "Ca_I", "Fe_Cr", "DIB_6283"]
    assert sets["DIB_6283"] == [(6270.0, 6295.0)]
    assert dib.DIB_WINDOW == (6270.0, 6295.0)


@pytest.mark.skipif(not _FEATURES.exists(), reason="artifacts/features.npz not present")
def test_dib_window_covers_eight_bins_including_the_two_shap_bins() -> None:
    from src.interpret.ablation import _window_mask

    dib = _load_dib_module()
    wave_centers = np.load(_FEATURES, allow_pickle=False)["wave_centers"]
    mask = _window_mask(wave_centers, dib.dib_line_sets()["DIB_6283"])
    assert int(mask.sum()) == 8
    centers = wave_centers[mask]
    assert np.any(np.abs(centers - 6283.1) < 0.1)
    assert np.any(np.abs(centers - 6286.2) < 0.1)
    # the window overlaps none of the five catalogued sets
    for name, windows in dib.abr.all_line_sets().items():
        assert not (mask & _window_mask(wave_centers, windows)).any(), name
