# SPDX-License-Identifier: MIT
"""Interpretability module for the MK stellar-type classifier.

The registry in `lines.py` is the single source of truth for diagnostic
rest-wavelengths used by both the line-match audit and the masked-line
ablation (`ablation.py`). `occlusion.py` holds the sliding-window occlusion
trace and re-exports the ablation names for backwards compatibility.
"""
