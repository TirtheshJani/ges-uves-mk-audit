"""Smoke tests for scripts/build_manuscript_dossier.py.

The dossier is generated from gate-accepted through artifacts. The smoke test runs the script in-process against the
existing artifacts and asserts both output files materialise with the
expected key claim_ids.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_manuscript_dossier import main as dossier_main

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"

REQUIRED_CLAIM_IDS = (
    "macro_f1_test",
    "delta_acc_h_balmer_f",
    "agreement_rate",
)

# Forbidden punctuation codepoints, derived from chr() at runtime so
# this test file itself contains no literal em or en dash bytes.
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)


def _artifacts_present() -> bool:
    needed = [
        ART / "metrics.json",
        ART / "interpret" / "triangulation_report.json",
        ART / "interpret" / "shap_stability.json",
        ART / "ablation" / "gate_eval.json",
        ART / "ablation" / "line_match.json",
        ART / "benchmark" / "benchmark_report.json",
    ]
    return all(p.exists() for p in needed)


@pytest.mark.skipif(
    not _artifacts_present(),
    reason="-5 artifacts not present; smoke test skipped",
)
def test_dossier_smoke(tmp_path: Path) -> None:
    """Run the script and verify both outputs are non-empty with key claims."""
    csv_out = tmp_path / "manuscript_numbers.csv"
    md_out = tmp_path / "manuscript_numbers.md"

    rc = dossier_main(["--csv-out", str(csv_out), "--md-out", str(md_out)])
    assert rc == 0

    assert csv_out.exists()
    assert md_out.exists()
    assert csv_out.stat().st_size > 0
    assert md_out.stat().st_size > 0

    with csv_out.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) > 0

    claim_ids = {r["claim_id"] for r in rows}
    for needed in REQUIRED_CLAIM_IDS:
        assert needed in claim_ids, f"missing claim_id {needed!r} in dossier"

    md_text = md_out.read_text(encoding="utf-8")
    for needed in REQUIRED_CLAIM_IDS:
        assert needed in md_text, f"missing claim_id {needed!r} in markdown"

    for path in (csv_out, md_out):
        text = path.read_text(encoding="utf-8")
        assert EM_DASH not in text, f"em dash found in {path}"
        assert EN_DASH not in text, f"en dash found in {path}"
