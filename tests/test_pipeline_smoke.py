"""Smoke-test the full deterministic pipeline against the synthetic dataset.

No LLM is called here. We verify:

  * Every claim in `data/synthetic/claims.jsonl` parses cleanly.
  * The deterministic evidence layer runs without exception.
  * Brief-sample claims surface the expected critical signals.
  * Adversarial claims surface the expected nuance signals (these are the
    cases that prove the system reasons beyond CARC-code lookup).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gabeo.evidence import run_all
from gabeo.ingest import load_claims_jsonl

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "claims.jsonl"


@pytest.fixture(scope="module")
def claims():  # type: ignore[no-untyped-def]
    if not DATA.exists():
        pytest.skip(f"Synthetic dataset not generated at {DATA}")
    return load_claims_jsonl(DATA)


def test_dataset_loads_cleanly(claims):  # type: ignore[no-untyped-def]
    assert len(claims) >= 30
    assert any(c.is_denied for c in claims)
    assert any(not c.is_denied for c in claims)


def test_all_claims_run_evidence_without_error(claims):  # type: ignore[no-untyped-def]
    for claim in claims:
        items = run_all(claim)
        assert isinstance(items, list)


def _by_id(claims, claim_id):  # type: ignore[no-untyped-def]
    matches = [c for c in claims if c.claim_id == claim_id]
    assert matches, f"Missing expected claim {claim_id}"
    return matches[0]


def test_brief_sample_A_timely_filing_is_flagged(claims):  # type: ignore[no-untyped-def]
    claim = _by_id(claims, "CLM-2026-00142")
    items = run_all(claim)
    window = next(i for i in items if i.check_name == "timely_filing.window_check")
    assert not window.passed
    assert window.expected_value == 180


def test_brief_sample_C_diagnosis_supports_appeal(claims):  # type: ignore[no-untyped-def]
    """Brief sample C: M54.5 IS supportive for 72148, so the medical-necessity
    denial is wrong on its face."""
    claim = _by_id(claims, "CLM-2026-00391")
    items = run_all(claim)
    supports = [
        i for i in items if i.check_name.startswith("medical_necessity.") and i.passed
    ]
    assert supports


def test_adversarial_29_secondary_anchor_passes_window(claims):  # type: ignore[no-untyped-def]
    claim = _by_id(claims, "CLM-ADV-29-SECONDARY")
    items = run_all(claim)
    window = next(i for i in items if i.check_name == "timely_filing.window_check")
    assert window.passed, "Secondary claim should pass when anchored to EOB date"
    assert any(i.check_name == "timely_filing.secondary_anchor" for i in items)


def test_adversarial_50_repoint_to_diag2_is_recognized(claims):  # type: ignore[no-untyped-def]
    claim = _by_id(claims, "CLM-ADV-50-DIAG2")
    items = run_all(claim)
    assert any(
        i.check_name == "medical_necessity.secondary_supports" and i.passed for i in items
    )


def test_adversarial_18_bilateral_missing_modifier_is_flagged(claims):  # type: ignore[no-untyped-def]
    claim = _by_id(claims, "CLM-ADV-18-BILATERAL")
    items = run_all(claim)
    assert any(
        "bilateral_modifier_missing" in i.check_name and not i.passed for i in items
    )


def test_adversarial_197_auth_present_supports_appeal(claims):  # type: ignore[no-untyped-def]
    claim = _by_id(claims, "CLM-ADV-197-AUTH-PRESENT")
    items = run_all(claim)
    assert any(i.check_name == "prior_auth.required_and_present" and i.passed for i in items)
