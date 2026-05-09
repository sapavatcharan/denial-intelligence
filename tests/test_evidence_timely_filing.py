"""Tests for the timely-filing evidence extractor."""

from __future__ import annotations

from gabeo.evidence.timely_filing import check_timely_filing
from gabeo.ingest import join_claim


def _claim(**overrides):  # type: ignore[no-untyped-def]
    rem = {
        "pc_ClaimID": "CLM-T-001",
        "pc_ClaimStatus": "4",
        "pc_ClaimAmount": 1000,
        "pc_ClaimPaid": 0,
        "pc_InsuranceType": "Commercial",
        "pc_ReceivedDate": "2026-03-20",
        "cp_PayerID": "AETNA",
        "pcl_ProcedureCode": "99214",
        "pcl_ChargedAmount": 1000,
        "pcla_AdjustmentGroup": "CO",
        "pcla_AdjustmentReason": "29",
        "pcla_AdjustmentAmount": 1000,
    }
    sub = {
        "ec_ClaimNo": "CLM-T-001",
        "ec_PayerID": "AETNA",
        "ec_InsuranceType": "Commercial",
        "ec_ServiceDateFrom": "2025-06-15",
        "ec_PrincipalDiagnosis": "J06.9",
    }
    rem.update(overrides.get("rem", {}))
    sub.update(overrides.get("sub", {}))
    return join_claim(rem, sub)


def test_aetna_late_filing_flagged_over_limit():
    claim = _claim()
    items = check_timely_filing(claim)
    window = next(i for i in items if i.check_name == "timely_filing.window_check")
    assert window.passed is False
    assert window.observed_value == 278
    assert window.expected_value == 90


def test_medicare_within_365_days_passes():
    claim = _claim(
        rem={"pc_InsuranceType": "Medicare", "pc_ReceivedDate": "2026-01-15", "cp_PayerID": "MEDICARE"},
        sub={"ec_PayerID": "MEDICARE", "ec_InsuranceType": "Medicare", "ec_ServiceDateFrom": "2025-08-01"},
    )
    items = check_timely_filing(claim)
    window = next(i for i in items if i.check_name == "timely_filing.window_check")
    assert window.passed is True
    assert window.expected_value == 365


def test_secondary_payer_anchors_to_eob_date_and_passes():
    claim = _claim(
        rem={"pc_ReceivedDate": "2026-03-01"},
        sub={
            "ec_ServiceDateFrom": "2025-06-15",
            "ec_OtherPayerName": "Medicare",
            "ec_OtherPayerPaidDate": "2026-02-10",
        },
    )
    items = check_timely_filing(claim)
    window = next(i for i in items if i.check_name == "timely_filing.window_check")
    assert window.passed is True
    assert "OtherPayerPaidDate" in window.fields_referenced[0]
    assert any(i.check_name == "timely_filing.secondary_anchor" for i in items)


def test_delay_reason_code_is_surfaced():
    claim = _claim(sub={"ec_DelayReasonCode": "9"})
    items = check_timely_filing(claim)
    assert any(i.check_name == "timely_filing.delay_reason_present" for i in items)
