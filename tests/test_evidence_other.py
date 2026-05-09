"""Smoke tests for the remaining evidence extractors."""

from __future__ import annotations

from gabeo.evidence.coding import check_coding
from gabeo.evidence.duplicate import check_duplicate
from gabeo.evidence.medical_necessity import check_medical_necessity
from gabeo.evidence.missing_info import check_missing_info
from gabeo.evidence.prior_auth import check_prior_auth
from gabeo.ingest import join_claim


def test_prior_auth_required_but_missing_for_aetna_mri():
    rem = {
        "pc_ClaimID": "CLM-PA-001",
        "pcl_ProcedureCode": "72148",
        "pcl_ChargedAmount": 1500,
        "pcla_AdjustmentGroup": "CO",
        "pcla_AdjustmentReason": "197",
        "pcla_AdjustmentAmount": 1500,
    }
    sub = {
        "ec_ClaimNo": "CLM-PA-001",
        "ec_PayerID": "AETNA",
        "ec_PayerName": "Aetna",
        "ec_InsuranceType": "Commercial",
        "ec_ServiceDateFrom": "2026-02-20",
        "ec_PrincipalDiagnosis": "M54.5",
        "ec_PriorAuthorization": "",
    }
    claim = join_claim(rem, sub)
    items = check_prior_auth(claim)
    assert any(
        i.check_name == "prior_auth.required_but_missing" and not i.passed for i in items
    )


def test_prior_auth_present_supports_appeal():
    rem = {
        "pc_ClaimID": "CLM-PA-002",
        "pcl_ProcedureCode": "72148",
        "pcla_AdjustmentReason": "197",
    }
    sub = {
        "ec_ClaimNo": "CLM-PA-002",
        "ec_PayerID": "AETNA",
        "ec_PriorAuthorization": "AUTH-12345",
        "ec_ServiceDateFrom": "2026-02-20",
        "ec_PrincipalDiagnosis": "M54.5",
    }
    claim = join_claim(rem, sub)
    items = check_prior_auth(claim)
    assert any(i.check_name == "prior_auth.required_and_present" and i.passed for i in items)


def test_medical_necessity_secondary_dx_supports():
    rem = {
        "pc_ClaimID": "CLM-MN-001",
        "pcl_ProcedureCode": "72148",
        "pcla_AdjustmentReason": "50",
    }
    sub = {
        "ec_ClaimNo": "CLM-MN-001",
        "ec_PayerID": "AETNA",
        "ec_ServiceDateFrom": "2026-02-20",
        "ec_PrincipalDiagnosis": "Z00.00",
        "ec_Diag2": "M51.16",
    }
    claim = join_claim(rem, sub)
    items = check_medical_necessity(claim)
    assert any(
        i.check_name == "medical_necessity.secondary_supports" and i.passed for i in items
    )


def test_duplicate_bilateral_modifier_missing_flagged():
    rem = {
        "pc_ClaimID": "CLM-DUP-001",
        "pcl_ProcedureCode": "27447",
        "pcla_AdjustmentReason": "18",
    }
    sub = {
        "ec_ClaimNo": "CLM-DUP-001",
        "ec_PayerID": "UHC",
        "ec_ServiceDateFrom": "2026-01-10",
        "ec_PrincipalDiagnosis": "M17.11",
    }
    claim = join_claim(rem, sub)
    items = check_duplicate(claim)
    assert any(
        "bilateral_modifier_missing" in i.check_name and not i.passed for i in items
    )


def test_coding_modifier_satisfied_with_LT():
    rem = {
        "pc_ClaimID": "CLM-COD-001",
        "pcl_ProcedureCode": "27447",
        "pcl_ProcedureModifier1": "LT",
        "pcla_AdjustmentReason": "4",
    }
    sub = {
        "ec_ClaimNo": "CLM-COD-001",
        "ec_PayerID": "UHC",
        "ec_ServiceDateFrom": "2026-01-10",
        "ec_PrincipalDiagnosis": "M17.11",
    }
    claim = join_claim(rem, sub)
    items = check_coding(claim)
    assert any("required_modifier" in i.check_name and i.passed for i in items)


def test_missing_info_rarc_n386_flags_missing_dx():
    rem = {
        "pc_ClaimID": "CLM-MI-001",
        "pcl_ProcedureCode": "72148",
        "pcl_RemarkCodes": "N386",
        "pcla_AdjustmentReason": "50",
    }
    sub = {
        "ec_ClaimNo": "CLM-MI-001",
        "ec_PayerID": "AETNA",
        "ec_ServiceDateFrom": "2026-02-20",
        "ec_PrincipalDiagnosis": "",
    }
    claim = join_claim(rem, sub)
    items = check_missing_info(claim)
    assert any(i.check_name == "missing_info.rarc_N386" and not i.passed for i in items)
