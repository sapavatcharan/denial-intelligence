"""Generate a synthetic dataset of 835/837 claims with gold labels.

Output: a single JSONL where each line has:
    {"835": {...}, "837": {...}, "gold": {...}}

The dataset deliberately includes:
  * The 4 sample claims from the brief (as anchors / reproducibility check).
  * Textbook denials, one per top CARC code.
  * Clean paid claims (so the historical retrieval store has positive examples).
  * Adversarial cases where a naive CARC-code lookup would give the wrong answer.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PAYERS = [
    ("MEDICARE",  "Medicare Part B",         "Medicare"),
    ("AETNA",     "Aetna",                   "Commercial"),
    ("UHC",       "United Healthcare",       "Commercial"),
    ("CIGNA",     "Cigna",                   "Commercial"),
    ("BCBS-IL",   "Blue Cross Blue Shield",  "Commercial"),
    ("HUMANA",    "Humana",                  "Commercial"),
    ("MEDICAID",  "State Medicaid",          "Medicaid"),
]

PROCEDURES = [
    ("99213", ["any"], 180.0),
    ("99214", ["any"], 280.0),
    ("72148", ["M51", "M54.5", "M54.4"], 1500.0),
    ("70553", ["G44", "R51", "S06"], 2400.0),
    ("27447", ["M17", "Z47.1"], 32000.0),
    ("29881", ["S83.2", "M23"], 4800.0),
    ("45378", ["K", "Z12.11"], 1200.0),
    ("93000", ["I", "R00"], 80.0),
    ("11042", ["L", "I70"], 600.0),
    ("20610", ["M", "S"], 220.0),
]

DIAG_BY_PREFIX = {
    "M17":   ["M17.11", "M17.12"],
    "M51":   ["M51.16", "M51.17"],
    "M54.5": ["M54.5"],
    "M54.4": ["M54.41", "M54.42"],
    "S83.2": ["S83.241A", "S83.242A"],
    "M23":   ["M23.211", "M23.221"],
    "K":     ["K35.20", "K57.30"],
    "Z12.11":["Z12.11"],
    "I":     ["I10", "I25.10", "I50.9"],
    "R00":   ["R00.0", "R00.1"],
    "G44":   ["G44.1"],
    "R51":   ["R51"],
    "S06":   ["S06.0X0A"],
    "L":     ["L97.213"],
    "I70":   ["I70.232"],
    "M":     ["M25.561", "M25.562"],
    "S":     ["S83.241A"],
    "Z47.1": ["Z47.1"],
}


def _rand_dx_for_proc(rng: random.Random, proc: tuple[str, list[str], float]) -> str:
    prefixes = proc[1]
    if "any" in prefixes:
        return rng.choice(["I10", "J06.9", "K35.20", "M54.5", "Z00.00"])
    pref = rng.choice(prefixes)
    return rng.choice(DIAG_BY_PREFIX.get(pref, [pref]))


def _date_iso(d: date) -> str:
    return d.isoformat()


def _make_paid_claim(claim_id: str, rng: random.Random) -> dict[str, Any]:
    payer_id, payer_name, ins = rng.choice(PAYERS)
    proc = rng.choice(PROCEDURES)
    dx = _rand_dx_for_proc(rng, proc)
    service_date = date(2025, 6, 1) + timedelta(days=rng.randint(0, 200))
    received = service_date + timedelta(days=rng.randint(7, 45))
    amt = round(proc[2] * rng.uniform(0.95, 1.05), 2)
    paid = round(amt * rng.uniform(0.55, 0.85), 2)
    rem = {
        "pc_ClaimID": claim_id,
        "pc_ClaimStatus": "1",
        "pc_ClaimAmount": amt,
        "pc_ClaimPaid": paid,
        "pc_PatientResponsibility": round(amt - paid, 2),
        "pc_InsuranceType": ins,
        "pc_ReceivedDate": _date_iso(received),
        "pc_StatementBegin": _date_iso(service_date),
        "pc_StatementEnd": _date_iso(service_date),
        "cp_PayerID": payer_id,
        "cp_PayerName": payer_name,
        "pcl_ProcedureCode": proc[0],
        "pcl_ChargedAmount": amt,
        "pcl_PaidAmount": paid,
        "pcl_AllowedAmount": paid,
        "pcl_ServiceDate": _date_iso(service_date),
    }
    sub = {
        "ec_ClaimNo": claim_id,
        "ec_Amount": amt,
        "ec_PayerID": payer_id,
        "ec_PayerName": payer_name,
        "ec_InsuranceType": ins,
        "ec_PrincipalDiagnosis": dx,
        "ec_BillProvNPI": "9999900001",
        "ec_RendProvNPI": "9999900002",
        "ec_ServiceDateFrom": _date_iso(service_date),
        "ec_ServiceDateTo": _date_iso(service_date),
        "ec_ClaimFrequency": "1",
        "ec_PlaceOfService": rng.choice(["11", "21", "22"]),
    }
    gold = {
        "is_denied": False,
        "primary_carc": None,
        "expected_recoverability": None,
        "expected_root_cause_keywords": ["paid"],
        "scenario": "clean_paid",
    }
    return {"835": rem, "837": sub, "gold": gold}


def _denied_skeleton(
    claim_id: str,
    payer: tuple[str, str, str],
    proc: tuple[str, list[str], float],
    carc: str,
    *,
    service_date: date,
    received: date,
    dx: str | None = None,
    extras_835: dict[str, Any] | None = None,
    extras_837: dict[str, Any] | None = None,
    remark_codes: str | None = None,
) -> dict[str, Any]:
    payer_id, payer_name, ins = payer
    amt = round(proc[2] * 1.0, 2)
    rem: dict[str, Any] = {
        "pc_ClaimID": claim_id,
        "pc_ClaimStatus": "4",
        "pc_ClaimAmount": amt,
        "pc_ClaimPaid": 0.0,
        "pc_InsuranceType": ins,
        "pc_ReceivedDate": _date_iso(received),
        "pc_StatementBegin": _date_iso(service_date),
        "pc_StatementEnd": _date_iso(service_date),
        "cp_PayerID": payer_id,
        "cp_PayerName": payer_name,
        "pcl_ProcedureCode": proc[0],
        "pcl_ChargedAmount": amt,
        "pcl_PaidAmount": 0.0,
        "pcla_AdjustmentGroup": "CO",
        "pcla_AdjustmentReason": carc,
        "pcla_AdjustmentAmount": amt,
    }
    if remark_codes:
        rem["pcl_RemarkCodes"] = remark_codes
    if extras_835:
        rem.update(extras_835)
    sub: dict[str, Any] = {
        "ec_ClaimNo": claim_id,
        "ec_Amount": amt,
        "ec_PayerID": payer_id,
        "ec_PayerName": payer_name,
        "ec_InsuranceType": ins,
        "ec_PrincipalDiagnosis": dx if dx is not None else "",
        "ec_BillProvNPI": "9999900001",
        "ec_RendProvNPI": "9999900002",
        "ec_ServiceDateFrom": _date_iso(service_date),
        "ec_ServiceDateTo": _date_iso(service_date),
        "ec_ClaimFrequency": "1",
    }
    if extras_837:
        sub.update(extras_837)
    return {"835": rem, "837": sub}


def _textbook_claims() -> list[dict[str, Any]]:
    """One textbook denial per top CARC, with deterministic gold labels."""
    out: list[dict[str, Any]] = []

    out.append({
        **_denied_skeleton(
            "CLM-SYN-29-001",
            ("AETNA", "Aetna", "Commercial"),
            ("99214", ["any"], 280.0),
            "29",
            service_date=date(2025, 6, 15),
            received=date(2026, 3, 20),
            dx="J06.9",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "29",
            "expected_recoverability": "not_recoverable",
            "expected_root_cause_keywords": ["timely", "278", "90"],
            "scenario": "textbook_timely_filing",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-16-001",
            ("MEDICARE", "Medicare Part B", "Medicare"),
            ("27447", ["M17", "Z47.1"], 32000.0),
            "16",
            service_date=date(2026, 1, 8),
            received=date(2026, 2, 10),
            dx="M17.11",
            remark_codes="N20",
            extras_837={"ec_PriorAuthorization": "AUTH-998877", "ec_TypeOfBill": "131"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "16",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["modifier", "N20"],
            "scenario": "textbook_missing_info",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-50-001",
            ("AETNA", "Aetna", "Commercial"),
            ("72148", ["M54.5"], 1500.0),
            "50",
            service_date=date(2026, 2, 20),
            received=date(2026, 3, 1),
            dx="M54.5",
            remark_codes="N386",
            extras_837={"ec_RendProvSpecialty": "Radiology"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "50",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["medical necessity", "M54.5"],
            "scenario": "textbook_medical_necessity",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-18-001",
            ("UHC", "United Healthcare", "Commercial"),
            ("99213", ["any"], 180.0),
            "18",
            service_date=date(2026, 1, 10),
            received=date(2026, 1, 25),
            dx="J20.9",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "18",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["duplicate"],
            "scenario": "textbook_duplicate",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-197-001",
            ("AETNA", "Aetna", "Commercial"),
            ("70553", ["G44"], 2400.0),
            "197",
            service_date=date(2026, 2, 5),
            received=date(2026, 2, 25),
            dx="G44.1",
            extras_837={"ec_PriorAuthorization": ""},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "197",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["prior authorization", "Aetna"],
            "scenario": "textbook_prior_auth_missing",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-4-001",
            ("UHC", "United Healthcare", "Commercial"),
            ("27447", ["M17"], 32000.0),
            "4",
            service_date=date(2026, 1, 10),
            received=date(2026, 1, 30),
            dx="M17.11",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "4",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["modifier", "LT", "RT"],
            "scenario": "textbook_coding_modifier",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-97-001",
            ("CIGNA", "Cigna", "Commercial"),
            ("11042", ["L"], 600.0),
            "97",
            service_date=date(2026, 1, 5),
            received=date(2026, 1, 18),
            dx="L97.213",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "97",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["bundled", "59"],
            "scenario": "textbook_bundling",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-252-001",
            ("MEDICARE", "Medicare Part B", "Medicare"),
            ("70553", ["S06"], 2400.0),
            "252",
            service_date=date(2026, 1, 12),
            received=date(2026, 1, 25),
            dx="S06.0X0A",
            remark_codes="N350",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "252",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["documentation", "attachment"],
            "scenario": "textbook_attachment_required",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-SYN-11-001",
            ("HUMANA", "Humana", "Commercial"),
            ("45378", ["K"], 1200.0),
            "11",
            service_date=date(2026, 1, 20),
            received=date(2026, 2, 1),
            dx="Z00.00",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "11",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["diagnosis", "procedure"],
            "scenario": "textbook_dx_proc_inconsistent",
        },
    })

    return out


def _adversarial_claims() -> list[dict[str, Any]]:
    """Cases that defeat naive CARC-code lookup. The hardest 1% of the dataset."""
    out: list[dict[str, Any]] = []

    out.append({
        **_denied_skeleton(
            "CLM-ADV-29-SECONDARY",
            ("AETNA", "Aetna", "Commercial"),
            ("99214", ["any"], 280.0),
            "29",
            service_date=date(2025, 6, 15),
            received=date(2026, 3, 1),
            dx="J06.9",
            extras_837={
                "ec_OtherPayerName": "Medicare Part B",
                "ec_OtherPayerPaid": 100.0,
                "ec_OtherPayerPaidDate": "2026-02-10",
            },
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "29",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["secondary", "EOB", "appeal"],
            "scenario": "adversarial_29_secondary_anchor",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-ADV-50-DIAG2",
            ("AETNA", "Aetna", "Commercial"),
            ("72148", ["M54.5"], 1500.0),
            "50",
            service_date=date(2026, 2, 20),
            received=date(2026, 3, 1),
            dx="Z00.00",
            extras_837={"ec_Diag2": "M51.16", "ec_RendProvSpecialty": "Radiology"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "50",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["secondary", "diagnosis", "M51.16", "repoint"],
            "scenario": "adversarial_50_repoint_to_diag2",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-ADV-18-BILATERAL",
            ("UHC", "United Healthcare", "Commercial"),
            ("27447", ["M17"], 32000.0),
            "18",
            service_date=date(2026, 1, 10),
            received=date(2026, 1, 25),
            dx="M17.11",
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "18",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["bilateral", "LT", "RT", "modifier"],
            "scenario": "adversarial_18_bilateral_missing_modifier",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-ADV-197-AUTH-PRESENT",
            ("AETNA", "Aetna", "Commercial"),
            ("70553", ["G44"], 2400.0),
            "197",
            service_date=date(2026, 2, 5),
            received=date(2026, 2, 25),
            dx="G44.1",
            extras_837={"ec_PriorAuthorization": "AUTH-AET-441900"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "197",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["authorization", "AUTH-AET-441900", "appeal"],
            "scenario": "adversarial_197_auth_present_appeal",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-ADV-29-DELAY-REASON",
            ("UHC", "United Healthcare", "Commercial"),
            ("99213", ["any"], 180.0),
            "29",
            service_date=date(2025, 7, 1),
            received=date(2026, 3, 15),
            dx="J20.9",
            extras_837={"ec_DelayReasonCode": "9"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "29",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["delay reason", "appeal"],
            "scenario": "adversarial_29_delay_reason_present",
        },
    })

    out.append({
        **_denied_skeleton(
            "CLM-ADV-50-PRINCIPAL-OK",
            ("AETNA", "Aetna", "Commercial"),
            ("72148", ["M51", "M54.5"], 1500.0),
            "50",
            service_date=date(2026, 2, 20),
            received=date(2026, 3, 1),
            dx="M51.16",
            extras_837={"ec_RendProvSpecialty": "Radiology"},
        ),
        "gold": {
            "is_denied": True,
            "primary_carc": "50",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["M51.16", "supportive", "appeal"],
            "scenario": "adversarial_50_principal_supports_appeal_strong",
        },
    })

    return out


def _brief_sample_claims() -> list[dict[str, Any]]:
    """Verbatim from the assignment PDF, used as anchors."""
    out = []
    out.append({
        "835": {
            "pc_ClaimID": "CLM-2026-00142",
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 4500.00,
            "pc_ClaimPaid": 0.00,
            "pc_InsuranceType": "Commercial",
            "pc_ReceivedDate": "2026-03-20",
            "pc_StatementBegin": "2025-06-15",
            "pc_StatementEnd": "2025-06-15",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "29",
            "pcla_AdjustmentAmount": 4500.00,
            "pcl_ProcedureCode": "99214",
            "pcl_ChargedAmount": 4500.00,
            "pcl_PaidAmount": 0.00,
            "cp_PayerID": "BCBS-IL",
            "cp_PayerName": "Blue Cross Blue Shield",
        },
        "837": {
            "ec_ClaimNo": "CLM-2026-00142",
            "ec_PayerName": "Blue Cross Blue Shield",
            "ec_PayerID": "BCBS-IL",
            "ec_InsuranceType": "Commercial",
            "ec_ServiceDateFrom": "2025-06-15",
            "ec_PrincipalDiagnosis": "J06.9",
            "ec_BillProvNPI": "1234567890",
            "ec_DelayReasonCode": "",
            "ec_ClaimFrequency": "1",
            "ec_SubscriberID": "XYZ123456",
        },
        "gold": {
            "is_denied": True,
            "primary_carc": "29",
            "expected_recoverability": "not_recoverable",
            "expected_root_cause_keywords": ["timely filing", "BCBS", "180"],
            "scenario": "brief_sample_A_timely_filing",
        },
    })

    out.append({
        "835": {
            "pc_ClaimID": "CLM-2026-00287",
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 12800.00,
            "pc_ClaimPaid": 0.00,
            "pc_InsuranceType": "Medicare",
            "pc_ReceivedDate": "2026-02-10",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "16",
            "pcla_AdjustmentAmount": 12800.00,
            "pcl_ProcedureCode": "27447",
            "pcl_ProcedureModifier1": "",
            "pcl_RemarkCodes": "N20",
            "cp_PayerID": "MEDICARE",
            "cp_PayerName": "Medicare Part B",
        },
        "837": {
            "ec_ClaimNo": "CLM-2026-00287",
            "ec_PayerName": "Medicare Part B",
            "ec_PayerID": "MEDICARE",
            "ec_InsuranceType": "Medicare",
            "ec_ServiceDateFrom": "2026-01-08",
            "ec_PrincipalDiagnosis": "M17.11",
            "ec_PriorAuthorization": "AUTH-998877",
            "ec_BillProvNPI": "9876543210",
            "ec_TypeOfBill": "131",
            "ec_ClaimFrequency": "1",
        },
        "gold": {
            "is_denied": True,
            "primary_carc": "16",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["missing", "modifier", "27447"],
            "scenario": "brief_sample_B_missing_info_modifier",
        },
    })

    out.append({
        "835": {
            "pc_ClaimID": "CLM-2026-00391",
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 8200.00,
            "pc_ClaimPaid": 0.00,
            "pc_InsuranceType": "Commercial",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "50",
            "pcla_AdjustmentAmount": 8200.00,
            "pcl_ProcedureCode": "72148",
            "pcl_RemarkCodes": "N386",
            "cp_PayerID": "AETNA",
            "cp_PayerName": "Aetna",
            "pc_ReceivedDate": "2026-03-01",
        },
        "837": {
            "ec_ClaimNo": "CLM-2026-00391",
            "ec_PayerName": "Aetna",
            "ec_PayerID": "AETNA",
            "ec_InsuranceType": "Commercial",
            "ec_ServiceDateFrom": "2026-02-20",
            "ec_PrincipalDiagnosis": "M54.5",
            "ec_Diag2": "M51.16",
            "ec_PriorAuthorization": "",
            "ec_BillProvNPI": "5678901234",
            "ec_RendProvSpecialty": "Radiology",
        },
        "gold": {
            "is_denied": True,
            "primary_carc": "50",
            "expected_recoverability": "recoverable",
            "expected_root_cause_keywords": ["medical necessity", "M51.16", "repoint"],
            "scenario": "brief_sample_C_medical_necessity_secondary_dx",
        },
    })

    out.append({
        "835": {
            "pc_ClaimID": "CLM-2026-00455",
            "pc_ClaimStatus": "4",
            "pc_ClaimAmount": 3200.00,
            "pc_ClaimPaid": 0.00,
            "pc_InsuranceType": "Commercial",
            "pcla_AdjustmentGroup": "CO",
            "pcla_AdjustmentReason": "18",
            "pcla_AdjustmentAmount": 3200.00,
            "pcl_ProcedureCode": "99213",
            "cp_PayerID": "UHC",
            "cp_PayerName": "United Healthcare",
            "pc_ReceivedDate": "2026-01-25",
        },
        "837": {
            "ec_ClaimNo": "CLM-2026-00455",
            "ec_PayerName": "United Healthcare",
            "ec_PayerID": "UHC",
            "ec_InsuranceType": "Commercial",
            "ec_ServiceDateFrom": "2026-01-10",
            "ec_ClaimFrequency": "1",
            "ec_PrincipalDiagnosis": "J20.9",
            "ec_BillProvNPI": "1234567890",
        },
        "gold": {
            "is_denied": True,
            "primary_carc": "18",
            "expected_recoverability": "needs_review",
            "expected_root_cause_keywords": ["duplicate", "history"],
            "scenario": "brief_sample_D_duplicate",
        },
    })

    return out


def generate(n: int, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    out.extend(_brief_sample_claims())
    out.extend(_textbook_claims())
    out.extend(_adversarial_claims())
    paid_needed = max(n - len(out), 10)
    for i in range(paid_needed):
        out.append(_make_paid_claim(f"CLM-PAID-{i:04d}", rng))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", type=str, default="data/synthetic/claims.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    claims = generate(args.n, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")

    paid = sum(1 for c in claims if not c["gold"]["is_denied"])
    denied = len(claims) - paid
    print(f"Wrote {len(claims)} claims to {out_path} ({denied} denied, {paid} paid).")


if __name__ == "__main__":
    main()
