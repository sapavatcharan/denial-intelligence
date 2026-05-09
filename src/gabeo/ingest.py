"""Load 835 / 837 records (JSON or dict) into validated Pydantic models.

We accept the raw EDI flat structures that the brief uses (single dict per
claim with `pc_*`, `pcl_*`, `pcla_*`, `cp_*`, `ec_*` keys). Anything we
don't model is silently dropped via `extra='ignore'`. This is deliberate:
the 835/837 standards have hundreds of fields, and we want to be robust
to anything we don't recognize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    Claim,
    ClaimSubmission,
    LineAdjustment,
    Remittance,
    RemittanceLine,
)


def _gather_modifiers(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for i in range(1, 5):
        v = record.get(f"pcl_ProcedureModifier{i}")
        if v is not None and str(v).strip():
            out.append(str(v).strip())
    return out


def _gather_remark_codes(record: dict[str, Any]) -> list[str]:
    raw = record.get("pcl_RemarkCodes") or record.get("remark_codes")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if x and str(x).strip()]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _build_adjustments(record: dict[str, Any]) -> list[LineAdjustment]:
    if "adjustments" in record and isinstance(record["adjustments"], list):
        return [LineAdjustment.model_validate(a) for a in record["adjustments"]]
    if "pcla_AdjustmentReason" in record:
        return [LineAdjustment.model_validate(record)]
    return []


def _build_lines(record: dict[str, Any]) -> list[RemittanceLine]:
    if "lines" in record and isinstance(record["lines"], list):
        out: list[RemittanceLine] = []
        for line_rec in record["lines"]:
            line = RemittanceLine.model_validate(line_rec)
            if not line.modifiers:
                line.modifiers = _gather_modifiers(line_rec)
            if not line.remark_codes:
                line.remark_codes = _gather_remark_codes(line_rec)
            if not line.adjustments:
                line.adjustments = _build_adjustments(line_rec)
            out.append(line)
        return out

    has_line_fields = any(k in record for k in ("pcl_ProcedureCode", "pcl_ChargedAmount"))
    if not has_line_fields:
        return []

    line = RemittanceLine.model_validate(record)
    line.modifiers = _gather_modifiers(record)
    line.remark_codes = _gather_remark_codes(record)
    line.adjustments = _build_adjustments(record)
    return [line]


def remittance_from_dict(record: dict[str, Any]) -> Remittance:
    rem = Remittance.model_validate(record)
    rem.lines = _build_lines(record)
    return rem


def submission_from_dict(record: dict[str, Any]) -> ClaimSubmission:
    sub = ClaimSubmission.model_validate(record)
    additional: list[str] = []
    for i in range(2, 26):
        v = record.get(f"ec_Diag{i}")
        if v:
            additional.append(str(v))
    sub.additional_diagnoses = additional
    return sub


def join_claim(remittance_record: dict[str, Any], submission_record: dict[str, Any]) -> Claim:
    rem = remittance_from_dict(remittance_record)
    sub = submission_from_dict(submission_record)
    if rem.claim_id != sub.claim_no:
        raise ValueError(
            f"Claim ID mismatch: 835 pc_ClaimID={rem.claim_id!r} != 837 ec_ClaimNo={sub.claim_no!r}"
        )
    return Claim(claim_id=rem.claim_id, remittance=rem, submission=sub)


def load_claims_jsonl(path: str | Path) -> list[Claim]:
    """Load a JSONL file where each line is `{"835": {...}, "837": {...}, ...}`."""
    path = Path(path)
    out: list[Claim] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rem_rec = rec.get("835") or rec.get("remittance") or {}
        sub_rec = rec.get("837") or rec.get("submission") or {}
        if not rem_rec or not sub_rec:
            continue
        out.append(join_claim(rem_rec, sub_rec))
    return out


def load_gold_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """Pull the inline gold labels from the synthetic dataset."""
    path = Path(path)
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        gold = rec.get("gold")
        if not gold:
            continue
        cid = (rec.get("835") or {}).get("pc_ClaimID")
        if cid:
            out[cid] = gold
    return out
