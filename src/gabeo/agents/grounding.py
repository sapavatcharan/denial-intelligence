"""Grounding gate: verify every cited field exists on the input claim.

This is the single most important guardrail in the system. The LLM is allowed
to *narrate* - it is not allowed to *fabricate field names or values*. The gate
walks the verdict's `supporting_evidence`, looks up each `field_name` on the
claim's raw input, and compares the value.

If any citation fails, we re-prompt once with a corrective hint listing the
violations. If it still fails, we mark the verdict as `needs_review` and
attach the violations so a human can audit.
"""

from __future__ import annotations

from typing import Any, Callable

from ..schemas import Claim, SupportingEvidenceCitation


def _line_modifier(claim: Claim, idx: int) -> str | None:
    if not claim.remittance or not claim.remittance.lines:
        return None
    mods = claim.remittance.lines[0].modifiers
    return mods[idx] if idx < len(mods) else None


def _first_adj(claim: Claim, attr: str) -> Any:
    if not claim.remittance:
        return None
    for line in claim.remittance.lines:
        for adj in line.adjustments:
            v = getattr(adj, attr, None)
            return v.value if hasattr(v, "value") else v
    return None


def _diag(claim: Claim, idx: int) -> str | None:
    if not claim.submission:
        return None
    extras = claim.submission.additional_diagnoses
    return extras[idx - 1] if 0 < idx <= len(extras) else None


_FIELD_ACCESSORS: dict[str, Callable[[Claim], Any]] = {
    "pc_ClaimID":              lambda c: c.remittance.claim_id if c.remittance else None,
    "pc_ClaimStatus":          lambda c: c.remittance.claim_status.value if c.remittance and c.remittance.claim_status else None,
    "pc_ClaimAmount":          lambda c: c.remittance.claim_amount if c.remittance else None,
    "pc_ClaimPaid":             lambda c: c.remittance.claim_paid if c.remittance else None,
    "pc_InsuranceType":        lambda c: c.remittance.insurance_type if c.remittance else None,
    "pc_ReceivedDate":         lambda c: str(c.remittance.received_date) if c.remittance and c.remittance.received_date else None,
    "pc_StatementBegin":       lambda c: str(c.remittance.statement_begin) if c.remittance and c.remittance.statement_begin else None,
    "pc_StatementEnd":         lambda c: str(c.remittance.statement_end) if c.remittance and c.remittance.statement_end else None,
    "pc_PriorAuthNum":         lambda c: c.remittance.prior_auth_num if c.remittance else None,
    "cp_PayerID":              lambda c: c.remittance.payer_id if c.remittance else None,
    "cp_PayerName":            lambda c: c.remittance.payer_name if c.remittance else None,
    "pcl_ProcedureCode":       lambda c: c.primary_procedure,
    "pcl_ProcedureModifier1":  lambda c: _line_modifier(c, 0),
    "pcl_ProcedureModifier2":  lambda c: _line_modifier(c, 1),
    "pcl_ProcedureModifier3":  lambda c: _line_modifier(c, 2),
    "pcl_ProcedureModifier4":  lambda c: _line_modifier(c, 3),
    "pcl_ChargedAmount":       lambda c: c.remittance.lines[0].charged_amount if c.remittance and c.remittance.lines else None,
    "pcl_PaidAmount":          lambda c: c.remittance.lines[0].paid_amount if c.remittance and c.remittance.lines else None,
    "pcl_AllowedAmount":       lambda c: c.remittance.lines[0].allowed_amount if c.remittance and c.remittance.lines else None,
    "pcl_RemarkCodes":         lambda c: ",".join(c.remittance.lines[0].remark_codes) if c.remittance and c.remittance.lines else None,
    "pcla_AdjustmentGroup":    lambda c: _first_adj(c, "group"),
    "pcla_AdjustmentReason":   lambda c: c.primary_carc,
    "pcla_AdjustmentAmount":   lambda c: _first_adj(c, "amount"),
    "ec_ClaimNo":              lambda c: c.submission.claim_no if c.submission else None,
    "ec_PayerName":            lambda c: c.submission.payer_name if c.submission else None,
    "ec_PayerID":              lambda c: c.submission.payer_id if c.submission else None,
    "ec_InsuranceType":        lambda c: c.submission.insurance_type if c.submission else None,
    "ec_PrincipalDiagnosis":   lambda c: c.submission.principal_diagnosis if c.submission else None,
    "ec_Diag2":                lambda c: _diag(c, 1),
    "ec_Diag3":                lambda c: _diag(c, 2),
    "ec_Diag4":                lambda c: _diag(c, 3),
    "ec_BillProvNPI":          lambda c: c.submission.bill_provider_npi if c.submission else None,
    "ec_RendProvNPI":          lambda c: c.submission.rendering_provider_npi if c.submission else None,
    "ec_RendProvSpecialty":    lambda c: c.submission.rendering_provider_specialty if c.submission else None,
    "ec_ServiceDateFrom":      lambda c: str(c.submission.service_date_from) if c.submission and c.submission.service_date_from else None,
    "ec_ServiceDateTo":        lambda c: str(c.submission.service_date_to) if c.submission and c.submission.service_date_to else None,
    "ec_PriorAuthorization":   lambda c: c.submission.prior_authorization if c.submission else None,
    "ec_TypeOfBill":           lambda c: c.submission.type_of_bill if c.submission else None,
    "ec_ClaimFrequency":       lambda c: c.submission.claim_frequency if c.submission else None,
    "ec_DelayReasonCode":      lambda c: c.submission.delay_reason_code if c.submission else None,
    "ec_SubscriberID":         lambda c: c.submission.subscriber_id if c.submission else None,
    "ec_OtherPayerName":       lambda c: c.submission.other_payer_name if c.submission else None,
    "ec_OtherPayerPaidDate":   lambda c: str(c.submission.other_payer_paid_date) if c.submission and c.submission.other_payer_paid_date else None,
}


def field_value(claim: Claim, field_name: str) -> Any:
    accessor = _FIELD_ACCESSORS.get(field_name)
    if accessor is None:
        return None
    return accessor(claim)


def known_fields() -> list[str]:
    return sorted(_FIELD_ACCESSORS.keys())


def validate_citations(
    claim: Claim, citations: list[SupportingEvidenceCitation]
) -> list[str]:
    """Return a list of human-readable violations. Empty list means OK."""
    violations: list[str] = []
    for cit in citations:
        if cit.field_name not in _FIELD_ACCESSORS:
            violations.append(
                f"field_name '{cit.field_name}' is not a recognized 835/837 column."
            )
            continue
        actual = field_value(claim, cit.field_name)
        if actual is None and cit.field_value not in ("", "None", "null"):
            violations.append(
                f"field_name '{cit.field_name}' is not present on this claim, but the model "
                f"asserted value '{cit.field_value}'."
            )
            continue
        if actual is not None and str(actual) != str(cit.field_value):
            violations.append(
                f"field_name '{cit.field_name}': model asserted '{cit.field_value}' but "
                f"actual value is '{actual}'."
            )
    return violations
