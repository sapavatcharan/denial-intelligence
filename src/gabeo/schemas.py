"""Pydantic models for EDI 835/837 inputs, joined claims, and analysis outputs.

The 835/837 standards have hundreds of fields each. We model only the subset
that drives denial reasoning today, accepting and ignoring the rest. This
keeps the schema readable while staying robust to unfamiliar incoming columns.

The `Verdict` model is the contract for the root-cause agent's structured
output. It is intentionally narrow: every supporting evidence item must cite
a real field name, which the grounding gate enforces post-LLM.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimStatus(str, Enum):
    PROCESSED_PRIMARY = "1"
    PROCESSED_SECONDARY = "2"
    PROCESSED_TERTIARY = "3"
    DENIED = "4"
    PROCESSED_PRIMARY_FORWARDED = "19"
    REVERSAL = "22"


class AdjustmentGroup(str, Enum):
    CONTRACTUAL = "CO"
    PATIENT_RESPONSIBILITY = "PR"
    OTHER = "OA"
    PAYER_INITIATED = "PI"
    CORRECTION = "CR"


class Recoverability(str, Enum):
    RECOVERABLE = "recoverable"
    NOT_RECOVERABLE = "not_recoverable"
    NEEDS_REVIEW = "needs_review"


class LineAdjustment(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    group: AdjustmentGroup | None = Field(default=None, alias="pcla_AdjustmentGroup")
    reason_code: str | None = Field(default=None, alias="pcla_AdjustmentReason")
    amount: float = Field(default=0.0, alias="pcla_AdjustmentAmount")
    quantity: float | None = Field(default=None, alias="pcla_AdjustmentQty")


class RemittanceLine(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    procedure_code: str | None = Field(default=None, alias="pcl_ProcedureCode")
    modifiers: list[str] = Field(default_factory=list)
    charged_amount: float = Field(default=0.0, alias="pcl_ChargedAmount")
    paid_amount: float = Field(default=0.0, alias="pcl_PaidAmount")
    allowed_amount: float | None = Field(default=None, alias="pcl_AllowedAmount")
    units_paid: float | None = Field(default=None, alias="pcl_UnitsPaid")
    service_date: date | None = Field(default=None, alias="pcl_ServiceDate")
    remark_codes: list[str] = Field(default_factory=list)
    adjustments: list[LineAdjustment] = Field(default_factory=list)

    @field_validator("modifiers", mode="before")
    @classmethod
    def _split_modifiers(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        if isinstance(v, list):
            return [str(m).strip() for m in v if m and str(m).strip()]
        return []

    @field_validator("remark_codes", mode="before")
    @classmethod
    def _split_remarks(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [r.strip() for r in v.split(",") if r.strip()]
        if isinstance(v, list):
            return [str(r).strip() for r in v if r and str(r).strip()]
        return []


class Remittance(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    claim_id: str = Field(alias="pc_ClaimID")
    claim_status: ClaimStatus | None = Field(default=None, alias="pc_ClaimStatus")
    claim_amount: float = Field(default=0.0, alias="pc_ClaimAmount")
    claim_paid: float = Field(default=0.0, alias="pc_ClaimPaid")
    patient_responsibility: float = Field(default=0.0, alias="pc_PatientResponsibility")
    insurance_type: str | None = Field(default=None, alias="pc_InsuranceType")
    received_date: date | None = Field(default=None, alias="pc_ReceivedDate")
    statement_begin: date | None = Field(default=None, alias="pc_StatementBegin")
    statement_end: date | None = Field(default=None, alias="pc_StatementEnd")
    prior_auth_num: str | None = Field(default=None, alias="pc_PriorAuthNum")
    payer_id: str | None = Field(default=None, alias="cp_PayerID")
    payer_name: str | None = Field(default=None, alias="cp_PayerName")
    lines: list[RemittanceLine] = Field(default_factory=list)


class ClaimSubmission(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    claim_no: str = Field(alias="ec_ClaimNo")
    amount: float = Field(default=0.0, alias="ec_Amount")
    payer_name: str | None = Field(default=None, alias="ec_PayerName")
    payer_id: str | None = Field(default=None, alias="ec_PayerID")
    insurance_type: str | None = Field(default=None, alias="ec_InsuranceType")
    place_of_service: str | None = Field(default=None, alias="ec_PlaceOfService")
    principal_diagnosis: str | None = Field(default=None, alias="ec_PrincipalDiagnosis")
    additional_diagnoses: list[str] = Field(default_factory=list)
    bill_provider_npi: str | None = Field(default=None, alias="ec_BillProvNPI")
    rendering_provider_npi: str | None = Field(default=None, alias="ec_RendProvNPI")
    rendering_provider_specialty: str | None = Field(default=None, alias="ec_RendProvSpecialty")
    service_date_from: date | None = Field(default=None, alias="ec_ServiceDateFrom")
    service_date_to: date | None = Field(default=None, alias="ec_ServiceDateTo")
    prior_authorization: str | None = Field(default=None, alias="ec_PriorAuthorization")
    type_of_bill: str | None = Field(default=None, alias="ec_TypeOfBill")
    claim_frequency: str | None = Field(default=None, alias="ec_ClaimFrequency")
    delay_reason_code: str | None = Field(default=None, alias="ec_DelayReasonCode")
    patient_relationship: str | None = Field(default=None, alias="ec_PatientRelationship")
    subscriber_id: str | None = Field(default=None, alias="ec_SubscriberID")
    other_payer_name: str | None = Field(default=None, alias="ec_OtherPayerName")
    other_payer_paid: float | None = Field(default=None, alias="ec_OtherPayerPaid")
    other_payer_paid_date: date | None = Field(default=None, alias="ec_OtherPayerPaidDate")
    service_lines: list[dict[str, Any]] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_id: str
    remittance: Remittance | None = None
    submission: ClaimSubmission | None = None

    @property
    def is_denied(self) -> bool:
        if self.remittance and self.remittance.claim_status == ClaimStatus.DENIED:
            return True
        if self.remittance and self.remittance.claim_paid == 0 and self.remittance.claim_amount > 0:
            return True
        return False

    @property
    def primary_carc(self) -> str | None:
        if not self.remittance:
            return None
        best: tuple[float, str | None] = (-1.0, None)
        for line in self.remittance.lines:
            for adj in line.adjustments:
                if adj.reason_code and adj.amount > best[0]:
                    best = (adj.amount, adj.reason_code)
        return best[1]

    @property
    def primary_procedure(self) -> str | None:
        if self.remittance and self.remittance.lines:
            return self.remittance.lines[0].procedure_code
        return None

    @property
    def total_denied_amount(self) -> float:
        if not self.remittance:
            return 0.0
        return sum(
            adj.amount
            for line in self.remittance.lines
            for adj in line.adjustments
            if adj.group == AdjustmentGroup.CONTRACTUAL
        )

    @property
    def all_diagnoses(self) -> list[str]:
        if not self.submission:
            return []
        out: list[str] = []
        if self.submission.principal_diagnosis:
            out.append(self.submission.principal_diagnosis)
        out.extend(self.submission.additional_diagnoses)
        return out


class EvidenceItem(BaseModel):
    """A single piece of deterministic evidence about a claim.

    The `fields_referenced` list is the contract with the grounding gate:
    every field listed must exist as a real input field on the claim.
    """

    check_name: str
    passed: bool
    severity: str = Field(default="info", description="info | warning | critical")
    message: str
    observed_value: str | int | float | bool | None = None
    expected_value: str | int | float | bool | None = None
    fields_referenced: list[str] = Field(default_factory=list)


class SupportingEvidenceCitation(BaseModel):
    field_name: str
    field_value: str
    why_relevant: str


class Verdict(BaseModel):
    claim_id: str
    root_cause: str
    carc_interpretation: str
    recoverability: Recoverability
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_components: dict[str, float] = Field(default_factory=dict)
    recommended_action: str
    supporting_evidence: list[SupportingEvidenceCitation]
    deterministic_evidence: list[EvidenceItem] = Field(default_factory=list)
    similar_paid_claims: list[str] = Field(default_factory=list)
    model_used: str | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
