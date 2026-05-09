"""Root-cause agent: orchestrates evidence + retrieval + LLM + grounding gate.

Pipeline per claim:

  1. Run all deterministic evidence extractors.
  2. Pull historical context (top-K similar paid claims) - optional.
  3. Build a compact, grounded user prompt.
  4. Call the LLM (router decides cheap vs strong).
  5. Validate the verdict's `supporting_evidence` against the grounding gate.
  6. If violations exist, re-prompt once with a corrective hint.
  7. Calibrate confidence using the deterministic evidence and return.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..evidence import run_all
from ..llm.client import LLMClient
from ..llm.mock_client import MockLLMClient
from ..llm.router import select_model
from ..reference import carc, rarc
from ..schemas import (
    Claim,
    EvidenceItem,
    Recoverability,
    SupportingEvidenceCitation,
    Verdict,
)
from .grounding import field_value, known_fields, validate_citations

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


class _LLMVerdict(BaseModel):
    """The narrow shape we ask the LLM to return."""

    root_cause: str
    carc_interpretation: str
    recoverability: Recoverability
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    supporting_evidence: list[SupportingEvidenceCitation]


def _format_evidence(items: list[EvidenceItem]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(
            f"- [{item.severity.upper()}] {item.check_name} "
            f"(passed={item.passed}): {item.message} "
            f"(fields_referenced={item.fields_referenced})"
        )
    return "\n".join(lines) if lines else "(no deterministic findings)"


def _format_claim_table(claim: Claim) -> str:
    rows: list[str] = []
    for fname in known_fields():
        v = field_value(claim, fname)
        if v is None or v == "":
            continue
        rows.append(f"{fname} = {v}")
    return "\n".join(rows)


def _carc_context(claim: Claim) -> str:
    code = claim.primary_carc
    if not code:
        return "(no CARC code found on the claim)"
    info = carc(code)
    if not info:
        return f"CARC {code} (unknown to local catalog)"
    return (
        f"CARC {code} - family: {info['family']}; description: {info['description']}; "
        f"appealable: {info['generally_appealable']}; common resolution: {info['common_resolution']}"
    )


def _rarc_context(claim: Claim) -> str:
    if not claim.remittance:
        return ""
    codes: list[str] = []
    for line in claim.remittance.lines:
        for c in line.remark_codes:
            ref = rarc(c)
            if ref:
                codes.append(f"RARC {c}: {ref['description']}")
    return "\n".join(codes)


def _build_user_prompt(
    claim: Claim,
    evidence: list[EvidenceItem],
    similar_paid_claim_ids: list[str],
    historical_summary: str,
    correction_hint: str | None = None,
) -> str:
    parts = [
        "<claim_id>" + claim.claim_id + "</claim_id>",
        "<carc_context>\n" + _carc_context(claim) + "\n</carc_context>",
    ]
    rarc_text = _rarc_context(claim)
    if rarc_text:
        parts.append("<rarc_context>\n" + rarc_text + "\n</rarc_context>")
    parts.append("<claim_data>\n" + _format_claim_table(claim) + "\n</claim_data>")
    parts.append("<deterministic_evidence>\n" + _format_evidence(evidence) + "\n</deterministic_evidence>")
    if similar_paid_claim_ids or historical_summary:
        parts.append(
            "<historical_context>\n"
            f"similar_paid_claim_ids: {similar_paid_claim_ids}\n"
            f"summary: {historical_summary}\n"
            "</historical_context>"
        )
    parts.append(
        "<allowed_field_names>\n"
        + ", ".join(known_fields())
        + "\n</allowed_field_names>"
    )
    if correction_hint:
        parts.append(
            "<correction_required>\n"
            "Your previous response cited fields that did not match the claim. "
            "Fix every violation below and resubmit. Cite ONLY field names from "
            "<allowed_field_names>.\n"
            + correction_hint
            + "\n</correction_required>"
        )
    parts.append(
        "Return a single JSON object matching the schema described in the system prompt."
    )
    return "\n\n".join(parts)


def _calibrate_confidence(
    llm_confidence: float, evidence: list[EvidenceItem]
) -> tuple[float, dict[str, float]]:
    if not evidence:
        return llm_confidence, {"llm": llm_confidence, "evidence": 0.0, "blended": llm_confidence}
    fraction_passed = sum(1 for e in evidence if e.passed) / len(evidence)
    blended = round(0.5 * llm_confidence + 0.5 * fraction_passed, 3)
    return blended, {
        "llm": round(llm_confidence, 3),
        "evidence_pass_rate": round(fraction_passed, 3),
        "blended": blended,
    }


def _build_default_client() -> LLMClient | MockLLMClient:
    """Resolve the LLM client.

      1. `GABEO_MOCK_LLM=1` -> always mock.
      2. No `OPENAI_API_KEY` -> mock with a warning.
      3. Otherwise the real client; runtime quota errors auto-fallback to mock.
    """
    if os.environ.get("GABEO_MOCK_LLM") == "1":
        return MockLLMClient()
    if not os.environ.get("OPENAI_API_KEY"):
        return MockLLMClient()
    return LLMClient()


class RootCauseAgent:
    def __init__(self, llm: LLMClient | MockLLMClient | None = None) -> None:
        self._llm = llm or _build_default_client()
        self._system_prompt = (PROMPTS_DIR / "root_cause_analysis.md").read_text()

    def analyze(
        self,
        claim: Claim,
        *,
        similar_paid_claim_ids: list[str] | None = None,
        historical_summary: str = "",
    ) -> Verdict:
        evidence = run_all(claim)
        model = select_model(evidence)
        similar_ids = similar_paid_claim_ids or []

        user = _build_user_prompt(claim, evidence, similar_ids, historical_summary)
        try:
            response = self._llm.chat(
                model=model,
                system=self._system_prompt,
                user=user,
                response_model=_LLMVerdict,
            )
        except Exception as exc:
            err = repr(exc).lower()
            if isinstance(self._llm, LLMClient) and (
                "ratelimit" in err
                or "insufficient_quota" in err
                or "429" in err
                or "authentication" in err
            ):
                self._llm = MockLLMClient()
                response = self._llm.chat(
                    model=model,
                    system=self._system_prompt,
                    user=user,
                    response_model=_LLMVerdict,
                )
            else:
                raise
        verdict_raw: _LLMVerdict = response.parsed  # type: ignore[assignment]

        violations = validate_citations(claim, verdict_raw.supporting_evidence)
        if violations:
            hint = "\n".join(f"- {v}" for v in violations)
            user2 = _build_user_prompt(
                claim, evidence, similar_ids, historical_summary, correction_hint=hint
            )
            response = self._llm.chat(
                model=model,
                system=self._system_prompt,
                user=user2,
                response_model=_LLMVerdict,
            )
            verdict_raw = response.parsed  # type: ignore[assignment]
            violations = validate_citations(claim, verdict_raw.supporting_evidence)

        recoverability = verdict_raw.recoverability
        if violations:
            recoverability = Recoverability.NEEDS_REVIEW

        blended, components = _calibrate_confidence(verdict_raw.confidence, evidence)
        return Verdict(
            claim_id=claim.claim_id,
            root_cause=verdict_raw.root_cause,
            carc_interpretation=verdict_raw.carc_interpretation,
            recoverability=recoverability,
            confidence=blended,
            confidence_components=components,
            recommended_action=verdict_raw.recommended_action,
            supporting_evidence=verdict_raw.supporting_evidence,
            deterministic_evidence=evidence,
            similar_paid_claims=similar_ids,
            model_used=response.model,
            cost_usd=round(response.cost_usd, 6),
            latency_ms=response.latency_ms,
        )


def verdict_to_jsonable(v: Verdict) -> dict[str, Any]:
    return json.loads(v.model_dump_json())
