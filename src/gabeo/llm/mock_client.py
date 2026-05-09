"""A deterministic mock LLM client.

Two reasons this exists:

  1. CI / local dev: lets the entire pipeline run without an API key, which
     is the difference between a system you can prove works and one you
     can only describe.
  2. Reliability fallback: in production, if the OpenAI API is down or
     rate-limited, we can fall back to this client so the billing team
     still gets a (less narrative but fully grounded) verdict.

The mock returns a `Verdict`-shaped JSON object whose `supporting_evidence`
is built directly from the deterministic evidence layer's `fields_referenced`.
Because of that, every citation always passes the grounding gate.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from .client import LLMResponse


def _select_recoverability(
    evidence: list[dict[str, Any]], carc_family: str | None
) -> str:
    has_secondary_anchor = any(
        e["check_name"] == "timely_filing.secondary_anchor" and e["passed"] for e in evidence
    )
    has_dx_repointing = any(
        e["check_name"] == "medical_necessity.secondary_supports" and e["passed"]
        for e in evidence
    )
    has_principal_supports = any(
        e["check_name"] == "medical_necessity.principal_supports" and e["passed"]
        for e in evidence
    )
    has_bilateral_modifier_missing = any(
        "bilateral_modifier_missing" in e["check_name"] and not e["passed"] for e in evidence
    )
    has_pa_required_and_present = any(
        e["check_name"] == "prior_auth.required_and_present" and e["passed"] for e in evidence
    )
    has_delay_reason = any(
        e["check_name"] == "timely_filing.delay_reason_present" for e in evidence
    )
    timely_window = next(
        (e for e in evidence if e["check_name"] == "timely_filing.window_check"), None
    )

    if carc_family == "timely_filing":
        if has_secondary_anchor and timely_window and timely_window["passed"]:
            return "recoverable"
        if has_delay_reason:
            return "needs_review"
        if timely_window and not timely_window["passed"]:
            return "not_recoverable"
        return "needs_review"

    if carc_family == "medical_necessity":
        if has_dx_repointing or has_principal_supports:
            return "recoverable"
        return "needs_review"

    if carc_family == "duplicate":
        if has_bilateral_modifier_missing:
            return "recoverable"
        return "needs_review"

    if carc_family == "prior_auth":
        if has_pa_required_and_present:
            return "recoverable"
        return "needs_review"

    if carc_family == "coding":
        return "recoverable"

    if carc_family == "missing_info":
        return "recoverable"

    if carc_family in {"contractual", "eligibility", "non_covered"}:
        return "not_recoverable"

    return "needs_review"


def _build_action(recoverability: str, carc_family: str | None) -> str:
    if recoverability == "not_recoverable":
        return "Write off and document the denial reason in the contractual-adjustment log."
    if carc_family == "medical_necessity":
        return "Repoint the diagnosis pointer to the supportive secondary diagnosis and resubmit; if denied again, appeal with clinical documentation."
    if carc_family == "prior_auth":
        return "Resubmit with the prior authorization number on file; if no auth exists, request a retro-authorization with clinical justification."
    if carc_family == "duplicate":
        return "Verify the line is not a true duplicate; if it represents a paired-organ or repeat service, append the appropriate modifier (LT/RT/50/59) and resubmit."
    if carc_family == "coding":
        return "Apply the missing required modifier and resubmit the corrected claim."
    if carc_family == "missing_info":
        return "Populate the missing field(s) called out by the RARC code and resubmit a corrected claim."
    if carc_family == "timely_filing":
        return "Submit a timely-filing appeal with documentation supporting the delay reason."
    return "Send to a billing analyst for case-by-case review."


def _build_root_cause(evidence: list[dict[str, Any]], carc: str) -> str:
    parts: list[str] = [f"Denial driven by CARC {carc}."]
    for e in evidence:
        if e["severity"] in {"critical", "warning"}:
            parts.append(e["message"])
    return " ".join(parts)[:1200]


def _supporting_evidence(
    evidence: list[dict[str, Any]], claim_summary: dict[str, Any]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    ordered = sorted(
        evidence,
        key=lambda e: (
            0 if e["severity"] == "critical" else 1 if e["severity"] == "warning" else 2,
            0 if not e["passed"] else 1,
        ),
    )
    for ev in ordered:
        for fname in ev["fields_referenced"]:
            if fname in seen:
                continue
            value = claim_summary.get(fname)
            if value is None:
                continue
            seen.add(fname)
            out.append(
                {
                    "field_name": fname,
                    "field_value": str(value),
                    "why_relevant": ev["message"][:240],
                }
            )
            if len(out) >= 5:
                break
        if len(out) >= 5:
            break
    return out


def _between(text: str, start: str, end: str) -> str | None:
    if start not in text or end not in text:
        return None
    return text.split(start, 1)[1].split(end, 1)[0]


def _parse_user_payload(user: str) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    evidence: list[dict[str, Any]] = []
    claim_summary: dict[str, Any] = {}
    carc: str | None = None

    carc_block = _between(user, "<carc_context>", "</carc_context>")
    if carc_block:
        for tok in carc_block.split():
            if tok.isdigit() and len(tok) <= 3 and not carc:
                carc = tok
                break

    claim_block = _between(user, "<claim_data>", "</claim_data>") or ""
    for line in claim_block.splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            claim_summary[k.strip()] = v.strip()

    ev_block = _between(user, "<deterministic_evidence>", "</deterministic_evidence>") or ""
    for line in ev_block.splitlines():
        line = line.strip()
        if not line.startswith("- ["):
            continue
        try:
            severity = line[3 : line.index("]")].strip().lower()
            after_sev = line[line.index("]") + 1 :].strip()
            check_name = after_sev.split(" (passed=", 1)[0].strip()
            passed_str = after_sev.split("(passed=", 1)[1].split(")", 1)[0].strip()
            passed = passed_str.lower() == "true"
            message = after_sev.split("): ", 1)[1] if "): " in after_sev else ""
            fields = []
            if "fields_referenced=[" in line:
                fr = line.split("fields_referenced=[", 1)[1].split("]", 1)[0]
                fields = [f.strip().strip("'\"") for f in fr.split(",") if f.strip()]
            evidence.append(
                {
                    "check_name": check_name,
                    "severity": severity,
                    "passed": passed,
                    "message": message,
                    "fields_referenced": fields,
                }
            )
        except (ValueError, IndexError):
            continue
    return evidence, claim_summary, carc


_CARC_FAMILY = {
    "29": "timely_filing",
    "16": "missing_info",
    "252": "missing_info",
    "50": "medical_necessity",
    "11": "medical_necessity",
    "167": "medical_necessity",
    "18": "duplicate",
    "97": "duplicate",
    "197": "prior_auth",
    "198": "prior_auth",
    "4": "coding",
    "45": "contractual",
    "27": "eligibility",
    "96": "non_covered",
    "204": "non_covered",
}


class MockLLMClient:
    """Drop-in replacement for `LLMClient` that produces grounded, deterministic verdicts."""

    def __init__(self, *, jitter_ms: int = 5) -> None:
        self._jitter_ms = jitter_ms

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        start = time.perf_counter()
        evidence, claim_summary, carc = _parse_user_payload(user)
        family = _CARC_FAMILY.get(carc or "", None)
        recoverability = _select_recoverability(evidence, family)

        verdict = {
            "root_cause": _build_root_cause(evidence, carc or "?"),
            "carc_interpretation": (
                f"CARC {carc} ({family}): the payer adjudicated this charge under the {family} family of denials."
                if family
                else f"CARC {carc}: standard adjustment reason."
            ),
            "recoverability": recoverability,
            "confidence": 0.7,
            "recommended_action": _build_action(recoverability, family),
            "supporting_evidence": _supporting_evidence(evidence, claim_summary),
        }
        text = json.dumps(verdict)
        parsed: BaseModel | None = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate(verdict)
            except ValidationError as exc:
                raise ValueError(f"Mock verdict failed validation: {exc}") from exc

        time.sleep(self._jitter_ms / 1000.0)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=f"mock:{model}",
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
