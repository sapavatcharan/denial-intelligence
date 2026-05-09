"""Cheap-first model router.

Triage every claim with the small model. Escalate to the strong model only
when the deterministic evidence layer suggests the call is hard:

  * Many warning-level evidence items (the case has nuance).
  * Conflicting evidence (some checks pass, some fail in the same family).
  * Critical-but-recoverable signals (e.g., bilateral modifier missing).
"""

from __future__ import annotations

import os

from ..schemas import EvidenceItem


def select_model(evidence: list[EvidenceItem]) -> str:
    triage = os.environ.get("GABEO_LLM_TRIAGE_MODEL", "gpt-4o-mini")
    strong = os.environ.get("GABEO_LLM_STRONG_MODEL", "gpt-4o")

    warnings = sum(1 for e in evidence if e.severity == "warning")
    criticals = sum(1 for e in evidence if e.severity == "critical")
    failed = sum(1 for e in evidence if not e.passed)
    has_secondary_anchor = any(e.check_name == "timely_filing.secondary_anchor" for e in evidence)
    has_dx_repointing = any(
        e.check_name == "medical_necessity.secondary_supports" for e in evidence
    )

    hard = (
        warnings >= 2
        or (criticals >= 1 and failed >= 2)
        or has_secondary_anchor
        or has_dx_repointing
    )
    return strong if hard else triage
