"""Deterministic evidence extractors.

Each `check_*` function takes a `Claim` and returns a list of `EvidenceItem`s.
The functions are intentionally pure and side-effect-free so they're easy
to unit-test and so the LLM reasoner can rely on them as ground truth.
"""

from __future__ import annotations

from collections.abc import Callable

from ..schemas import Claim, EvidenceItem
from .coding import check_coding
from .duplicate import check_duplicate
from .medical_necessity import check_medical_necessity
from .missing_info import check_missing_info
from .prior_auth import check_prior_auth
from .timely_filing import check_timely_filing

EXTRACTORS: dict[str, Callable[[Claim], list[EvidenceItem]]] = {
    "timely_filing": check_timely_filing,
    "prior_auth": check_prior_auth,
    "coding": check_coding,
    "medical_necessity": check_medical_necessity,
    "duplicate": check_duplicate,
    "missing_info": check_missing_info,
}


def run_all(claim: Claim) -> list[EvidenceItem]:
    """Run every extractor against the claim. Order is stable for reproducibility."""
    items: list[EvidenceItem] = []
    for name in (
        "timely_filing",
        "prior_auth",
        "coding",
        "medical_necessity",
        "duplicate",
        "missing_info",
    ):
        items.extend(EXTRACTORS[name](claim))
    return items


__all__ = [
    "EXTRACTORS",
    "run_all",
    "check_coding",
    "check_duplicate",
    "check_medical_necessity",
    "check_missing_info",
    "check_prior_auth",
    "check_timely_filing",
]
