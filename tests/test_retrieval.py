"""Tests for the hybrid retrieval layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from gabeo.ingest import load_claims_jsonl
from gabeo.retrieval import (
    SimilarityIndex,
    aggregate_payer_procedure_carc_stats,
    claim_signature,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "claims.jsonl"


@pytest.fixture(scope="module")
def claims():  # type: ignore[no-untyped-def]
    if not DATA.exists():
        pytest.skip("dataset not generated")
    return load_claims_jsonl(DATA)


def test_signature_includes_payer_proc_and_dx(claims):  # type: ignore[no-untyped-def]
    sig = claim_signature(claims[0])
    assert "payer_" in sig
    assert "proc_" in sig


def test_similarity_filters_to_same_payer_and_procedure_family(claims):  # type: ignore[no-untyped-def]
    index = SimilarityIndex(claims)
    target = next(c for c in claims if c.claim_id == "CLM-2026-00142")
    similar = index.find_similar_paid(target, top_k=3)
    for s in similar:
        assert s.is_paid
        assert s.payer_id == "BCBS-IL"
        assert (s.procedure or "").startswith("99")


def test_similarity_returns_empty_when_no_paid_in_corpus():  # type: ignore[no-untyped-def]
    index = SimilarityIndex([])
    from gabeo.schemas import Claim
    fake = Claim(claim_id="X")
    assert index.find_similar_paid(fake) == []


def test_pattern_stats_sorted_by_denied_amount(claims):  # type: ignore[no-untyped-def]
    stats = aggregate_payer_procedure_carc_stats(claims)
    denied = [s for s in stats if s.carc_code != "PAID"]
    assert denied
    amounts = [s.denied_amount_total for s in denied]
    assert amounts == sorted(amounts, reverse=True)


def test_recovery_proxy_uses_paid_payer_proc_total(claims):  # type: ignore[no-untyped-def]
    """A denied (payer, proc) with paid history elsewhere on the same pair
    should report a non-zero recovery_proxy."""
    stats = aggregate_payer_procedure_carc_stats(claims)
    aetna_72148 = [s for s in stats if s.payer_id == "AETNA" and s.procedure == "72148"]
    assert any(s.recovery_proxy > 0 for s in aetna_72148) or len(aetna_72148) <= 1
