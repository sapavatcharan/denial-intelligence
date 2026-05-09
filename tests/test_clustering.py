"""Tests for the clustering layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from gabeo.clustering import build_batch_brief, cluster_denials
from gabeo.clustering.batch_intelligence import render_brief_markdown
from gabeo.ingest import load_claims_jsonl

DATA = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "claims.jsonl"


@pytest.fixture(scope="module")
def claims():  # type: ignore[no-untyped-def]
    if not DATA.exists():
        pytest.skip("dataset not generated")
    return load_claims_jsonl(DATA)


def test_clusters_have_only_denied_claims(claims):  # type: ignore[no-untyped-def]
    clusters = cluster_denials(claims)
    by_id = {c.claim_id: c for c in claims}
    for cluster in clusters:
        for cid in cluster.claim_ids:
            assert by_id[cid].is_denied


def test_clusters_sorted_by_dollars(claims):  # type: ignore[no-untyped-def]
    clusters = cluster_denials(claims)
    amounts = [c.total_denied_amount for c in clusters]
    assert amounts == sorted(amounts, reverse=True)


def test_brief_includes_action_text(claims):  # type: ignore[no-untyped-def]
    clusters = cluster_denials(claims)
    clusters = build_batch_brief(clusters)
    md = render_brief_markdown(clusters)
    assert "Denial Batch Brief" in md
    assert "$" in md
    assert "claims" in md
