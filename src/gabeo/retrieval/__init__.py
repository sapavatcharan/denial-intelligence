"""Hybrid retrieval: structured filters + TF-IDF cosine similarity."""

from .similar_claims import (
    PatternStats,
    SimilarClaim,
    SimilarityIndex,
    aggregate_payer_procedure_carc_stats,
    claim_signature,
)

__all__ = [
    "PatternStats",
    "SimilarClaim",
    "SimilarityIndex",
    "aggregate_payer_procedure_carc_stats",
    "claim_signature",
]
