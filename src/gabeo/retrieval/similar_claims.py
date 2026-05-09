"""Hybrid retrieval over a corpus of joined Claims.

Two-stage:

  1. Hard structured filter: payer (exact) AND insurance_type (exact)
     AND CPT family (first 3 chars). This mirrors how a billing analyst
     actually defines similarity - payer / insurance / procedure family
     are not negotiable.

  2. TF-IDF cosine similarity on a "claim signature" string. The signature
     captures diagnosis prefix, place of service, provider specialty,
     amount bucket, and modifiers. This handles the long-tail soft signals.

Why TF-IDF instead of dense embeddings:
  * Free, deterministic, reproducible (eval reruns are stable).
  * For 10^4-10^5 claims and a few-token signature, TF-IDF cosine matches
    or beats sentence-transformers in head-to-head IR benchmarks on
    short structured strings.
  * No torch / API dependency; small install.
  * The interface is a single class, so swapping in OpenAI embeddings or
    `sentence-transformers/bge-small-en-v1.5` is one method change.

The pattern aggregator (`aggregate_payer_procedure_carc_stats`) computes
per-(payer x procedure x CARC) denial rate, $ at risk, and historical
recovery proxy, used by the root-cause prompt and the clustering layer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import Claim


def _amount_bucket(amount: float) -> str:
    if amount < 100:
        return "amt_lt_100"
    if amount < 500:
        return "amt_100_500"
    if amount < 2000:
        return "amt_500_2k"
    if amount < 10000:
        return "amt_2k_10k"
    return "amt_gt_10k"


def claim_signature(claim: Claim) -> str:
    """Build the short, structured signature used for both filters and TF-IDF."""
    sub = claim.submission
    rem = claim.remittance
    payer = (sub.payer_id if sub and sub.payer_id else (rem.payer_id if rem else "")) or ""
    ins = (sub.insurance_type if sub and sub.insurance_type else (rem.insurance_type if rem else "")) or ""
    proc = claim.primary_procedure or ""
    proc_family = proc[:3] if proc else ""
    pos = (sub.place_of_service if sub else "") or ""
    spec = (sub.rendering_provider_specialty if sub else "") or ""
    dx_tokens: list[str] = []
    for dx in claim.all_diagnoses:
        dx_tokens.append(dx.replace(".", "_"))
        if "." in dx:
            dx_tokens.append(dx.split(".", 1)[0])
    amt_bucket = _amount_bucket((rem.claim_amount if rem else 0.0) or 0.0)
    parts = [
        f"payer_{payer}",
        f"ins_{ins}",
        f"proc_{proc}",
        f"procfam_{proc_family}",
        f"pos_{pos}",
        f"spec_{spec}",
        amt_bucket,
        *dx_tokens,
    ]
    return " ".join(p for p in parts if p and not p.endswith("_"))


@dataclass(frozen=True)
class SimilarClaim:
    claim_id: str
    score: float
    is_paid: bool
    payer_id: str | None
    procedure: str | None
    diagnosis: str | None
    amount: float


@dataclass(frozen=True)
class PatternStats:
    payer_id: str
    procedure: str
    carc_code: str
    n_total: int
    n_denied: int
    denied_amount_total: float
    paid_amount_total: float

    @property
    def denial_rate(self) -> float:
        return self.n_denied / self.n_total if self.n_total else 0.0

    @property
    def recovery_proxy(self) -> float:
        """Historical paid $ as a fraction of (paid + denied) $ for the (payer, procedure)."""
        denom = self.paid_amount_total + self.denied_amount_total
        return self.paid_amount_total / denom if denom else 0.0


class SimilarityIndex:
    """In-memory hybrid index over a corpus of Claim objects."""

    def __init__(self, corpus: Iterable[Claim]) -> None:
        self._claims: list[Claim] = list(corpus)
        self._signatures: list[str] = [claim_signature(c) for c in self._claims]
        self._vectorizer = TfidfVectorizer(token_pattern=r"\S+")
        self._matrix = (
            self._vectorizer.fit_transform(self._signatures) if self._signatures else None
        )

    def _payer(self, c: Claim) -> str:
        return (
            (c.submission.payer_id if c.submission else None)
            or (c.remittance.payer_id if c.remittance else None)
            or ""
        )

    def _ins(self, c: Claim) -> str:
        return (
            (c.submission.insurance_type if c.submission else None)
            or (c.remittance.insurance_type if c.remittance else None)
            or ""
        )

    def _filter_indices(self, claim: Claim, *, paid_only: bool) -> list[int]:
        payer = self._payer(claim)
        ins = self._ins(claim)
        proc = claim.primary_procedure or ""
        proc_family = proc[:3] if proc else ""

        out: list[int] = []
        for i, c in enumerate(self._claims):
            if c.claim_id == claim.claim_id:
                continue
            if paid_only and c.is_denied:
                continue
            c_payer = self._payer(c)
            c_ins = self._ins(c)
            c_proc = c.primary_procedure or ""
            c_proc_family = c_proc[:3] if c_proc else ""
            if payer and c_payer != payer:
                continue
            if ins and c_ins != ins:
                continue
            if proc_family and c_proc_family != proc_family:
                continue
            out.append(i)
        return out

    def find_similar_paid(self, claim: Claim, *, top_k: int = 5) -> list[SimilarClaim]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        candidate_idx = self._filter_indices(claim, paid_only=True)
        if not candidate_idx:
            return []
        query_vec = self._vectorizer.transform([claim_signature(claim)])
        sub_matrix = self._matrix[candidate_idx]
        sims = cosine_similarity(query_vec, sub_matrix)[0]
        top = np.argsort(-sims)[:top_k]
        out: list[SimilarClaim] = []
        for idx in top:
            actual_idx = candidate_idx[int(idx)]
            c = self._claims[actual_idx]
            rem = c.remittance
            sub = c.submission
            out.append(
                SimilarClaim(
                    claim_id=c.claim_id,
                    score=float(sims[int(idx)]),
                    is_paid=not c.is_denied,
                    payer_id=self._payer(c),
                    procedure=c.primary_procedure,
                    diagnosis=sub.principal_diagnosis if sub else None,
                    amount=(rem.claim_amount if rem else 0.0) or 0.0,
                )
            )
        return out

    def __len__(self) -> int:
        return len(self._claims)


def aggregate_payer_procedure_carc_stats(corpus: Iterable[Claim]) -> list[PatternStats]:
    """Compute per-(payer x procedure x CARC) statistics across the corpus."""
    buckets: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n_total": 0, "n_denied": 0, "denied_amount": 0.0, "paid_amount": 0.0}
    )
    paid_by_payer_proc: dict[tuple[str, str], float] = defaultdict(float)
    for c in corpus:
        rem = c.remittance
        sub = c.submission
        payer = (
            (sub.payer_id if sub and sub.payer_id else (rem.payer_id if rem else None))
            or "UNKNOWN"
        )
        proc = c.primary_procedure or "UNKNOWN"
        carc = c.primary_carc or "PAID"
        key = (payer, proc, carc)
        b = buckets[key]
        b["n_total"] += 1
        if c.is_denied:
            b["n_denied"] += 1
            b["denied_amount"] += rem.claim_amount if rem else 0.0
        else:
            paid_by_payer_proc[(payer, proc)] += rem.claim_paid if rem else 0.0
    # Spread paid totals across all CARC rows for the same (payer, proc) so the
    # recovery_proxy on a denied row reflects historical performance for that pair.
    out: list[PatternStats] = []
    for (payer, proc, carc), b in buckets.items():
        out.append(
            PatternStats(
                payer_id=payer,
                procedure=proc,
                carc_code=carc,
                n_total=int(b["n_total"]),
                n_denied=int(b["n_denied"]),
                denied_amount_total=float(b["denied_amount"]),
                paid_amount_total=float(paid_by_payer_proc[(payer, proc)]),
            )
        )
    return sorted(out, key=lambda s: -s.denied_amount_total)
