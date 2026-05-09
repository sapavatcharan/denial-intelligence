"""Cluster denials and produce billing-team action briefs.

Approach: for tens to thousands of denied claims, *meaningful* clusters in
this domain are dominated by structured signal (payer, CARC family,
procedure family). We group by `(payer_id, carc_family, procedure_family)`
and rank clusters by total denied $ - the dimension a billing manager
prioritizes by. This is more interpretable than HDBSCAN over a sparse
one-hot vector at this scale and produces stable, reviewable groups.

For each cluster we compute:
  * count of claims
  * total denied $
  * top procedures and dx codes within the cluster
  * historical recovery proxy from the pattern aggregator
and feed those aggregates to the LLM cluster narrator (or the mock).

The narrator returns a single human-readable paragraph that a manager can
read in 15 seconds and act on.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMClient
from ..llm.mock_client import MockLLMClient
from ..reference import carc as carc_lookup
from ..retrieval import PatternStats, aggregate_payer_procedure_carc_stats
from ..schemas import Claim

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


@dataclass
class DenialCluster:
    cluster_id: str
    payer_id: str
    carc_code: str
    carc_family: str
    procedure_family: str
    n_claims: int
    total_denied_amount: float
    top_procedures: list[tuple[str, int]]
    top_diagnoses: list[tuple[str, int]]
    claim_ids: list[str]
    historical_recovery_proxy: float
    historical_denial_rate: float
    narrative: str = ""
    payer_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "payer_id": self.payer_id,
            "payer_name": self.payer_name,
            "carc_code": self.carc_code,
            "carc_family": self.carc_family,
            "procedure_family": self.procedure_family,
            "n_claims": self.n_claims,
            "total_denied_amount": round(self.total_denied_amount, 2),
            "top_procedures": self.top_procedures,
            "top_diagnoses": self.top_diagnoses,
            "claim_ids": self.claim_ids,
            "historical_recovery_proxy": round(self.historical_recovery_proxy, 3),
            "historical_denial_rate": round(self.historical_denial_rate, 3),
            "narrative": self.narrative,
        }


@dataclass
class _Bucket:
    claims: list[Claim] = field(default_factory=list)


def cluster_denials(corpus: Iterable[Claim]) -> list[DenialCluster]:
    """Group denied claims by (payer, carc_family, procedure_family)."""
    corpus = list(corpus)
    stats = aggregate_payer_procedure_carc_stats(corpus)
    stats_lookup = _stats_index(stats)

    buckets: dict[tuple[str, str, str], _Bucket] = defaultdict(_Bucket)
    for c in corpus:
        if not c.is_denied:
            continue
        carc = c.primary_carc or "UNKNOWN"
        info = carc_lookup(carc) or {}
        family = info.get("family", "unknown")
        proc = c.primary_procedure or "UNKNOWN"
        proc_family = proc[:3] if proc else "UNKNOWN"
        sub = c.submission
        rem = c.remittance
        payer = (
            (sub.payer_id if sub and sub.payer_id else (rem.payer_id if rem else None))
            or "UNKNOWN"
        )
        buckets[(payer, family, proc_family)].claims.append(c)

    clusters: list[DenialCluster] = []
    for (payer, family, proc_family), bucket in buckets.items():
        ids = [c.claim_id for c in bucket.claims]
        total = sum((c.remittance.claim_amount if c.remittance else 0.0) for c in bucket.claims)
        proc_counter = Counter(c.primary_procedure or "?" for c in bucket.claims)
        dx_counter = Counter(
            (c.submission.principal_diagnosis if c.submission else None) or "?"
            for c in bucket.claims
        )
        carc_counter = Counter(c.primary_carc or "?" for c in bucket.claims)
        dominant_carc = carc_counter.most_common(1)[0][0]
        recovery_proxy = 0.0
        denial_rate = 0.0
        n = 0
        for proc, _ in proc_counter.most_common(3):
            stat = stats_lookup.get((payer, proc, dominant_carc))
            if stat:
                recovery_proxy += stat.recovery_proxy
                denial_rate += stat.denial_rate
                n += 1
        if n:
            recovery_proxy /= n
            denial_rate /= n
        cluster_payer_name = next(
            (
                (c.submission.payer_name if c.submission else None)
                or (c.remittance.payer_name if c.remittance else None)
                for c in bucket.claims
                if (c.submission and c.submission.payer_name)
                or (c.remittance and c.remittance.payer_name)
            ),
            None,
        )
        clusters.append(
            DenialCluster(
                cluster_id=f"{payer}|{family}|{proc_family}",
                payer_id=payer,
                payer_name=cluster_payer_name,
                carc_code=dominant_carc,
                carc_family=family,
                procedure_family=proc_family,
                n_claims=len(bucket.claims),
                total_denied_amount=total,
                top_procedures=proc_counter.most_common(3),
                top_diagnoses=dx_counter.most_common(3),
                claim_ids=ids,
                historical_recovery_proxy=recovery_proxy,
                historical_denial_rate=denial_rate,
            )
        )
    clusters.sort(key=lambda c: -c.total_denied_amount)
    return clusters


def _stats_index(stats: list[PatternStats]) -> dict[tuple[str, str, str], PatternStats]:
    return {(s.payer_id, s.procedure, s.carc_code): s for s in stats}


def _cluster_user_prompt(c: DenialCluster) -> str:
    return (
        f"<cluster>\n"
        f"cluster_id: {c.cluster_id}\n"
        f"payer_id: {c.payer_id}\n"
        f"payer_name: {c.payer_name or '?'}\n"
        f"dominant_carc_code: {c.carc_code}\n"
        f"carc_family: {c.carc_family}\n"
        f"procedure_family: {c.procedure_family}\n"
        f"n_claims: {c.n_claims}\n"
        f"total_denied_usd: ${c.total_denied_amount:,.2f}\n"
        f"top_procedures: {c.top_procedures}\n"
        f"top_diagnoses: {c.top_diagnoses}\n"
        f"historical_denial_rate: {c.historical_denial_rate:.0%}\n"
        f"historical_recovery_proxy: {c.historical_recovery_proxy:.0%}\n"
        f"</cluster>\n\n"
        "Write the one-paragraph manager brief now."
    )


def _build_default_client() -> LLMClient | MockLLMClient:
    if os.environ.get("GABEO_MOCK_LLM") == "1":
        return MockLLMClient()
    if not os.environ.get("OPENAI_API_KEY"):
        return MockLLMClient()
    try:
        return LLMClient()
    except Exception:
        return MockLLMClient()


def _mock_narrative(c: DenialCluster) -> str:
    """Deterministic fallback narrative when no LLM is available."""
    info = carc_lookup(c.carc_code) or {}
    desc = info.get("description", "Denial cluster.")
    action_map = {
        "prior_auth": "Escalate to the prior-authorization team to obtain or re-attach authorization numbers and resubmit.",
        "medical_necessity": "Route to the coding team to repoint diagnoses or attach LCD/NCD-supportive documentation, then appeal.",
        "timely_filing": "If a delay reason exists, submit a timely-filing appeal with documentation; otherwise write off.",
        "duplicate": "Verify against the claim history; for paired-organ services, append LT/RT modifiers and resubmit.",
        "coding": "Apply the missing modifier and resubmit corrected claims via the coding team.",
        "missing_info": "Populate the field(s) called out by the RARC code and resubmit corrected claims.",
        "contractual": "Write off as a contractual adjustment.",
        "non_covered": "Verify benefits and bill the patient if appropriate.",
    }
    action = action_map.get(c.carc_family, "Send to a billing analyst for case-by-case review.")
    return (
        f"You have {c.n_claims} {c.payer_name or c.payer_id} claim(s) totaling "
        f"${c.total_denied_amount:,.2f} denied with CARC {c.carc_code} ({desc}). "
        f"Top procedure(s): {c.top_procedures[0][0] if c.top_procedures else '?'}. "
        f"Historical denial rate for this payer/procedure pair is "
        f"{c.historical_denial_rate:.0%}; historical paid share is {c.historical_recovery_proxy:.0%}. "
        f"{action}"
    )


def build_batch_brief(
    clusters: list[DenialCluster],
    *,
    llm: LLMClient | MockLLMClient | None = None,
    top_n: int = 10,
) -> list[DenialCluster]:
    """Attach an LLM-written narrative to the top N clusters by $ at risk."""
    client = llm or _build_default_client()
    system_prompt = (PROMPTS_DIR / "cluster_summary.md").read_text()

    for cluster in clusters[:top_n]:
        try:
            response = client.chat(
                model=os.environ.get("GABEO_LLM_TRIAGE_MODEL", "gpt-4o-mini"),
                system=system_prompt,
                user=_cluster_user_prompt(cluster),
            )
            cluster.narrative = response.text.strip()
        except Exception:
            cluster.narrative = _mock_narrative(cluster)
    for cluster in clusters[top_n:]:
        cluster.narrative = _mock_narrative(cluster)
    return clusters


def render_brief_markdown(clusters: list[DenialCluster]) -> str:
    """Render a Markdown brief for the human team."""
    lines: list[str] = ["# Denial Batch Brief\n"]
    grand_total = sum(c.total_denied_amount for c in clusters)
    lines.append(
        f"**{len(clusters)}** clusters covering **{sum(c.n_claims for c in clusters)}** denied claims "
        f"totaling **${grand_total:,.2f}** at risk.\n"
    )
    for i, c in enumerate(clusters, 1):
        lines.append(f"## {i}. {c.payer_name or c.payer_id} | CARC {c.carc_code} | proc {c.procedure_family}xx\n")
        lines.append(
            f"- **{c.n_claims}** claims, **${c.total_denied_amount:,.2f}** at risk\n"
            f"- Top procedures: {c.top_procedures}\n"
            f"- Top diagnoses: {c.top_diagnoses}\n"
            f"- Historical denial rate: {c.historical_denial_rate:.0%} | "
            f"recovery proxy: {c.historical_recovery_proxy:.0%}\n"
        )
        lines.append(f"\n{c.narrative}\n")
    return "\n".join(lines)
