"""Run the root-cause agent on the four sample claims from the assignment PDF.

Output: docs/brief_samples_run.md - human-readable verdict + grounded
citations + deterministic evidence trail for each of the four claims.

This is the script the README points reviewers at as a one-shot reproduction
of the assignment's worked examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from gabeo.agents.root_cause_agent import RootCauseAgent  # noqa: E402
from gabeo.ingest import load_claims_jsonl  # noqa: E402
from gabeo.retrieval import SimilarityIndex  # noqa: E402
from gabeo.schemas import Verdict  # noqa: E402

BRIEF_IDS = [
    "CLM-2026-00142",
    "CLM-2026-00287",
    "CLM-2026-00391",
    "CLM-2026-00455",
]


def render(verdict: Verdict) -> str:
    sup = "\n".join(
        f"- `{s.field_name}` = `{s.field_value}` — {s.why_relevant}"
        for s in verdict.supporting_evidence
    ) or "- (none)"
    det = "\n".join(
        f"- [{e.severity.upper()}] [{'PASS' if e.passed else 'FAIL'}] "
        f"`{e.check_name}` — {e.message}"
        for e in verdict.deterministic_evidence
    ) or "- (none)"
    return (
        f"## {verdict.claim_id}\n\n"
        f"- **Recoverability:** `{verdict.recoverability.value}`\n"
        f"- **Confidence:** {verdict.confidence:.2f} "
        f"(components: {verdict.confidence_components})\n"
        f"- **Model:** {verdict.model_used} | cost: "
        f"${verdict.cost_usd:.5f} | latency: {verdict.latency_ms} ms\n\n"
        f"**Root cause.** {verdict.root_cause}\n\n"
        f"**CARC interpretation.** {verdict.carc_interpretation}\n\n"
        f"**Recommended action.** {verdict.recommended_action}\n\n"
        f"**Supporting evidence (cited fields):**\n\n{sup}\n\n"
        f"**Deterministic checks that fired:**\n\n{det}\n\n---\n"
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    data_path = ROOT / "data" / "synthetic" / "claims.jsonl"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}; run `gabeo synth` first.", file=sys.stderr)
        sys.exit(1)
    claims = load_claims_jsonl(data_path)
    by_id = {c.claim_id: c for c in claims}
    index = SimilarityIndex(claims)
    agent = RootCauseAgent()

    parts: list[str] = [
        "# Brief Sample Verdicts (live LLM run)\n",
        "These are the unedited verdicts produced by `gabeo analyze` on the "
        "four sample claims from the assignment PDF, plus the deterministic "
        "evidence that grounded each verdict.\n",
    ]
    for cid in BRIEF_IDS:
        if cid not in by_id:
            print(f"WARN: {cid} not in dataset", file=sys.stderr)
            continue
        target = by_id[cid]
        similar = index.find_similar_paid(target, top_k=5)
        verdict = agent.analyze(
            target,
            similar_paid_claim_ids=[s.claim_id for s in similar],
            historical_summary=(
                f"Top similar paid claim: {similar[0].claim_id} "
                f"({similar[0].payer_id}, score {similar[0].score:.2f})."
                if similar
                else ""
            ),
        )
        parts.append(render(verdict))

    out = ROOT / "docs" / "brief_samples_run.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
