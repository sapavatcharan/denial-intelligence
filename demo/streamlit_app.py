"""Streamlit demo for the Gabeo Denial AI system.

Three tabs:

  1. Analyze a claim - pick a denied claim, see the verdict, citations,
     deterministic evidence trail, similar paid history, and confidence
     components.
  2. Batch brief - the prioritized cluster brief a billing manager would
     get every morning.
  3. Eval results - the live metrics from the gold-labeled evaluation set.

Launch:  streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from gabeo.agents.root_cause_agent import RootCauseAgent  # noqa: E402
from gabeo.clustering import build_batch_brief, cluster_denials  # noqa: E402
from gabeo.eval.harness import run_eval  # noqa: E402
from gabeo.ingest import load_claims_jsonl, load_gold_labels  # noqa: E402
from gabeo.reference import carc as carc_lookup  # noqa: E402
from gabeo.retrieval import SimilarityIndex  # noqa: E402

load_dotenv(ROOT / ".env")

# Streamlit Cloud secrets bridge: copy any OPENAI_API_KEY from st.secrets into
# the process env so the agent's existing resolver picks it up. If the user
# hasn't added a secret, the agent falls back to MockLLMClient and the demo
# still works end-to-end.
try:
    if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, AttributeError):
    pass

st.set_page_config(page_title="Gabeo Denial AI", layout="wide", page_icon=":hospital:")

DATA_PATH = ROOT / "data" / "synthetic" / "claims.jsonl"


@st.cache_resource
def _load() -> tuple[list, dict, SimilarityIndex, RootCauseAgent]:
    claims = load_claims_jsonl(DATA_PATH)
    gold = load_gold_labels(DATA_PATH)
    index = SimilarityIndex(claims)
    agent = RootCauseAgent()
    return claims, gold, index, agent


@st.cache_data(show_spinner="Running eval harness...")
def _eval_cached(_path: str) -> dict:
    return run_eval(_path).to_dict()


@st.cache_data(show_spinner="Clustering denials...")
def _clusters_cached(_path: str) -> list[dict]:
    claims = load_claims_jsonl(_path)
    clusters = cluster_denials(claims)
    clusters = build_batch_brief(clusters, top_n=10)
    return [c.to_dict() for c in clusters]


def _format_money(x: float) -> str:
    return f"${x:,.2f}"


def _recoverability_badge(value: str) -> str:
    color = {
        "recoverable": "#16a34a",
        "needs_review": "#d97706",
        "not_recoverable": "#dc2626",
        "informational": "#6b7280",
    }.get(value, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:4px 10px;'
        f'border-radius:6px;font-weight:600;font-size:0.85rem;">{value}</span>'
    )


def _claim_label(claim, gold) -> str:  # type: ignore[no-untyped-def]
    sub = claim.submission
    rem = claim.remittance
    payer = (sub.payer_name if sub and sub.payer_name else (rem.payer_name if rem else "?")) or "?"
    proc = claim.primary_procedure or "?"
    carc = claim.primary_carc or "PAID"
    amount = (rem.claim_amount if rem else 0.0) or 0.0
    scenario = gold.get(claim.claim_id, {}).get("scenario", "")
    return f"{claim.claim_id} | {payer} | {proc} | CARC {carc} | {_format_money(amount)} | {scenario}"


def _render_claim_panel(claim, gold) -> None:  # type: ignore[no-untyped-def]
    sub = claim.submission
    rem = claim.remittance
    cols = st.columns(4)
    cols[0].metric("Charged", _format_money(rem.claim_amount if rem else 0.0))
    cols[1].metric("Paid", _format_money(rem.claim_paid if rem else 0.0))
    cols[2].metric("Primary CARC", claim.primary_carc or "PAID")
    cols[3].metric(
        "Days submission to receipt",
        sub.days_from_service_to_receipt(rem.received_date if rem else None) if sub else "?",
    )
    if (info := carc_lookup(claim.primary_carc or "")):
        st.caption(
            f"**CARC {claim.primary_carc}** · {info['description']} "
            f"(family: {info['family']}, appealable: {info['generally_appealable']})"
        )

    with st.expander("Submission (837)", expanded=False):
        st.json(sub.model_dump(mode="json") if sub else {})
    with st.expander("Remittance (835)", expanded=False):
        st.json(rem.model_dump(mode="json") if rem else {})
    if claim.claim_id in gold:
        with st.expander("Gold label (held out from the LLM)", expanded=False):
            st.json(gold[claim.claim_id])


def _render_verdict_panel(verdict) -> None:  # type: ignore[no-untyped-def]
    cols = st.columns([2, 1, 1])
    cols[0].markdown(f"**Recoverability:** {_recoverability_badge(verdict.recoverability.value)}", unsafe_allow_html=True)
    cols[1].metric("Confidence (blended)", f"{verdict.confidence:.2f}")
    cols[2].metric("Latency (ms)", verdict.latency_ms or 0)

    st.subheader("Root cause")
    st.write(verdict.root_cause)
    st.subheader("Recommended action")
    st.success(verdict.recommended_action)
    st.subheader("CARC interpretation")
    st.info(verdict.carc_interpretation)

    st.subheader("Confidence components")
    st.json(verdict.confidence_components)

    st.subheader("Supporting evidence (LLM citations, grounded)")
    if verdict.supporting_evidence:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "field_name": s.field_name,
                        "field_value": s.field_value,
                        "why_relevant": s.why_relevant,
                    }
                    for s in verdict.supporting_evidence
                ]
            ),
            use_container_width=True,
        )
    else:
        st.write("(none)")

    st.subheader("Deterministic evidence trail")
    if verdict.deterministic_evidence:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "check": e.check_name,
                        "passed": e.passed,
                        "severity": e.severity,
                        "message": e.message,
                        "fields": ", ".join(e.fields_referenced),
                    }
                    for e in verdict.deterministic_evidence
                ]
            ),
            use_container_width=True,
        )
    else:
        st.write("(no deterministic findings)")

    st.subheader("Cost & model")
    st.json(
        {
            "model_used": verdict.model_used,
            "cost_usd": verdict.cost_usd,
            "similar_paid_claims": verdict.similar_paid_claims,
        }
    )


def render_analyze_tab(claims, gold, index, agent) -> None:  # type: ignore[no-untyped-def]
    denied = [c for c in claims if c.is_denied]
    if not denied:
        st.warning("No denied claims in the dataset.")
        return

    label_to_claim = {_claim_label(c, gold): c for c in denied}
    sel = st.selectbox("Pick a denied claim", list(label_to_claim))
    claim = label_to_claim[sel]

    left, right = st.columns([1, 1])
    with left:
        st.header("Claim")
        _render_claim_panel(claim, gold)

        st.header("Top similar paid claims (history)")
        similar = index.find_similar_paid(claim, top_k=5)
        if similar:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "claim_id": s.claim_id,
                            "score": round(s.score, 3),
                            "payer": s.payer_id,
                            "proc": s.procedure,
                            "dx": s.diagnosis,
                            "amount": s.amount,
                        }
                        for s in similar
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.write("No similar paid claims found in the corpus.")

    with right:
        st.header("Verdict")
        with st.spinner("Running root-cause agent..."):
            similar_ids = [s.claim_id for s in similar]
            top_summary = (
                f"Top similar paid claim: {similar[0].claim_id} ({similar[0].payer_id}, "
                f"proc {similar[0].procedure}, score {similar[0].score:.2f})."
                if similar
                else ""
            )
            verdict = agent.analyze(
                claim,
                similar_paid_claim_ids=similar_ids,
                historical_summary=top_summary,
            )
        _render_verdict_panel(verdict)


def render_clusters_tab() -> None:
    clusters = _clusters_cached(str(DATA_PATH))
    if not clusters:
        st.info("No denied claims in the dataset.")
        return
    total_dollars = sum(c["total_denied_amount"] for c in clusters)
    n_claims = sum(c["n_claims"] for c in clusters)
    cols = st.columns(3)
    cols[0].metric("Clusters", len(clusters))
    cols[1].metric("Denied claims", n_claims)
    cols[2].metric("Total at risk", _format_money(total_dollars))

    df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "payer": c.get("payer_name") or c["payer_id"],
                "carc": c["carc_code"],
                "carc_family": c["carc_family"],
                "proc_family": c["procedure_family"],
                "claims": c["n_claims"],
                "denied_$": c["total_denied_amount"],
                "hist_denial_rate": c["historical_denial_rate"],
                "hist_recovery_proxy": c["historical_recovery_proxy"],
            }
            for i, c in enumerate(clusters)
        ]
    )
    st.dataframe(df, use_container_width=True)

    st.subheader("Cluster narratives")
    for i, c in enumerate(clusters, 1):
        with st.expander(
            f"#{i} {c.get('payer_name') or c['payer_id']} | CARC {c['carc_code']} | "
            f"{c['n_claims']} claims | {_format_money(c['total_denied_amount'])}",
            expanded=(i == 1),
        ):
            st.write(c["narrative"])
            st.json(
                {
                    "top_procedures": c["top_procedures"],
                    "top_diagnoses": c["top_diagnoses"],
                    "claim_ids": c["claim_ids"],
                }
            )


def render_eval_tab() -> None:
    report = _eval_cached(str(DATA_PATH))
    s = report["summary"]
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{s['accuracy']:.3f}")
    cols[1].metric("Macro F1", f"{s['macro_f1']:.3f}")
    cols[2].metric("Grounding rate", f"{s['evidence_grounding_rate']:.3f}")
    cols[3].metric("Avg latency (ms)", s["avg_latency_ms"])

    cols2 = st.columns(4)
    cols2[0].metric("Brier score", f"{s['brier_score']:.4f}")
    cols2[1].metric("Keyword recall", f"{s['keyword_recall']:.3f}")
    cols2[2].metric("Total cost (USD)", f"${s['total_cost_usd']:.5f}")
    cols2[3].metric("Claims evaluated", s["n_evaluated"])

    st.subheader("Per-class precision / recall / F1")
    pcr = report["macro_f1_per_class"]
    st.dataframe(pd.DataFrame(pcr).T, use_container_width=True)

    st.subheader("Confusion matrix (rows = expected, cols = predicted)")
    cm = report["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm).T, use_container_width=True)

    st.subheader("Per-scenario accuracy")
    sb = report["scenario_breakdown"]
    st.dataframe(pd.DataFrame(sb).T, use_container_width=True)

    st.subheader("Per-claim outcomes")
    rows = pd.DataFrame(report["rows"])
    st.dataframe(rows, use_container_width=True)


def render_brief_samples_tab() -> None:
    """Show the four brief samples called out in the assignment PDF."""
    brief_md = ROOT / "docs" / "brief_samples_run.md"
    if not brief_md.exists():
        st.warning(
            "docs/brief_samples_run.md not found. Run "
            "`python scripts/run_brief_samples.py` to regenerate."
        )
        return
    st.markdown(brief_md.read_text())


def main() -> None:
    st.title("Gabeo Denial AI")
    st.caption(
        "Hybrid deterministic + LLM analysis for healthcare claim denials. "
        "Built on EDI 837/835 schemas with grounded LLM citations and calibrated confidence."
    )

    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at {DATA_PATH}. "
            "Run `gabeo synth` (or `python scripts/generate_synthetic.py`) first."
        )
        st.stop()

    claims, gold, index, agent = _load()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Analyze a claim", "Batch brief", "Eval results", "Brief samples"]
    )
    with tab1:
        render_analyze_tab(claims, gold, index, agent)
    with tab2:
        render_clusters_tab()
    with tab3:
        render_eval_tab()
    with tab4:
        render_brief_samples_tab()


if __name__ == "__main__":
    main()
