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

# --- Global styling ----------------------------------------------------------
# A small CSS injection is the simplest way to add visual polish without
# pulling in a UI framework. We scope all of it so it cannot collide with
# Streamlit's own classes after future releases.
st.markdown(
    """
    <style>
      .gabeo-hero {
        background: linear-gradient(135deg, #7c3aed 0%, #ec4899 50%, #f59e0b 100%);
        padding: 28px 32px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.25);
      }
      .gabeo-hero h1 {
        margin: 0 0 6px 0;
        font-size: 2rem;
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.01em;
      }
      .gabeo-hero p { margin: 0; opacity: 0.95; font-size: 1.0rem; }
      .gabeo-hero .gabeo-pill {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 8px;
      }
      .gabeo-card {
        background: #ffffff;
        border: 1px solid #ede9fe;
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
      }
      .gabeo-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        color: #ffffff;
        margin-right: 4px;
      }
      .gabeo-sev-critical { background: #dc2626; }
      .gabeo-sev-warning  { background: #d97706; }
      .gabeo-sev-info     { background: #0ea5e9; }
      .gabeo-pass         { background: #16a34a; }
      .gabeo-fail         { background: #dc2626; }
      .gabeo-conf-bar {
        height: 8px;
        border-radius: 999px;
        background: #ede9fe;
        overflow: hidden;
        margin-top: 4px;
      }
      .gabeo-conf-bar > div {
        height: 100%;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
      }
      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #faf5ff 0%, #ffffff 100%);
        border: 1px solid #ede9fe;
        border-radius: 10px;
        padding: 10px 14px;
      }
      .stTabs [data-baseweb="tab"] {
        font-weight: 600;
      }
      .stTabs [aria-selected="true"] {
        color: #7c3aed !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = ROOT / "data" / "synthetic" / "claims.jsonl"

# Recoverability + severity color palettes used across the app.
_RECOVER_COLORS: dict[str, str] = {
    "recoverable": "#16a34a",
    "needs_review": "#d97706",
    "not_recoverable": "#dc2626",
    "informational": "#6b7280",
}


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
    color = _RECOVER_COLORS.get(value, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:6px 14px;'
        f'border-radius:999px;font-weight:700;font-size:0.95rem;'
        f'box-shadow:0 2px 6px {color}40;">{value.replace("_", " ").upper()}</span>'
    )


def _severity_badge(severity: str, passed: bool) -> str:
    label = "PASS" if passed else "FAIL"
    cls = "gabeo-pass" if passed else "gabeo-fail"
    sev_cls = {
        "critical": "gabeo-sev-critical",
        "warning": "gabeo-sev-warning",
        "info": "gabeo-sev-info",
    }.get(severity.lower(), "gabeo-sev-info")
    return (
        f'<span class="gabeo-badge {sev_cls}">{severity.upper()}</span>'
        f'<span class="gabeo-badge {cls}">{label}</span>'
    )


def _confidence_bar(value: float) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f'<div class="gabeo-conf-bar"><div style="width:{pct:.0f}%;"></div></div>'


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
    days_to_receipt: int | str = "?"
    if sub and sub.service_date_from and rem and rem.received_date:
        days_to_receipt = (rem.received_date - sub.service_date_from).days
    cols = st.columns(4)
    cols[0].metric("Charged", _format_money(rem.claim_amount if rem else 0.0))
    cols[1].metric("Paid", _format_money(rem.claim_paid if rem else 0.0))
    cols[2].metric("Primary CARC", claim.primary_carc or "PAID")
    cols[3].metric("Days service to receipt", days_to_receipt)
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
    color = _RECOVER_COLORS.get(verdict.recoverability.value, "#6b7280")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{color}18,{color}08);
                    border:1px solid {color}40;border-left:5px solid {color};
                    border-radius:12px;padding:18px 22px;margin-bottom:14px;">
          <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div>{_recoverability_badge(verdict.recoverability.value)}</div>
            <div style="color:#475569;font-size:0.85rem;">
              model: <code>{verdict.model_used or '?'}</code> ·
              latency: <b>{verdict.latency_ms or 0} ms</b> ·
              cost: <b>${(verdict.cost_usd or 0):.5f}</b>
            </div>
          </div>
          <div style="margin-top:10px;color:#0f172a;">
            <b>Confidence (blended):</b> {verdict.confidence:.2f}
            {_confidence_bar(verdict.confidence)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Root cause")
    st.write(verdict.root_cause)
    st.markdown("#### Recommended action")
    st.success(verdict.recommended_action)
    st.markdown("#### CARC interpretation")
    st.info(verdict.carc_interpretation)

    st.markdown("#### Confidence components")
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

    st.markdown("#### Deterministic evidence trail")
    if verdict.deterministic_evidence:
        for e in verdict.deterministic_evidence:
            border_color = (
                "#16a34a"
                if e.passed
                else ("#dc2626" if e.severity == "critical" else "#d97706")
            )
            fields_html = ""
            if e.fields_referenced:
                fields_html = (
                    "<div style='margin-top:6px;font-size:0.78rem;color:#64748b;'>"
                    + " ".join(
                        f"<code style='background:#f1f5f9;padding:1px 6px;border-radius:4px;'>{f}</code>"
                        for f in e.fields_referenced
                    )
                    + "</div>"
                )
            st.markdown(
                f"""
                <div class='gabeo-card' style='border-left-color:{border_color};'>
                  <div>{_severity_badge(e.severity, e.passed)}
                       <code style='color:#7c3aed;'>{e.check_name}</code></div>
                  <div style='margin-top:6px;color:#0f172a;'>{e.message}</div>
                  {fields_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("(no deterministic findings)")

    st.markdown("#### Similar paid claims used as historical context")
    if verdict.similar_paid_claims:
        st.write(", ".join(f"`{cid}`" for cid in verdict.similar_paid_claims))
    else:
        st.write("(none)")


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
                "carc": str(c["carc_code"]),
                "carc_family": c["carc_family"],
                "proc_family": c["procedure_family"],
                "claims": int(c["n_claims"]),
                "denied_$": float(c["total_denied_amount"]),
                "hist_denial_rate": float(c["historical_denial_rate"]),
                "hist_recovery_proxy": float(c["historical_recovery_proxy"]),
            }
            for i, c in enumerate(clusters)
        ]
    )
    st.dataframe(df, use_container_width=True)

    st.subheader("Cluster narratives")
    family_color = {
        "timely_filing": "#dc2626",
        "prior_auth": "#7c3aed",
        "medical_necessity": "#0ea5e9",
        "coding": "#f59e0b",
        "duplicate": "#ec4899",
        "missing_info": "#10b981",
        "contractual": "#6b7280",
        "non_covered": "#64748b",
    }
    for i, c in enumerate(clusters, 1):
        accent = family_color.get(c["carc_family"], "#7c3aed")
        with st.expander(
            f"#{i} {c.get('payer_name') or c['payer_id']} | CARC {c['carc_code']} | "
            f"{c['n_claims']} claims | {_format_money(c['total_denied_amount'])}",
            expanded=(i == 1),
        ):
            st.markdown(
                f"""
                <div class='gabeo-card' style='border-left-color:{accent};'>
                  <div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;'>
                    <span class='gabeo-badge' style='background:{accent};'>
                      {c['carc_family'].replace('_', ' ').upper()}
                    </span>
                    <span style='color:#475569;font-size:0.85rem;'>
                      Hist denial rate: <b>{c['historical_denial_rate']:.0%}</b> ·
                      Recovery proxy: <b>{c['historical_recovery_proxy']:.0%}</b>
                    </span>
                  </div>
                  <div style='margin-top:10px;color:#0f172a;line-height:1.55;'>{c['narrative']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
    safe_rows = []
    for r in report["rows"]:
        rr = dict(r)
        rr["keywords_expected"] = ", ".join(rr.get("keywords_expected") or [])
        rr["keywords_present"] = ", ".join(rr.get("keywords_present") or [])
        safe_rows.append(rr)
    st.dataframe(pd.DataFrame(safe_rows), use_container_width=True)


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


def _safe_tab(label: str, fn, *args) -> None:  # type: ignore[no-untyped-def]
    """Render a tab inside an error boundary so a single tab can't kill the app."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The '{label}' tab hit an error. Other tabs still work.")
        st.exception(exc)


def main() -> None:
    st.markdown(
        """
        <div class="gabeo-hero">
          <h1>Gabeo Denial AI</h1>
          <p>Hybrid deterministic + LLM analysis for healthcare claim denials.
             Built on EDI 837 / 835 schemas with grounded LLM citations and calibrated confidence.</p>
          <div>
            <span class="gabeo-pill">Pydantic v2</span>
            <span class="gabeo-pill">TF-IDF retrieval</span>
            <span class="gabeo-pill">Grounding gate</span>
            <span class="gabeo-pill">Mock-LLM fallback</span>
            <span class="gabeo-pill">26 tests passing</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        _safe_tab("Analyze a claim", render_analyze_tab, claims, gold, index, agent)
    with tab2:
        _safe_tab("Batch brief", render_clusters_tab)
    with tab3:
        _safe_tab("Eval results", render_eval_tab)
    with tab4:
        _safe_tab("Brief samples", render_brief_samples_tab)


if __name__ == "__main__":
    main()
