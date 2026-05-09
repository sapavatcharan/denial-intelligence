# Gabeo Denial AI

> Hybrid deterministic + LLM analysis for healthcare claim denials, built on the
> EDI 837 (claim) and EDI 835 (remittance advice) schemas.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://denial-intelligence.streamlit.app)
&nbsp;[![Deploy your own](https://img.shields.io/badge/Deploy-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io/deploy?repository=sapavatcharan%2Fdenial-intelligence&branch=main&mainModule=demo%2Fstreamlit_app.py)

**Live demo:** https://denial-intelligence.streamlit.app
*(running with the deterministic mock LLM by default; add an `OPENAI_API_KEY` Streamlit secret to switch to GPT-4o.)*

The system answers three questions a billing team asks every morning:

1. **Why was this claim denied, and is it recoverable?** (root-cause + recoverability + recommended action, every fact cited and grounded in the input data)
2. **What does our history with this payer / procedure / diagnosis tell us about how to fix it?** (hybrid retrieval + pattern aggregator)
3. **Of the hundreds of denials in today's batch, where should we focus first?** (clustering + a prioritized batch brief in plain English)

This repo contains a working CLI, a Streamlit demo, an evaluation harness with
inline gold labels on a synthetic dataset, and a deterministic mock LLM so the
entire pipeline runs end-to-end without an OpenAI key.

---

## Headline numbers

Run `gabeo eval` to reproduce. On the synthetic gold set
(`data/synthetic/claims.jsonl`, 19 denied claims spanning 7 scenarios incl. four
adversarial cases):

| Metric                          | Value      |
| ------------------------------- | ---------- |
| Recoverability accuracy         | **0.947**  |
| Macro-F1 (recoverability)       | **0.958**  |
| Evidence grounding rate         | **1.000**  |
| Brier score (lower = better)    | 0.132      |
| Root-cause keyword recall       | 0.816      |
| Avg latency / claim             | ~6 ms*     |
| Total LLM cost / 19-claim eval  | $0.000*    |

\* The numbers above are with the deterministic `MockLLMClient` (the OpenAI key
in `.env` returns `insufficient_quota`). The same code path runs against
`gpt-4o-mini` and `gpt-4o` when a funded key is present; expected production
latency is 1.5–4 s/claim and cost ~$0.003/claim with the cost-aware router.

Also see:

* [`docs/eval_results.md`](docs/eval_results.md) — full eval breakdown
* [`docs/batch_brief.md`](docs/batch_brief.md) — auto-generated batch brief
* [`docs/brief_samples_run.md`](docs/brief_samples_run.md) — verdicts on the four claims from the assignment PDF
* [`docs/architecture.md`](docs/architecture.md) — system design
* [`docs/design_decisions.md`](docs/design_decisions.md) — every non-trivial trade-off, with the alternative considered

---

## Quickstart (5 minutes)

```bash
# 1. Install deps and create the venv (uses uv).
make install

# 2. Generate the synthetic dataset (40 claims, mix paid/denied, gold labels inline).
gabeo synth

# 3. Reproduce the brief samples report.
python scripts/run_brief_samples.py

# 4. Run the eval harness end-to-end.
gabeo eval

# 5. Cluster denials and write the manager brief.
gabeo cluster

# 6. Launch the demo UI.
make demo
```

If you have an OpenAI key, drop it in `.env` (template in `.env.example`) and
the same commands route to `gpt-4o-mini` / `gpt-4o` automatically. With no key
or with `GABEO_MOCK_LLM=1` set, the system falls back to `MockLLMClient` and
nothing else changes.

---

## What's in here

```
src/gabeo/
  schemas.py            # Pydantic v2 models for Claim, Submission (837), Remittance (835), Verdict
  ingest.py             # Flat JSON -> typed Claim, with proper modifier / RARC handling
  reference.py          # Loaders for CARC / RARC / payer-filing-limit / dx-procedure tables
  evidence/             # Six deterministic extractors (timely_filing, prior_auth, coding,
                        #   medical_necessity, duplicate, missing_info)
  llm/
    client.py           # Thin OpenAI client w/ structured output, retries, cost telemetry
    router.py           # Cheap (gpt-4o-mini) vs strong (gpt-4o) routing on evidence complexity
    mock_client.py      # Deterministic, grounded mock so the system runs without an API key
  agents/
    grounding.py        # Hard grounding gate over LLM-cited fields
    root_cause_agent.py # Orchestrates evidence + retrieval + LLM + grounding + calibration
  retrieval/            # Structured-filter + TF-IDF hybrid retrieval and pattern aggregator
  clustering/           # Denial clusters + LLM-narrated manager brief
  eval/                 # Metrics + harness + Markdown / JSON report rendering
  cli.py                # Typer CLI: analyze / cluster / eval / synth / env

prompts/                # System prompts for root-cause, cluster summary, synthetic gen
data/reference/         # Curated CARC / RARC / payer-filing / dx-procedure JSON tables
data/synthetic/         # Generated claims.jsonl (gitignored - run `gabeo synth`)
demo/streamlit_app.py   # 4-tab Streamlit demo: Analyze, Batch brief, Eval, Brief samples
docs/                   # Architecture, design decisions, eval results, batch brief
scripts/                # generate_synthetic.py, run_brief_samples.py
tests/                  # 26 unit + integration tests
```

---

## Why this design

The three differentiating choices, all explained at length in
[`docs/design_decisions.md`](docs/design_decisions.md):

1. **Domain logic stays in Python; the LLM reasons over evidence.** The six
   evidence extractors encode CARC-family rules (timely-filing limits per
   payer, prior-auth requirements, modifier expectations, dx/procedure
   compatibility, duplicate vs bilateral, missing fields per RARC). The LLM
   never invents these — it interprets the evidence and writes the explanation
   a human can act on. This is what gets us the 100% evidence-grounding rate.

2. **Grounding gate.** Every field the LLM cites in its `supporting_evidence`
   is verified against the actual input claim via a closed list of
   `_FIELD_ACCESSORS`. Mismatches trigger a corrective re-prompt; if the second
   attempt still fails, recoverability is downgraded to `needs_review`. No
   silent hallucination.

3. **Calibrated confidence.** The reported confidence is a 50/50 blend of the
   LLM's self-reported score and the deterministic-evidence pass rate, exposed
   in `confidence_components` for auditing. Brier score on the eval set is
   ~0.13.

---

## Reproducing every output in the PDF

The four sample claims in the assignment PDF
(`CLM-2026-00142`, `CLM-2026-00287`, `CLM-2026-00391`, `CLM-2026-00455`) are
included verbatim in the synthetic dataset. Their verdicts live in
[`docs/brief_samples_run.md`](docs/brief_samples_run.md), regenerated by:

```bash
python scripts/run_brief_samples.py
```

Each verdict shows the recoverability label, the calibrated confidence with
its components, the recommended action, the LLM's grounded citations, and the
deterministic evidence trail.

---

## Make targets

```
make install   # uv venv + dependencies + editable install
make test      # pytest -q (26 tests)
make lint      # ruff check
make synth     # regenerate data/synthetic/claims.jsonl
make analyze   # gabeo analyze (all denied claims)
make eval      # gabeo eval -> docs/eval_results.{md,json}
make cluster   # gabeo cluster -> docs/batch_brief.{md,json}
make demo      # streamlit run demo/streamlit_app.py
make clean     # remove generated artifacts
```
