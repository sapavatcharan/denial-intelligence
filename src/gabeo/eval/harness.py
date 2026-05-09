"""Run the root-cause agent over the gold-labeled set and report metrics.

The harness:
  1. Loads the synthetic dataset + inline gold labels.
  2. Runs the root-cause agent on each denied claim.
  3. Compares the predicted recoverability against the gold label.
  4. Validates each verdict's `supporting_evidence` against the grounding gate.
  5. Reports root-cause accuracy, recoverability F1, evidence-grounding rate,
     calibration (Brier), per-scenario accuracy, total cost & latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.grounding import validate_citations
from ..agents.root_cause_agent import RootCauseAgent
from ..ingest import load_claims_jsonl, load_gold_labels
from ..schemas import Claim, Recoverability, Verdict
from .metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    evidence_grounding_rate,
    label_distribution,
    macro_f1,
    per_scenario_breakdown,
)


@dataclass
class EvalRow:
    claim_id: str
    scenario: str
    expected: str
    predicted: str
    correct: bool
    confidence: float
    grounding_violations: int
    n_evidence: int
    cost_usd: float
    latency_ms: int
    model_used: str | None = None
    keywords_expected: list[str] = field(default_factory=list)
    keywords_present: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    rows: list[EvalRow]
    total_cost_usd: float
    total_latency_ms: int
    accuracy: float
    macro_f1: float
    macro_f1_per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    evidence_grounding_rate: float
    brier_score: float
    keyword_recall: float
    label_distribution: dict[str, int]
    scenario_breakdown: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "n_evaluated": len(self.rows),
                "accuracy": round(self.accuracy, 3),
                "macro_f1": self.macro_f1,
                "evidence_grounding_rate": self.evidence_grounding_rate,
                "brier_score": self.brier_score,
                "keyword_recall": round(self.keyword_recall, 3),
                "total_cost_usd": round(self.total_cost_usd, 5),
                "total_latency_ms": self.total_latency_ms,
                "avg_latency_ms": (
                    round(self.total_latency_ms / len(self.rows), 1) if self.rows else 0.0
                ),
            },
            "label_distribution": self.label_distribution,
            "macro_f1_per_class": self.macro_f1_per_class,
            "confusion_matrix": self.confusion_matrix,
            "scenario_breakdown": self.scenario_breakdown,
            "rows": [r.__dict__ for r in self.rows],
        }


def _keyword_hits(verdict: Verdict, keywords: list[str]) -> list[str]:
    text = " ".join(
        [
            verdict.root_cause,
            verdict.recommended_action,
            verdict.carc_interpretation,
            *(c.why_relevant for c in verdict.supporting_evidence),
            *(e.message for e in verdict.deterministic_evidence),
        ]
    ).lower()
    return [k for k in keywords if k.lower() in text]


def run_eval(
    dataset_path: str | Path = "data/synthetic/claims.jsonl",
    *,
    agent: RootCauseAgent | None = None,
    progress: bool = False,
) -> EvalReport:
    claims = load_claims_jsonl(dataset_path)
    gold = load_gold_labels(dataset_path)
    agent = agent or RootCauseAgent()

    rows: list[EvalRow] = []
    citations_per_claim: list[list[bool]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    y_true_one_hot: list[float] = []
    y_pred_prob: list[float] = []

    denied_claims = [c for c in claims if c.is_denied and c.claim_id in gold]
    for i, claim in enumerate(denied_claims):
        if progress:
            print(f"[{i + 1}/{len(denied_claims)}] {claim.claim_id}", flush=True)
        g = gold[claim.claim_id]
        expected = str(g.get("expected_recoverability") or "needs_review")
        keywords = list(g.get("expected_root_cause_keywords") or [])
        scenario = str(g.get("scenario") or "unknown")
        start = time.perf_counter()
        verdict = agent.analyze(claim)
        elapsed = int((time.perf_counter() - start) * 1000)

        # Per-citation grounding checks for the metric.
        per_citation: list[bool] = []
        for cit in verdict.supporting_evidence:
            v = validate_citations(claim, [cit])
            per_citation.append(not v)
        citations_per_claim.append(per_citation)

        violations = validate_citations(claim, verdict.supporting_evidence)
        keyword_hits = _keyword_hits(verdict, keywords)

        predicted = verdict.recoverability.value
        rows.append(
            EvalRow(
                claim_id=claim.claim_id,
                scenario=scenario,
                expected=expected,
                predicted=predicted,
                correct=expected == predicted,
                confidence=verdict.confidence,
                grounding_violations=len(violations),
                n_evidence=len(verdict.deterministic_evidence),
                cost_usd=verdict.cost_usd or 0.0,
                latency_ms=verdict.latency_ms or elapsed,
                model_used=verdict.model_used,
                keywords_expected=keywords,
                keywords_present=keyword_hits,
            )
        )
        y_true.append(expected)
        y_pred.append(predicted)
        # Calibration: did the verdict match? Treat that as the "outcome" for the predicted-class probability.
        y_true_one_hot.append(1.0 if expected == predicted else 0.0)
        y_pred_prob.append(verdict.confidence)

    f1, per_class = macro_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    grounding = evidence_grounding_rate(citations_per_claim)
    brier = brier_score(y_true_one_hot, y_pred_prob)
    keyword_recall = (
        sum(len(r.keywords_present) for r in rows)
        / max(sum(len(r.keywords_expected) for r in rows), 1)
    )
    return EvalReport(
        rows=rows,
        total_cost_usd=sum(r.cost_usd for r in rows),
        total_latency_ms=sum(r.latency_ms for r in rows),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=f1,
        macro_f1_per_class=per_class,
        confusion_matrix=cm,
        evidence_grounding_rate=grounding,
        brier_score=brier,
        keyword_recall=keyword_recall,
        label_distribution=label_distribution(y_true),
        scenario_breakdown=per_scenario_breakdown(
            [r.scenario for r in rows], [r.correct for r in rows]
        ),
    )


def render_report_markdown(report: EvalReport) -> str:
    s = report.to_dict()["summary"]
    lines: list[str] = [
        "# Eval Results\n",
        "Generated by `gabeo eval`. The harness runs the root-cause agent on every "
        "denied claim in the synthetic dataset and compares the predicted recoverability "
        "against the inline gold label.\n",
        "## Headline metrics\n",
        f"| Metric | Value |\n|---|---|",
        f"| Claims evaluated | {s['n_evaluated']} |",
        f"| Recoverability accuracy | {s['accuracy']:.3f} |",
        f"| Macro F1 (recoverability) | {s['macro_f1']:.3f} |",
        f"| Evidence grounding rate | {s['evidence_grounding_rate']:.3f} |",
        f"| Brier score (lower is better) | {s['brier_score']:.4f} |",
        f"| Root-cause keyword recall | {s['keyword_recall']:.3f} |",
        f"| Total LLM cost (USD) | ${s['total_cost_usd']:.5f} |",
        f"| Avg latency per claim | {s['avg_latency_ms']} ms |",
        "",
        "## Per-class precision / recall / F1\n",
        "| Class | Precision | Recall | F1 |",
        "|---|---|---|---|",
    ]
    for label, m in report.macro_f1_per_class.items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines.append("")
    lines.append("## Confusion matrix (rows = expected, cols = predicted)\n")
    cm = report.confusion_matrix
    classes = sorted(cm.keys())
    lines.append("| | " + " | ".join(classes) + " |")
    lines.append("|---|" + "|".join(["---"] * len(classes)) + "|")
    for t in classes:
        row = [str(cm[t][p]) for p in classes]
        lines.append(f"| **{t}** | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-scenario accuracy\n")
    lines.append("| Scenario | n | correct | accuracy |")
    lines.append("|---|---|---|---|")
    for sc, m in sorted(report.scenario_breakdown.items()):
        lines.append(f"| {sc} | {int(m['n'])} | {int(m['correct'])} | {m['accuracy']:.2f} |")
    lines.append("")
    lines.append("## Per-claim outcomes\n")
    lines.append("| claim_id | scenario | expected | predicted | OK | conf | model |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report.rows:
        ok = "OK" if r.correct else "MISS"
        lines.append(
            f"| {r.claim_id} | {r.scenario} | {r.expected} | {r.predicted} | {ok} | "
            f"{r.confidence:.2f} | {r.model_used or '?'} |"
        )
    return "\n".join(lines)


def render_report_json(report: EvalReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
