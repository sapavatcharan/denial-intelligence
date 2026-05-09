"""Typer CLI: `gabeo analyze | cluster | eval | synth`.

Designed so a reviewer can clone the repo and reproduce every output in
under five minutes:

    gabeo synth                     # 40-claim dataset with gold labels
    gabeo analyze --claim-id CLM-2026-00142
    gabeo cluster --in data/synthetic/claims.jsonl --out docs/batch_brief.md
    gabeo eval                      # writes docs/eval_results.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

import typer

from .agents.root_cause_agent import RootCauseAgent, verdict_to_jsonable
from .clustering import build_batch_brief, cluster_denials
from .clustering.batch_intelligence import render_brief_markdown
from .eval.harness import render_report_json, render_report_markdown, run_eval
from .ingest import load_claims_jsonl
from .retrieval import SimilarityIndex

app = typer.Typer(
    name="gabeo",
    help="Gabeo Denial AI - claim denial analysis CLI",
    add_completion=False,
)
console = Console()

DEFAULT_DATA = "data/synthetic/claims.jsonl"


def _load_env() -> None:
    load_dotenv()


@app.callback()
def _root() -> None:
    _load_env()


@app.command()
def synth(
    n: int = typer.Option(40, "--n", help="Number of claims to generate"),
    out: str = typer.Option(DEFAULT_DATA, "--out", help="Output JSONL path"),
    seed: int = typer.Option(7, "--seed", help="RNG seed for reproducibility"),
) -> None:
    """Regenerate the synthetic dataset (40 claims, mix paid/denied with gold labels)."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "generate_synthetic.py"
    subprocess.run(
        [sys.executable, str(script), "--n", str(n), "--out", out, "--seed", str(seed)],
        check=True,
    )


@app.command()
def analyze(
    claim_id: str | None = typer.Option(None, "--claim-id", help="Single claim ID to analyze"),
    in_path: str = typer.Option(DEFAULT_DATA, "--in", help="Dataset path (JSONL)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
    use_history: bool = typer.Option(
        True, "--history/--no-history", help="Pass top-K similar paid claims to the LLM"
    ),
) -> None:
    """Run the root-cause agent on one claim or every denied claim in a dataset."""
    claims = load_claims_jsonl(in_path)
    by_id = {c.claim_id: c for c in claims}
    if claim_id:
        if claim_id not in by_id:
            typer.echo(f"Claim ID {claim_id!r} not found in {in_path}", err=True)
            raise typer.Exit(2)
        targets = [by_id[claim_id]]
    else:
        targets = [c for c in claims if c.is_denied]
    if not targets:
        typer.echo("No denied claims to analyze.", err=True)
        raise typer.Exit(0)

    index = SimilarityIndex(claims) if use_history else None
    agent = RootCauseAgent()
    results = []
    for target in targets:
        similar_ids: list[str] = []
        summary = ""
        if index is not None:
            similar = index.find_similar_paid(target, top_k=5)
            similar_ids = [s.claim_id for s in similar]
            if similar:
                top = similar[0]
                summary = (
                    f"Top similar paid claim: {top.claim_id} ({top.payer_id}, "
                    f"proc {top.procedure}, score {top.score:.2f})."
                )
        verdict = agent.analyze(
            target,
            similar_paid_claim_ids=similar_ids,
            historical_summary=summary,
        )
        results.append(verdict)

    if json_out:
        typer.echo(json.dumps([verdict_to_jsonable(v) for v in results], indent=2))
        return

    table = Table(title=f"Verdicts ({len(results)} claim(s))")
    table.add_column("claim_id", overflow="fold")
    table.add_column("recoverability")
    table.add_column("conf", justify="right")
    table.add_column("model", overflow="fold")
    table.add_column("recommended_action", overflow="fold")
    for v in results:
        table.add_row(
            v.claim_id,
            v.recoverability.value,
            f"{v.confidence:.2f}",
            v.model_used or "?",
            v.recommended_action,
        )
    console.print(table)


@app.command()
def cluster(
    in_path: str = typer.Option(DEFAULT_DATA, "--in", help="Dataset path (JSONL)"),
    out: str = typer.Option("docs/batch_brief.md", "--out", help="Markdown brief output path"),
    json_out: str | None = typer.Option(None, "--json-out", help="Optional JSON output path"),
    top_n: int = typer.Option(10, "--top-n", help="How many clusters get an LLM-written narrative"),
) -> None:
    """Group denied claims into prioritized clusters and write a batch action brief."""
    claims = load_claims_jsonl(in_path)
    clusters = cluster_denials(claims)
    clusters = build_batch_brief(clusters, top_n=top_n)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_brief_markdown(clusters))
    typer.echo(f"Wrote {len(clusters)} clusters to {out}")
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps([c.to_dict() for c in clusters], indent=2))
        typer.echo(f"Wrote JSON to {json_out}")


@app.command()
def eval(  # noqa: A001
    in_path: str = typer.Option(DEFAULT_DATA, "--in", help="Dataset path (JSONL)"),
    md_out: str = typer.Option("docs/eval_results.md", "--md-out", help="Markdown output path"),
    json_out: str | None = typer.Option(None, "--json-out", help="Optional JSON output path"),
    progress: bool = typer.Option(False, "--progress", help="Print per-claim progress"),
) -> None:
    """Run the eval harness on the gold-labeled dataset."""
    report = run_eval(in_path, progress=progress)
    Path(md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(md_out).write_text(render_report_markdown(report))
    typer.echo(f"Wrote {md_out}")
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(render_report_json(report))
        typer.echo(f"Wrote {json_out}")
    s = report.to_dict()["summary"]
    console.rule("Eval summary")
    console.print(s)


@app.command()
def env() -> None:
    """Show effective LLM configuration and which client will be used."""
    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    mock = os.environ.get("GABEO_MOCK_LLM") == "1"
    if mock:
        client = "MockLLMClient (forced via GABEO_MOCK_LLM=1)"
    elif not have_key:
        client = "MockLLMClient (no OPENAI_API_KEY)"
    else:
        client = "LLMClient (OpenAI)"
    typer.echo(f"OPENAI_API_KEY set: {have_key}")
    typer.echo(f"GABEO_LLM_TRIAGE_MODEL: {os.environ.get('GABEO_LLM_TRIAGE_MODEL', 'gpt-4o-mini')}")
    typer.echo(f"GABEO_LLM_STRONG_MODEL: {os.environ.get('GABEO_LLM_STRONG_MODEL', 'gpt-4o')}")
    typer.echo(f"Active client: {client}")


if __name__ == "__main__":
    app()
