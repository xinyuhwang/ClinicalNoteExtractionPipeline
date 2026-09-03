"""
CLI:

    python -m clinical_pipeline.cli extract "some note text" --mode ensemble
    python -m clinical_pipeline.cli extract-file note.txt --mode retry
    python -m clinical_pipeline.cli stats
    python -m clinical_pipeline.cli demo          # uses MockLLMClient, no API key needed
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)


def _get_pipeline(use_mock: bool, db_path: str):
    from .logging_store import LogStore
    from .pipeline import ClinicalPipeline

    if use_mock:
        from .mock_llm import MockLLMClient
        client = MockLLMClient()
    else:
        from .llm_client import LLMClient
        client = LLMClient()
    return ClinicalPipeline(client, log_store=LogStore(db_path))


@app.command()
def extract(
    text: str,
    mode: str = typer.Option("single", help="single | retry | ensemble | critique"),
    prompt_version: str = typer.Option("v1"),
    mock: bool = typer.Option(False, help="Use the offline mock LLM instead of a real API call"),
    db: str = typer.Option("clinical_pipeline_log.db"),
):
    pipeline = _get_pipeline(mock, db)
    result = pipeline.process_note(text, mode=mode, prompt_version=prompt_version)
    _print_result(result)


@app.command("extract-file")
def extract_file(
    path: Path,
    mode: str = typer.Option("single"),
    prompt_version: str = typer.Option("v1"),
    mock: bool = typer.Option(False),
    db: str = typer.Option("clinical_pipeline_log.db"),
):
    pipeline = _get_pipeline(mock, db)
    text = path.read_text()
    result = pipeline.process_note(text, mode=mode, prompt_version=prompt_version)
    _print_result(result)


@app.command()
def stats(db: str = typer.Option("clinical_pipeline_log.db")):
    from .logging_store import LogStore
    store = LogStore(db)
    typer.echo(json.dumps(store.stats(), indent=2))


@app.command()
def demo():
    """End-to-end demo using the offline mock LLM -- no API key needed."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    from sample_notes import SAMPLE_NOTES

    pipeline = _get_pipeline(use_mock=True, db_path="demo_log.db")
    for name, text in SAMPLE_NOTES.items():
        typer.echo(f"\n=== {name} ===")
        result = pipeline.process_note(text, mode="single")
        _print_result(result)
    typer.echo("\n=== stats ===")
    typer.echo(json.dumps(pipeline.log_store.stats(), indent=2))


def _print_result(result):
    typer.echo(f"mode={result.mode} model={result.model} latency_ms={result.latency_ms:.1f}")
    typer.echo(f"summary: {result.note_summary}")
    for vf in result.validated:
        f = vf.finding
        flag = "KEPT" if vf.kept else "DROPPED"
        typer.echo(f"  [{flag}] {f.condition} (conf={f.confidence}, negated={f.negated}) -- {f.evidence[:60]}...")
        for issue in vf.issues:
            typer.echo(f"      issue: {issue.issue_type} - {issue.detail}")


if __name__ == "__main__":
    app()
