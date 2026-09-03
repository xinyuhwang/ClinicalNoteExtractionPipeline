"""
Orchestration strategies that sit on top of a single LLMClient.extract() call:

- run_ensemble: call the model N times (temperature > 0) and keep findings
  that a majority of runs agree on, averaging their confidence.
- run_critique_loop: extract once, then ask the model (or a second model)
  to critique its own output against the source note.

Both are optional layers -- pipeline.py can call the LLM client directly
for the simple "single call" mode.
"""
from __future__ import annotations

import difflib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .models import ClinicalFinding, ExtractionResult
from .prompts import build_critique_prompt

CONDITION_MATCH_THRESHOLD = 0.75


@dataclass
class EnsembleOutcome:
    findings: list[ClinicalFinding]
    agreement: dict  # condition -> {"votes": int, "of": int, "avg_confidence": float}
    raw_runs: list[ExtractionResult] = field(default_factory=list)
    total_latency_ms: float = 0.0


def _condition_key(condition: str, existing_keys: list[str]) -> str:
    """Cluster near-identical condition names (e.g. 'T2DM' vs 'Type 2 Diabetes'
    won't merge -- this only catches case/whitespace/minor phrasing drift)."""
    norm = condition.strip().lower()
    for key in existing_keys:
        if difflib.SequenceMatcher(None, norm, key).ratio() >= CONDITION_MATCH_THRESHOLD:
            return key
    return norm


def run_ensemble(llm_client, note_text: str, n: int = 3, prompt_version: str = "v1",
                  temperature: float = 0.4, agreement_threshold: float = 0.5) -> EnsembleOutcome:
    """Call the LLM n times and keep findings a majority of runs support."""
    runs: list[ExtractionResult] = []
    total_latency_ms = 0.0
    for _ in range(n):
        resp = llm_client.extract(note_text, prompt_version=prompt_version, temperature=temperature)
        runs.append(resp.result)
        total_latency_ms += resp.latency_ms

    buckets: dict[str, list[ClinicalFinding]] = defaultdict(list)
    keys_in_order: list[str] = []
    for run in runs:
        for f in run.findings:
            key = _condition_key(f.condition, keys_in_order)
            if key not in keys_in_order:
                keys_in_order.append(key)
            buckets[key].append(f)

    merged: list[ClinicalFinding] = []
    agreement: dict = {}
    for key, votes in buckets.items():
        vote_frac = len(votes) / n
        agreement[key] = {
            "votes": len(votes),
            "of": n,
            "avg_confidence": round(sum(v.confidence for v in votes) / len(votes), 3),
        }
        if vote_frac >= agreement_threshold:
            best = max(votes, key=lambda v: v.confidence)
            merged.append(
                ClinicalFinding(
                    condition=best.condition,
                    evidence=best.evidence,
                    confidence=round(sum(v.confidence for v in votes) / len(votes), 3),
                    severity=best.severity,
                    negated=best.negated,
                )
            )

    return EnsembleOutcome(findings=merged, agreement=agreement, raw_runs=runs, total_latency_ms=total_latency_ms)


def run_critique_loop(llm_client, note_text: str, result: ExtractionResult) -> str:
    """Ask the model to critique its own (or another run's) extraction."""
    findings_json = json.dumps([f.model_dump(mode="json") for f in result.findings], indent=2)
    prompt = build_critique_prompt(note_text, findings_json)
    return llm_client.critique(prompt)
