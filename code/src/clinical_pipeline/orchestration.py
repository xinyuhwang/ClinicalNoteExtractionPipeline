"""
Orchestration strategies that sit on top of a single LLMClient.extract() call:

- run_ensemble: call the model N times (varying reasoning effort) and keep
  findings that a majority of runs agree on, averaging their confidence.
- run_critique_loop: extract once, then ask the model (or a second model)
  to critique its own output against the source note.

Both are optional layers -- pipeline.py can call the LLM client directly
for the simple "single call" mode.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .models import ClinicalFinding, ExtractionResult
from .prompts import build_critique_prompt

CONDITION_MATCH_THRESHOLD = 0.75

# Sampling parameters were removed from the current Claude models, so ensemble
# runs are varied by reasoning effort instead of by temperature.
DEFAULT_ENSEMBLE_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


@dataclass
class EnsembleOutcome:
    findings: list[ClinicalFinding]
    # label -> {"condition", "negated", "votes", "of", "avg_confidence"}
    agreement: dict
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
                  efforts: Sequence[str] = DEFAULT_ENSEMBLE_EFFORTS,
                  agreement_threshold: float = 0.5) -> EnsembleOutcome:
    """Call the LLM n times and keep findings a majority of *runs* support.

    Runs are varied by reasoning effort, cycling through `efforts`. This
    replaces the temperature sampling this used to rely on: `temperature` is
    rejected with a 400 by the current models. Effort variation is a weaker
    source of disagreement than temperature sampling was, so read a low vote
    count as "this finding is sensitive to how hard the model looked", not as
    a calibrated probability that the finding is wrong.
    """
    runs: list[ExtractionResult] = []
    total_latency_ms = 0.0
    for i in range(n):
        effort = efforts[i % len(efforts)] if efforts else None
        resp = llm_client.extract(note_text, prompt_version=prompt_version, effort=effort)
        runs.append(resp.result)
        total_latency_ms += resp.latency_ms

    # Cluster on (condition, negated), not condition alone: "chest pain" and
    # "no chest pain" are contradictory claims, so merging them would let one
    # confident run silently invert a negation the majority agreed on.
    cond_keys_by_negation: dict[bool, list[str]] = defaultdict(list)
    # (condition_key, negated) -> run index -> that run's best finding for it.
    # Keying by run index enforces one vote per run: a run that happens to list
    # the same finding twice must not out-vote a run that lists it once.
    per_run_best: dict[tuple[str, bool], dict[int, ClinicalFinding]] = defaultdict(dict)

    for run_idx, run in enumerate(runs):
        for f in run.findings:
            known = cond_keys_by_negation[f.negated]
            cond_key = _condition_key(f.condition, known)
            if cond_key not in known:
                known.append(cond_key)
            key = (cond_key, f.negated)
            previous = per_run_best[key].get(run_idx)
            if previous is None or f.confidence > previous.confidence:
                per_run_best[key][run_idx] = f

    merged: list[ClinicalFinding] = []
    agreement: dict = {}
    for (cond_key, negated), by_run in per_run_best.items():
        candidates = list(by_run.values())
        votes = len(candidates)  # distinct runs supporting this claim
        avg_confidence = round(sum(c.confidence for c in candidates) / votes, 3)
        label = f"{cond_key} (negated)" if negated else cond_key
        agreement[label] = {
            "condition": cond_key,
            "negated": negated,
            "votes": votes,
            "of": n,
            "avg_confidence": avg_confidence,
        }
        if votes / n >= agreement_threshold:
            best = max(candidates, key=lambda c: c.confidence)
            merged.append(
                ClinicalFinding(
                    condition=best.condition,
                    evidence=best.evidence,
                    confidence=avg_confidence,
                    severity=best.severity,
                    negated=negated,  # from the bucket, never from `best` alone
                )
            )

    return EnsembleOutcome(findings=merged, agreement=agreement, raw_runs=runs, total_latency_ms=total_latency_ms)


def run_critique_loop(llm_client, note_text: str, result: ExtractionResult) -> str:
    """Ask the model to critique its own (or another run's) extraction."""
    findings_json = json.dumps([f.model_dump(mode="json") for f in result.findings], indent=2)
    prompt = build_critique_prompt(note_text, findings_json)
    return llm_client.critique(prompt)


CRITIQUE_REJECTION_MARKERS = ("unsupported", "hallucin", "not supported", "mismatch")
_SENTENCE_SPLIT_RE = re.compile(r"[.;\n]+")


def critique_rejected_indices(critique_text: str, findings: list[ClinicalFinding]) -> set[int]:
    """Indices of findings the critique rejects.

    A finding counts as rejected only when a rejection marker appears in the
    *same sentence* that names it. Scanning the whole critique for markers
    instead would reject every finding it mentions as soon as it rejects one:
    given "Chest Pain is unsupported. Diabetes is well supported.", an
    unscoped scan drops both.

    This is a text heuristic over free-form prose, so it is deliberately
    conservative -- a critique it cannot parse rejects nothing.
    """
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT_RE.split(critique_text) if s.strip()]
    rejected: set[int] = set()
    for i, finding in enumerate(findings):
        condition = finding.condition.strip().lower()
        if not condition:
            continue
        for sentence in sentences:
            if condition in sentence and any(m in sentence for m in CRITIQUE_REJECTION_MARKERS):
                rejected.add(i)
                break
    return rejected
