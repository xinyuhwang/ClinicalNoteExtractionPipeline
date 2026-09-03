"""
The pipeline object. This is the thing you actually call:

    pipeline = ClinicalPipeline(llm_client)
    result = pipeline.process_note(note_text, mode="ensemble")

`mode`:
  - "single":    one LLM call, validate, log.
  - "retry":     one call; if any finding fails validation, re-prompt once
                 with the flagged findings and re-validate.
  - "ensemble":  N calls, majority-vote merge, validate, log.
  - "critique":  one call, then a second LLM call critiques the output;
                 findings the critique flags as unsupported are dropped.
"""
from __future__ import annotations

from typing import Literal, Optional

from .logging_store import LogStore, note_hash
from .models import ExtractionResult, PipelineRunResult
from .orchestration import run_critique_loop, run_ensemble
from .prompts import build_prompt
from .validation import DEFAULT_CONFIDENCE_THRESHOLD, flagged_findings_text, validate_all

Mode = Literal["single", "retry", "ensemble", "critique"]


class ClinicalPipeline:
    def __init__(self, llm_client, log_store: Optional[LogStore] = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.llm_client = llm_client
        self.log_store = log_store or LogStore()
        self.confidence_threshold = confidence_threshold

    def process_note(
        self,
        note_text: str,
        mode: Mode = "single",
        prompt_version: str = "v1",
        ensemble_n: int = 3,
    ) -> PipelineRunResult:
        if mode == "single":
            run = self._run_single(note_text, prompt_version)
        elif mode == "retry":
            run = self._run_with_retry(note_text, prompt_version)
        elif mode == "ensemble":
            run = self._run_ensemble(note_text, prompt_version, ensemble_n)
        elif mode == "critique":
            run = self._run_critique(note_text, prompt_version)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.log_store.log_run(run)
        return run

    # -- mode implementations --------------------------------------------------

    def _run_single(self, note_text: str, prompt_version: str) -> PipelineRunResult:
        resp = self.llm_client.extract(note_text, prompt_version=prompt_version)
        validated = validate_all(resp.result.findings, note_text, self.confidence_threshold)
        return PipelineRunResult(
            note_hash=note_hash(note_text),
            mode="single",
            model=resp.model,
            prompt_version=prompt_version,
            latency_ms=resp.latency_ms,
            raw_findings_count=len(resp.result.findings),
            validated=validated,
            note_summary=resp.result.note_summary,
        )

    def _run_with_retry(self, note_text: str, prompt_version: str) -> PipelineRunResult:
        resp = self.llm_client.extract(note_text, prompt_version=prompt_version)
        validated = validate_all(resp.result.findings, note_text, self.confidence_threshold)
        total_latency = resp.latency_ms

        needs_retry = any(vf.issues for vf in validated)
        if needs_retry:
            flagged = flagged_findings_text(validated)
            retry_resp = self.llm_client.extract(
                note_text,
                prompt_version="retry_low_confidence",
                extra_prompt_kwargs={"flagged_findings": flagged},
            )
            validated = validate_all(retry_resp.result.findings, note_text, self.confidence_threshold)
            total_latency += retry_resp.latency_ms
            resp = retry_resp  # use retry's summary/model for the record

        return PipelineRunResult(
            note_hash=note_hash(note_text),
            mode="retry",
            model=resp.model,
            prompt_version=prompt_version,
            latency_ms=total_latency,
            raw_findings_count=len(resp.result.findings),
            validated=validated,
            note_summary=resp.result.note_summary,
        )

    def _run_ensemble(self, note_text: str, prompt_version: str, n: int) -> PipelineRunResult:
        outcome = run_ensemble(self.llm_client, note_text, n=n, prompt_version=prompt_version)
        validated = validate_all(outcome.findings, note_text, self.confidence_threshold)
        summary = outcome.raw_runs[0].note_summary if outcome.raw_runs else ""
        model = getattr(self.llm_client, "model", "unknown")
        return PipelineRunResult(
            note_hash=note_hash(note_text),
            mode="ensemble",
            model=model,
            prompt_version=prompt_version,
            latency_ms=outcome.total_latency_ms,
            raw_findings_count=sum(len(r.findings) for r in outcome.raw_runs),
            validated=validated,
            note_summary=summary,
            ensemble_agreement=outcome.agreement,
        )

    def _run_critique(self, note_text: str, prompt_version: str) -> PipelineRunResult:
        resp = self.llm_client.extract(note_text, prompt_version=prompt_version)
        critique_text = run_critique_loop(self.llm_client, note_text, resp.result)

        validated = validate_all(resp.result.findings, note_text, self.confidence_threshold)
        # Lightweight heuristic: if the critique explicitly names a condition
        # as unsupported/hallucinated, drop it even if it passed validation.
        critique_lower = critique_text.lower()
        for vf in validated:
            cond_lower = vf.finding.condition.lower()
            if cond_lower in critique_lower and any(
                bad in critique_lower for bad in ["unsupported", "hallucin", "not supported", "mismatch"]
            ):
                vf.kept = False

        return PipelineRunResult(
            note_hash=note_hash(note_text),
            mode="critique",
            model=resp.model,
            prompt_version=prompt_version,
            latency_ms=resp.latency_ms,
            raw_findings_count=len(resp.result.findings),
            validated=validated,
            note_summary=resp.result.note_summary,
            critique_notes=critique_text,
        )
