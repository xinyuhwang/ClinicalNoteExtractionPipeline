"""
A deterministic fake LLM so the pipeline (validation, orchestration,
logging) can be exercised and demoed without network access or an API key.

It uses tiny keyword rules, not real NLP -- it exists purely to feed
realistic-shaped ExtractionResult objects through the rest of the system.
Swap LLMClient for a real one (llm_client.py) when you have API access.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .models import ClinicalFinding, ExtractionResult, Severity
from .llm_client import LLMResponse

_KEYWORD_RULES = [
    ("diabetes", "Type 2 Diabetes Mellitus", Severity.MODERATE),
    ("hypertension", "Hypertension", Severity.MODERATE),
    ("chest pain", "Chest Pain", Severity.HIGH),
    ("shortness of breath", "Dyspnea", Severity.HIGH),
    ("cough", "Cough", Severity.LOW),
    ("fever", "Fever", Severity.MODERATE),
    ("asthma", "Asthma", Severity.MODERATE),
    ("pneumonia", "Pneumonia", Severity.HIGH),
    ("headache", "Headache", Severity.LOW),
]

_NEGATION_MARKERS = ["no evidence of", "denies", "ruled out", "negative for", "without", "no signs of", "no history of"]


class MockLLMClient:
    """Drop-in stand-in for LLMClient with the same .extract() signature."""

    def __init__(self, model: str = "mock-llm-v0", jitter_confidence: bool = True):
        self.model = model
        self.jitter_confidence = jitter_confidence

    def extract(self, note_text: str, prompt_version: str = "v1", extra_prompt_kwargs=None, effort=None) -> LLMResponse:
        start = time.perf_counter()
        text_lower = note_text.lower()
        findings: list[ClinicalFinding] = []

        for keyword, condition, severity in _KEYWORD_RULES:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue

            window_start = max(0, idx - 40)
            snippet = note_text[window_start: idx + len(keyword) + 20].strip()

            negated = any(marker in text_lower[max(0, idx - 30): idx] for marker in _NEGATION_MARKERS)

            base_conf = 0.9 if not negated else 0.85
            if self.jitter_confidence:
                base_conf += random.uniform(-0.15, 0.05)
            base_conf = max(0.05, min(0.99, base_conf))

            findings.append(
                ClinicalFinding(
                    condition=condition,
                    evidence=snippet,
                    confidence=round(base_conf, 3),
                    severity=severity,
                    negated=negated,
                )
            )

        result = ExtractionResult(
            findings=findings,
            note_summary=f"Note mentions {len(findings)} tracked finding(s).",
        )
        latency_ms = (time.perf_counter() - start) * 1000
        raw_json = result.model_dump(mode="json")
        return LLMResponse(result=result, raw_json=raw_json, latency_ms=latency_ms, model=self.model)

    def critique(self, critique_prompt: str) -> str:
        return "Mock critique: all findings appear grounded in provided evidence."
