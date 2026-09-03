"""
Pydantic schemas for structured clinical note extraction.

Keep these tight: every field the LLM has to fill in is a chance for
malformed or hallucinated output, so we validate hard at the boundary.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class ClinicalFinding(BaseModel):
    """A single condition/finding extracted from a clinical note."""

    condition: str = Field(..., description="The medical condition or finding, e.g. 'Type 2 Diabetes'")
    evidence: str = Field(..., description="Verbatim or near-verbatim quote from the note supporting this finding")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in this finding, 0-1")
    severity: Severity = Field(default=Severity.UNKNOWN, description="Clinical severity if stated or implied")
    negated: bool = Field(default=False, description="True if the note explicitly rules this OUT (e.g. 'no signs of X')")

    @field_validator("condition", "evidence")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)


class ExtractionResult(BaseModel):
    """The full structured output for one note."""

    findings: list[ClinicalFinding] = Field(default_factory=list)
    note_summary: str = Field(default="", description="One-sentence plain summary of the note")


class ValidationIssue(BaseModel):
    """A problem found during post-hoc validation of a finding."""

    finding_index: int
    issue_type: str  # e.g. "low_confidence", "evidence_not_grounded", "duplicate"
    detail: str
    severity: str = "warning"  # "warning" | "error"


class ValidatedFinding(BaseModel):
    """A finding plus the outcome of validation, ready for logging."""

    finding: ClinicalFinding
    kept: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class PipelineRunResult(BaseModel):
    """Everything the pipeline produced for one note, for logging/return to caller."""

    note_hash: str
    mode: str  # "single" | "ensemble" | "critique"
    model: str
    prompt_version: str
    latency_ms: float
    raw_findings_count: int
    validated: list[ValidatedFinding]
    note_summary: str = ""
    critique_notes: Optional[str] = None
    ensemble_agreement: Optional[dict] = None

    @property
    def kept_findings(self) -> list[ClinicalFinding]:
        return [v.finding for v in self.validated if v.kept]
