"""
Post-hoc validation of LLM-extracted findings.

Two kinds of checks:
1. Schema-level (already handled by Pydantic in models.py at parse time).
2. Semantic/business-level (here): confidence thresholds, evidence
   grounding against the source note, and duplicate detection.

Findings that fail a hard rule are marked kept=False but are *not*
discarded from the log -- we always log what the model said, and
separately track what we decided to keep.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import ClinicalFinding, ValidatedFinding, ValidationIssue

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
GROUNDING_SIMILARITY_THRESHOLD = 0.5

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for", "is",
    "was", "with", "no", "not", "as", "by", "this", "that", "it", "be",
    "has", "have", "had", "patient", "note", "today", "reports", "history",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _longest_match_ratio(needle: str, haystack: str) -> float:
    """Longest contiguous overlap between needle and haystack, as a fraction
    of needle's length. Good at catching near-verbatim quotes even with
    minor whitespace/casing drift."""
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return match.size / len(needle) if needle else 0.0


def _grounding_score(evidence: str, source_text: str) -> float:
    """Combine two signals so a claim is only considered grounded if it's
    either (a) a genuine near-verbatim excerpt of the note, or (b) built
    from words that actually appear in the note -- not just sharing a
    common phrase like 'patient has' with otherwise unrelated content."""
    evidence_l = evidence.strip().lower()
    source_l = source_text.lower()
    if not evidence_l:
        return 0.0
    if evidence_l in source_l:
        return 1.0

    longest_ratio = _longest_match_ratio(evidence_l, source_l)

    ev_words = _significant_words(evidence)
    src_words = _significant_words(source_text)
    if ev_words:
        word_overlap = len(ev_words & src_words) / len(ev_words)
    else:
        word_overlap = longest_ratio  # nothing significant to check; fall back

    return max(longest_ratio, word_overlap)


def validate_finding(
    finding: ClinicalFinding,
    index: int,
    source_text: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    grounding_threshold: float = GROUNDING_SIMILARITY_THRESHOLD,
) -> ValidatedFinding:
    issues: list[ValidationIssue] = []

    if finding.confidence < confidence_threshold:
        issues.append(
            ValidationIssue(
                finding_index=index,
                issue_type="low_confidence",
                detail=f"confidence {finding.confidence:.2f} < threshold {confidence_threshold:.2f}",
                severity="warning",
            )
        )

    grounding_score = _grounding_score(finding.evidence, source_text)
    if grounding_score < grounding_threshold:
        issues.append(
            ValidationIssue(
                finding_index=index,
                issue_type="evidence_not_grounded",
                detail=f"evidence text has low similarity ({grounding_score:.2f}) to source note",
                severity="error",
            )
        )

    has_error = any(i.severity == "error" for i in issues)
    kept = not has_error and finding.confidence >= confidence_threshold

    return ValidatedFinding(finding=finding, kept=kept, issues=issues)


def validate_all(
    findings: list[ClinicalFinding],
    source_text: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[ValidatedFinding]:
    validated = [
        validate_finding(f, i, source_text, confidence_threshold=confidence_threshold)
        for i, f in enumerate(findings)
    ]
    _flag_duplicates(validated)
    return validated


def _dedup_rank(vf: ValidatedFinding) -> tuple[int, float]:
    """Ranking for picking the survivor among duplicates.

    Validation outcome outranks confidence. Ranking on confidence alone lets a
    finding that already failed grounding beat a grounded one purely by being
    more confident -- which drops the good finding and keeps nothing, since the
    winner stays kept=False.
    """
    return (1 if vf.kept else 0, vf.finding.confidence)


def _flag_duplicates(validated: list[ValidatedFinding]) -> None:
    """Mark duplicate conditions (case-insensitive exact match on the
    normalized condition string) -- keep the one that passed validation,
    breaking ties on confidence."""
    seen: dict[str, int] = {}  # normalized condition -> index of best kept so far
    for i, vf in enumerate(validated):
        key = vf.finding.condition.strip().lower()
        if key in seen:
            prev_idx = seen[key]
            prev = validated[prev_idx]
            loser_idx, winner_idx = (
                (prev_idx, i) if _dedup_rank(vf) > _dedup_rank(prev) else (i, prev_idx)
            )
            validated[loser_idx].kept = False
            validated[loser_idx].issues.append(
                ValidationIssue(
                    finding_index=loser_idx,
                    issue_type="duplicate",
                    detail=f"duplicate of finding at index {winner_idx} ('{key}'); weaker copy dropped",
                    severity="warning",
                )
            )
            seen[key] = winner_idx
        else:
            seen[key] = i


def flagged_findings_text(validated: list[ValidatedFinding]) -> str:
    """Render low-confidence/error findings as text for a retry prompt."""
    lines = []
    for vf in validated:
        if not vf.issues:
            continue
        issue_summary = "; ".join(f"{i.issue_type} ({i.detail})" for i in vf.issues)
        lines.append(f"- {vf.finding.condition} [confidence={vf.finding.confidence}]: {issue_summary}")
    return "\n".join(lines) if lines else "(none)"
