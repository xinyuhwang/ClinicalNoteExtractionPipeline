"""
Prompt builder. Templates are versioned strings so you can A/B them and
log which version produced which output (see logging_store.py).
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. You read raw "
    "clinical notes and extract discrete findings as structured data. "
    "You never invent findings that are not supported by the text. "
    "If the note explicitly rules a condition out (e.g. 'no evidence of X', "
    "'ruled out Y'), extract it with negated=true rather than omitting it. "
    "Evidence must be a direct or near-direct quote from the note."
)

_TEMPLATES: dict[str, str] = {
    "v1": (
        "Extract all clinical findings from the following note.\n\n"
        "NOTE:\n\"\"\"\n{note_text}\n\"\"\"\n\n"
        "For each finding, provide the condition, a supporting quote as "
        "evidence, a confidence score between 0 and 1, severity if stated, "
        "and whether it is negated. Also provide a one-sentence summary."
    ),
    "v2_strict": (
        "Extract all clinical findings from the following note. Be "
        "conservative: only extract a finding if the note contains "
        "explicit textual support for it. Do not infer conditions from "
        "lab values unless the note names the condition. If uncertain, "
        "lower the confidence score rather than omitting the finding.\n\n"
        "NOTE:\n\"\"\"\n{note_text}\n\"\"\"\n\n"
        "For each finding, provide the condition, a supporting quote as "
        "evidence, a confidence score between 0 and 1, severity if stated, "
        "and whether it is negated. Also provide a one-sentence summary."
    ),
    "retry_low_confidence": (
        "Your previous extraction included findings with low confidence. "
        "Re-examine the note below and either (a) find stronger textual "
        "evidence to raise your confidence, or (b) lower the confidence "
        "further / drop the finding if it genuinely isn't supported.\n\n"
        "NOTE:\n\"\"\"\n{note_text}\n\"\"\"\n\n"
        "Previously flagged findings:\n{flagged_findings}\n\n"
        "Return the FULL corrected list of findings, not just the flagged ones."
    ),
}


def build_prompt(note_text: str, version: str = "v1", **kwargs) -> str:
    """Build a user-turn prompt from a template version."""
    if version not in _TEMPLATES:
        raise ValueError(f"Unknown prompt version '{version}'. Known: {list(_TEMPLATES)}")
    return _TEMPLATES[version].format(note_text=note_text, **kwargs)


def build_critique_prompt(note_text: str, findings_json: str) -> str:
    return (
        "You are reviewing a clinician's automated note extraction for accuracy.\n\n"
        "ORIGINAL NOTE:\n\"\"\"\n{note}\n\"\"\"\n\n"
        "EXTRACTED FINDINGS (JSON):\n{findings}\n\n"
        "For each finding, check whether the 'evidence' text actually appears "
        "in (or is a faithful paraphrase of) the note, and whether the "
        "condition is a reasonable reading of that evidence. Reply with a "
        "short plain-text critique: list any findings that are unsupported, "
        "hallucinated, or mismatched with their evidence. If everything "
        "checks out, say so explicitly."
    ).format(note=note_text, findings=findings_json)
