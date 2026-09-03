# Clinical Note Extraction Pipeline
> Automatically extract structured clinical information from unstructured clinical notes using an LLM-based extraction pipeline.
 
* **Input:** Unstructured clinical note
* **Output:** Structured clinical findings (JSON), validated and logged
## Table of Contents
 
1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Requirements](#3-goals--requirements)
4. [System Architecture](#4-system-architecture)
5. [Pipeline](#5-pipeline)
6. [Clinical Information Schema](#6-clinical-information-schema)
7. [Extraction Methodology](#7-extraction-methodology)
8. [Data Processing](#8-data-processing)
9. [Evaluation](#9-evaluation)
10. [Error Analysis](#10-error-analysis)
11. [Example](#11-example)
12. [Technology Stack](#12-technology-stack)
13. [Project Structure](#13-project-structure)
14. [Running the Pipeline](#14-running-the-pipeline)
15. [Limitations & Safety Considerations](#15-limitations--safety-considerations)
16. [Future Improvements](#16-future-improvements)
17. [Results](#17-results)
---

## 1. Overview
This project converts unstructured clinical notes into structured, validated clinical findings. The pipeline treats extraction as one stage in a larger process: prompt → LLM → schema validation → business validation → orchestration → structured logging.
### Key Capabilities
 
- Clinical finding extraction (condition + supporting evidence)
- Negation detection ("no evidence of pneumonia" vs. "pneumonia")
- Confidence scoring per finding
- Severity tagging where stated
- Multiple extraction strategies: single-call, retry-on-failure, ensemble
  voting, and self-critique
- Evidence-grounding validation (catches hallucinated findings)
- Duplicate detection
- Full structured audit logging (SQLite) of every extraction decision
---
 
## 2. Problem Statement
 
Clinical notes contain valuable information about a patient's symptoms,
diagnoses, medications, procedures, and clinical history. However, this
information is usually recorded as free-form text, which makes it
difficult to analyze programmatically.
 
### Objective
 
Build a reliable extraction pipeline that transforms unstructured clinical
notes into structured representations suitable for downstream
applications such as:
 
- Clinical decision support
- Patient timeline construction
- Medical information retrieval
- Clinical analytics
- Healthcare AI applications
---
 
## 3. Goals & Requirements
 
**Goals**
 
- Convert free-text clinical notes into a fixed, validated schema.
- Never silently trust the model — every extracted claim is checked
  against the source text before being marked "kept."
- Support multiple extraction strategies (single call, retry, ensemble,
  self-critique) behind one interface, so strategy is a runtime choice,
  not a rewrite.
- Log every decision, not just the final output, so failure modes are
  diagnosable after the fact.
**Non-goals**
 
- HIPAA/PHI-compliant storage or transmission as shipped.
- Real-time/streaming extraction.
- Multi-tenant access control.
- Clinical-grade accuracy validation against a labeled gold-standard
  dataset (see [Section 9](#9-evaluation) for what evaluation currently
  does and doesn't cover).
**Functional requirements**
 
| Requirement | Addressed by |
|---|---|
| Extract discrete clinical findings from free text | `prompts.py`, `llm_client.py` |
| Represent findings as validated, typed data | `models.py` (Pydantic) |
| Reject or flag ungrounded / low-confidence claims | `validation.py` |
| Support cross-checking extraction via multiple calls | `orchestration.py` |
| Record a full audit trail of every run | `logging_store.py` |
 
---
 
## 4. System Architecture
 
```text
                Clinical Note
                     │
                     ▼
             ┌───────────────┐
             │ Prompt Builder │   prompts.py
             └───────┬───────┘
                     │  versioned prompt string
                     ▼
             ┌───────────────┐
             │   LLM Client   │   llm_client.py / mock_llm.py
             │ (tool-use call)│
             └───────┬───────┘
                     │  raw JSON matching schema
                     ▼
             ┌───────────────┐
             │Pydantic Models │   models.py
             │(schema checks) │
             └───────┬───────┘
                     │  typed ExtractionResult
                     ▼
             ┌───────────────┐
             │ Orchestration  │   orchestration.py
             │ single/retry/  │
             │ ensemble/      │
             │ critique       │
             └───────┬───────┘
                     │  candidate findings
                     ▼
             ┌───────────────┐
             │   Business     │   validation.py
             │  Validation    │
             │(confidence,    │
             │ grounding,     │
             │ dedup)         │
             └───────┬───────┘
                     │  ValidatedFinding[]
                     ▼
             ┌───────────────┐
             │Structured Log  │   logging_store.py
             │   (SQLite)     │
             └───────────────┘
```
 
Each stage depends only on the **data shape** of the stage before it (a
Pydantic model or plain dataclass), never on a concrete implementation.
This is what makes `MockLLMClient` and the real `LLMClient` interchangeable
throughout the pipeline, and what makes every stage independently
testable without a live API key.
 
---
 
## 5. Pipeline
 
### Step 1 — Prompt Construction
 
Prompts are versioned templates (`v1`, `v2_strict`, `retry_low_confidence`)
rather than strings built inline at the call site. The prompt version used
is logged with every run, so comparing prompt variants is a database
query, not a code-diffing exercise. The system prompt explicitly instructs
the model to extract **negated** findings rather than omit them, since a
missing finding and an explicitly-ruled-out finding are different facts.
 
### Step 2 — Structured Extraction
 
The note is sent to the LLM with a forced tool-use call: the
`ExtractionResult` Pydantic schema is converted to a JSON Schema and passed
as a tool the model *must* call. This avoids the common failure mode of
asking a model to "return JSON" in free text and then parsing markdown
fences or trailing commentary out of the response.
 
### Step 3 — Schema Validation
 
The raw tool-call output is parsed into `ClinicalFinding` /
`ExtractionResult` objects. This layer only checks *shape*: types, numeric
bounds (`confidence` in `[0, 1]`), non-blank strings. It does not judge
whether a claim is actually true.
 
### Step 4 — Orchestration (strategy-dependent)
 
Depending on `mode`, the pipeline either stops here (`single`), re-prompts
with specific feedback (`retry`), runs multiple calls and merges by
majority vote (`ensemble`), or runs a second self-critique call
(`critique`). See [Section 7](#7-extraction-methodology) for detail on
why each exists.
 
### Step 5 — Business Validation
 
Every candidate finding is checked for confidence, evidence grounding in
the source note, and duplication against other findings in the same
result. Findings that fail are marked `kept=False` and retained (not
deleted) along with the specific issue raised.
 
### Step 6 — Structured Logging
 
The full run — model, prompt version, mode, latency, every finding (kept
or not), and every validation issue — is written to SQLite in a single
transaction.
 
---
 
## 6. Clinical Information Schema
 
### Implemented schema
 
The current implementation uses a single, unified finding schema rather
than separate typed categories per concept type:
 
```python
class ClinicalFinding(BaseModel):
    condition: str        # the medical condition or finding
    evidence: str          # supporting quote from the note
    confidence: float      # 0.0 - 1.0
    severity: Severity     # low | moderate | high | unknown
    negated: bool           # True if the note rules this OUT
```
 
```json
{
  "findings": [
    {
      "condition": "Type 2 Diabetes Mellitus",
      "evidence": "history of Type 2 diabetes, diagnosed 2019",
      "confidence": 0.92,
      "severity": "moderate",
      "negated": false
    },
    {
      "condition": "Chest Pain",
      "evidence": "No evidence of chest pain or shortness of breath today",
      "confidence": 0.85,
      "severity": "unknown",
      "negated": true
    }
  ],
  "note_summary": "Routine follow-up; diabetes and hypertension noted, no acute cardiopulmonary symptoms."
}
```
 
This schema was chosen deliberately over a category-specific one
(separate `Medication`, `Procedure`, `LabResult` models) because it keeps
the validation and orchestration logic (Sections 5 and 7) uniform across
every kind of finding — one grounding check, one confidence rule, one
dedup rule, regardless of what's being extracted.
 
### Target / extended schema (not yet implemented)
 
A production system would likely want category-specific structure with
richer per-type attributes, for example:
 
| Category | Examples | Extra attributes a dedicated model would add |
|---|---|---|
| Symptoms | chest pain, nausea, fatigue | duration, severity, quality |
| Diagnoses | hypertension, diabetes | date of diagnosis, status (active/resolved) |
| Medications | lisinopril, metformin | dose, frequency, route, status |
| Procedures | MRI, biopsy | date, findings, ordering provider |
| Lab Results | glucose = 120 mg/dL | value, unit, reference range, flag |
| Vital Signs | BP = 130/80 mmHg | value, unit, timestamp |
| Body Location | chest, abdomen, left knee | laterality, region |
 
Extending `ClinicalFinding` into a tagged union of these per-category
models (still sharing the same `confidence`/`evidence`/`negated`
validation logic) is listed as a concrete next step in
[Section 16](#16-future-improvements).
 
---
 
## 7. Extraction Methodology
 
The pipeline is **LLM-centric with deterministic guardrails** — it does
not currently split responsibilities across separate rule-based NLP
components per concept type (no separate section-splitter, negation
detector, or relation extractor module). Instead, one structured LLM call
produces the full `ExtractionResult`, and deterministic code is
responsible for everything downstream of that call.
 
| Component | Approach | Purpose |
|---|---|---|
| Prompt construction | Rule-based (templates) | Consistent, versioned instructions to the model |
| Concept + negation + severity extraction | LLM (forced structured output) | Flexible language understanding |
| Schema conformance | Deterministic (Pydantic) | Guarantee well-formed output before anything else runs |
| Confidence thresholding | Deterministic (rule-based) | Reject low-confidence claims consistently |
| Evidence grounding | Deterministic (string-similarity heuristic) | Catch hallucinated findings |
| Duplicate resolution | Deterministic (fuzzy string match) | Avoid double-counting the same finding |
| Cross-run consistency | LLM + deterministic merge (ensemble mode) | Surface model disagreement instead of hiding it |
| Self-verification | LLM (critique mode) | Catch subtler errors string-matching can't |
 
### Why this split
 
Using the LLM for language understanding and deterministic code for
everything checkable is a deliberate boundary: anything that *can* be
verified without another model call (is confidence high enough? does this
evidence text actually appear in the note? is this a duplicate?) is
verified deterministically, so the model's judgment is only trusted where
there's no cheaper way to check it.
 
### Evidence grounding, in more detail
 
Grounding uses two signals rather than one:
 
1. **Longest-common-substring ratio** — catches near-verbatim quotes,
   tolerant of minor whitespace/casing drift.
2. **Significant-word overlap** (stopwords stripped) — catches
   fabrications that don't textually resemble the note at all.
A single-signal fuzzy match was tried first and rejected during
development: a fabricated finding scored as "grounded" because its
fabricated evidence opened with the same common phrase ("patient has...")
as the real note, despite describing a condition the note never
mentioned. Combining both signals closes that gap (see
[Section 10](#10-error-analysis)).
 
---
 
## 8. Data Processing
 
### Input handling
 
Notes are passed to the pipeline as plain text. No chunking, section
splitting, or de-identification is performed before the note reaches the
prompt builder — the full note text is included in the extraction prompt
as-is. For very long notes, this means the practical limit is the model's
context window; no truncation or summarization pre-pass is implemented.
 
### What gets persisted
 
- The **note itself is never persisted** by the pipeline. Only a
  truncated SHA-256 hash of the note (`note_hash`) is written to the log,
  specifically so the audit trail doesn't become a second, less-controlled
  copy of patient data.
- **Evidence snippets are persisted**, since they're required to debug
  grounding failures after the fact. This is a real, acknowledged
  trade-off — see [Section 15](#15-limitations--safety-considerations).
### Normalization
 
No terminology normalization (e.g., mapping free-text condition names to
ICD-10 or RxNorm codes) is currently implemented. Extracted `condition`
strings are whatever the model produces (e.g., "Type 2 Diabetes Mellitus"),
not a coded value. This is listed in [Section 16](#16-future-improvements).
 
---
 
## 9. Evaluation
 
### What has been evaluated so far
 
This project's testing to date verifies **pipeline correctness**, not
**extraction accuracy against real clinical data**:
 
- Unit tests (`tests/test_validation.py`) confirm the validation layer
  correctly keeps grounded, high-confidence findings and correctly drops
  low-confidence, ungrounded, and duplicate findings — including an
  adversarial hallucination test case (see [Section 10](#10-error-analysis)).
- Integration tests (`tests/test_pipeline_mock.py`) confirm the full
  pipeline runs end-to-end and that logging/stats aggregation is correct.
- All of the above run against `MockLLMClient`, a deterministic
  keyword-matching stand-in — it validates that the *pipeline* behaves
  correctly, not that a real LLM's *extractions* are accurate.
### What has not been evaluated
 
No evaluation has been run against real clinical notes with real LLM
calls, and no labeled gold-standard dataset exists for this project. The
metrics table below is therefore a **planned evaluation methodology**, not
a report of results — the values are intentionally left unfilled rather
than populated with placeholder numbers presented as real.
 
### Planned metrics
 
| Metric | What it measures |
|---|---|
| Precision / Recall / F1 (per entity type) | Extraction accuracy against annotated notes |
| Exact match | Whole-finding correctness (condition + status + evidence) |
| Entity-level accuracy | Was the concept identified at all |
| Attribute-level accuracy | Were confidence/severity/negation correct |
| Validation pass rate | Fraction of extracted findings that survive business validation (already computed live — see `LogStore.stats()`) |
| Schema validity rate | Fraction of LLM calls returning parseable structured output |
 
The one metric already measured in practice — **validation pass rate** —
comes for free from the logging layer (`pipeline.log_store.stats()`), and
is the closest current proxy for extraction reliability until a labeled
dataset is available.
 
---
## 10. Error Analysis
 
Rather than a report of errors found in a production deployment (none has
occurred — see [Section 9](#9-evaluation)), this section documents the
concrete failure mode found and fixed **during development** of the
validation layer, since it's representative of the class of error this
system is designed to catch.
 
### Case: hallucination that passes a naive grounding check
 
**Setup:** A synthetic adversarial test — a fabricated finding
("Hallucinated Condition") with evidence text sharing no real content with
the source note.
 
**First implementation:** Evidence grounding used a single sliding-window
fuzzy-match score (`difflib.SequenceMatcher` ratio). The fabricated
evidence ("patient has zebra flu and space sickness") scored above the
grounding threshold because it opened with "patient has," the same phrase
the real note happened to start with — despite the rest of the claim being
entirely invented.
 
**Fix:** Replaced the single metric with two independent signals
(longest-common-substring ratio + significant-word overlap with
stopwords removed), and re-ran the adversarial case to confirm it now
fails the check.
 
**Generalization:** Any string-similarity-based grounding check is
vulnerable to this class of error — content that opens with common,
low-information phrasing can inflate a naive similarity score regardless
of whether the substantive claim is supported. This is why grounding
combines two orthogonal signals rather than relying on one, and why the
`critique` orchestration mode exists as a second, semantically-aware
check rather than trusting string similarity alone.
 
### Known failure modes (by design, not yet observed in production use)
 
| Failure mode | Where it would be caught | Current mitigation |
|---|---|---|
| Malformed / non-JSON model output | LLM client boundary | Forced tool-use with schema-derived input |
| Low-confidence but well-formed claim | Business validation | Confidence threshold exclusion |
| Fluent, well-grounded-*sounding* fabrication | Not fully caught | Critique mode helps but is not a guarantee — see [Section 15](#15-limitations--safety-considerations) |
| Model inconsistent across repeated runs | Orchestration (ensemble) | Majority-vote merge; disagreement surfaced, not hidden |
| Silent regression after a prompt change | Logging | Prompt version logged per run |
 
---
 
## 11. Example
 
Using the shipped `MockLLMClient` and sample note `note_1_diabetes_htn`:
 
**Input note:**
 
> "Patient is a 58-year-old male presenting for routine follow-up. History
> notable for Type 2 diabetes, diagnosed 2019, currently managed with
> metformin. Blood pressure elevated at 148/92, consistent with
> hypertension. No evidence of chest pain or shortness of breath today.
> Patient denies fever."
 
**Command:**
 
```bash
python -m clinical_pipeline.cli extract "$(cat note.txt)" --mock --mode single
```
 
**Output:**
 
```
summary: Note mentions 5 tracked finding(s).
  [KEPT] Type 2 Diabetes Mellitus (conf=0.9, negated=False)
  [KEPT] Hypertension (conf=0.9, negated=False)
  [KEPT] Chest Pain (conf=0.85, negated=True)
  [KEPT] Dyspnea (conf=0.85, negated=True)
  [KEPT] Fever (conf=0.85, negated=True)
```
 
Note that the three negated findings (chest pain, dyspnea, fever) are
extracted as present-but-negated rather than silently omitted — this is
the behavior described in [Section 6](#6-clinical-information-schema) and
enforced by the extraction prompt in [Section 5](#5-pipeline).
 
---
 
## 12. Technology Stack
 
| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Structured output / validation | Pydantic v2 |
| LLM | Claude, via the Anthropic API (`anthropic` SDK), forced tool-use for structured extraction |
| Offline testing | Custom `MockLLMClient` (stdlib only, no external dependency) |
| Business validation | Standard library (`difflib`, `re`) — no external NLP dependency |
| Structured logging | SQLite (stdlib `sqlite3`) |
| CLI | Typer |
| Testing | Plain Python assertion scripts (no external test framework required) |
 
---
 
## 13. Project Structure
 
```
clinical-note-pipeline/
├── src/clinical_pipeline/
│   ├── __init__.py          # public API surface
│   ├── models.py            # Pydantic schemas
│   ├── prompts.py           # versioned prompt templates
│   ├── llm_client.py        # real Anthropic client (tool-use)
│   ├── mock_llm.py          # offline stand-in, same interface
│   ├── validation.py        # confidence / grounding / dedup rules
│   ├── orchestration.py     # ensemble voting, critique loop
│   ├── logging_store.py     # SQLite structured logging
│   ├── pipeline.py          # ClinicalPipeline: ties stages together
│   └── cli.py                # Typer CLI
├── examples/sample_notes.py
├── tests/
│   ├── test_validation.py
│   └── test_pipeline_mock.py
├── demo.py                   # end-to-end demo, mock LLM, no API key needed
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```
 
---
 
## 14. Running the Pipeline
 
### Zero-setup demo (no API key)
 
```bash
python demo.py
```
 
Runs all four orchestration modes against sample notes using
`MockLLMClient` — no network access required.
 
### With a real LLM
 
```bash
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY
export $(cat .env | xargs)
```
 
```python
from clinical_pipeline import ClinicalPipeline, LLMClient, LogStore
 
client = LLMClient(model="claude-sonnet-5")
pipeline = ClinicalPipeline(client, log_store=LogStore("clinical_log.db"))
 
result = pipeline.process_note(note_text, mode="ensemble", ensemble_n=3)
 
for finding in result.kept_findings:
    print(finding.condition, finding.confidence, finding.evidence)
```
 
### CLI
 
```bash
python -m clinical_pipeline.cli demo
python -m clinical_pipeline.cli extract "patient reports fever" --mock
python -m clinical_pipeline.cli extract-file note.txt --mode ensemble
python -m clinical_pipeline.cli stats
```
 
### Tests
 
```bash
python tests/test_validation.py
python tests/test_pipeline_mock.py
```
 
---
 
## 15. Limitations & Safety Considerations
 
**This is not a clinical decision-making tool.** It is a reference
architecture for LLM-based structured extraction, demonstrated on a
clinical text example. It has not been evaluated against real patient
data, has no regulatory clearance, and should not inform clinical
decisions in its current form.
 
**Known technical limitations:**
 
- `MockLLMClient` is keyword-matching, not real extraction — it validates
  pipeline plumbing, not accuracy. Any accuracy claim requires evaluation
  with the real `LLMClient` against labeled data, which has not been done
  (see [Section 9](#9-evaluation)).
- Evidence grounding is a string-similarity heuristic, not semantic
  verification — a fluent fabrication that doesn't textually resemble the
  note's phrasing pattern could still pass. The `critique` mode mitigates
  but does not eliminate this.
- Ensemble agreement and dedup similarity thresholds are tuned constants,
  not values calibrated against a labeled dataset.
- No concurrency — ensemble and retry calls run sequentially, which is a
  latency cost at scale.
- Terminology is not normalized to any clinical coding standard
  (ICD-10, RxNorm, SNOMED) — extracted text is free-form.
**PHI / data handling:**
 
- The log store persists a hash of the note, not the note text itself.
- `evidence` snippets — which may contain excerpts of clinical text — are
  persisted in plaintext in the local SQLite database, because they're
  needed to diagnose grounding failures. A real deployment handling actual
  patient data would need encryption at rest, access controls on the log
  store, and likely redaction or truncation of stored evidence, depending
  on applicable compliance requirements (e.g., HIPAA). This system
  demonstrates the *logging pattern*, not a compliant implementation of
  it.
---
 
## 16. Future Improvements
 
- **Extend the schema** from a single unified `ClinicalFinding` into
  category-specific models (medications with dose/frequency/route,
  procedures with dates, lab results with values and reference ranges) as
  outlined in [Section 6](#6-clinical-information-schema), while keeping
  the same validation/orchestration logic across categories.
- **Terminology normalization** — map extracted condition/medication names
  to standard vocabularies (ICD-10, RxNorm) as a post-extraction step.
- **Concurrent ensemble calls** via `asyncio.gather` to reduce ensemble
  mode's latency cost.
- **Real evaluation harness** — build or source a labeled clinical note
  dataset and compute the metrics defined in [Section 9](#9-evaluation)
  against actual `LLMClient` output, not just the mock.
- **Note-type routing** — classify note type (progress note, discharge
  summary, radiology report) before extraction and route to a
  type-specific schema/prompt.
- **Constrained vocabulary validation** — a `field_validator` that checks
  `condition` values against a fixed category list, so the model can't
  invent categories outside a controlled taxonomy.
- **Swap SQLite for Postgres/DuckDB** for concurrent multi-process
  logging — the schema was kept intentionally simple to make this port
  low-risk.
- **Prompt A/B evaluation** — use the two existing prompt versions (`v1`,
  `v2_strict`) with the eval harness above to measure whether the
  stricter prompt actually reduces the `evidence_not_grounded` rate.
---
 
## 17. Results
 
No quantitative extraction-accuracy results exist yet — that requires the
real evaluation harness described in [Section 16](#16-future-improvements),
which requires a labeled dataset not currently part of this project. What
follows are the results that **have** been demonstrated:
 
- **Validation correctly separates trustworthy from untrustworthy
  findings.** In the adversarial test case ([Section 10](#10-error-analysis)),
  a hallucinated finding sharing common phrasing with the real note was
  correctly rejected after the grounding check was corrected to use two
  independent signals instead of one.
- **Ensemble voting correctly surfaces disagreement instead of averaging
  over it.** In a controlled test with 3 simulated runs (2 agreeing on a
  finding, 1 not mentioning it), the majority finding was kept with
  confidence averaged across the agreeing runs, and the minority finding
  was correctly excluded while still being visible in
  `ensemble_agreement` for audit purposes.
- **The full pipeline runs correctly end-to-end**, including logging: a
  demo run across four sample notes and all four orchestration modes
  produced consistent, queryable statistics via `LogStore.stats()`
  (e.g., validation pass rate, average confidence, issue counts by type),
  confirming the observability layer works as designed.
- **Negation extraction works as intended** on the sample notes — findings
  like "no evidence of chest pain" are extracted as present-but-negated
  rather than omitted, which was a specific design requirement from
  [Section 2](#2-problem-statement).
These results validate the **architecture** — that the pipeline's stages
compose correctly and that its validation logic catches the failure modes
it was designed to catch. They do not yet constitute a claim about
real-world extraction accuracy, which remains future work.
