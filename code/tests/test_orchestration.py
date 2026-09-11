"""
Orchestration tests that do NOT go through MockLLMClient's happy path.

The mock only ever produces well-behaved, mutually consistent findings, so it
cannot express the cases ensemble/critique merging actually has to survive:
a run that repeats itself, runs that contradict each other on negation, or a
critique that rejects one finding while endorsing another. These use small
hand-built fake clients instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_pipeline.llm_client import LLMResponse
from clinical_pipeline.models import ClinicalFinding, ExtractionResult
from clinical_pipeline.orchestration import (
    DEFAULT_ENSEMBLE_EFFORTS,
    critique_rejected_indices,
    run_ensemble,
)

NOTE = "Patient has a history of Type 2 diabetes. No evidence of chest pain."


class ScriptedClient:
    """Returns a pre-scripted findings list per call, in order."""

    model = "scripted"

    def __init__(self, scripted_runs: list[list[ClinicalFinding]]):
        self.scripted_runs = scripted_runs
        self.efforts_seen: list = []

    def extract(self, note_text, prompt_version="v1", extra_prompt_kwargs=None, effort=None):
        self.efforts_seen.append(effort)
        findings = self.scripted_runs[len(self.efforts_seen) - 1]
        return LLMResponse(
            result=ExtractionResult(findings=list(findings)),
            raw_json={},
            latency_ms=1.0,
            model=self.model,
        )


def _finding(condition, confidence, negated=False, evidence="chest pain"):
    return ClinicalFinding(
        condition=condition, evidence=evidence, confidence=confidence, negated=negated
    )


def test_repeated_finding_within_one_run_counts_as_one_vote():
    # Run 1 lists Sepsis twice; runs 2 and 3 never mention it. That is 1 of 3
    # runs, not 2 of 3, and must not clear a majority threshold on its own.
    client = ScriptedClient([
        [_finding("Sepsis", 0.9), _finding("Sepsis", 0.9)],
        [],
        [],
    ])
    outcome = run_ensemble(client, NOTE, n=3)
    assert outcome.agreement["sepsis"]["votes"] == 1
    assert [f.condition for f in outcome.findings] == []


def test_majority_negation_is_not_overruled_by_a_confident_minority():
    # Two runs say the note rules chest pain OUT; one confident run says it is
    # present. The merged finding must stay negated.
    client = ScriptedClient([
        [_finding("Chest Pain", 0.70, negated=True)],
        [_finding("Chest Pain", 0.72, negated=True)],
        [_finding("Chest Pain", 0.99, negated=False)],
    ])
    outcome = run_ensemble(client, NOTE, n=3)
    assert [f.negated for f in outcome.findings] == [True]
    # The minority claim is excluded but still visible for audit.
    assert outcome.agreement["chest pain (negated)"]["votes"] == 2
    assert outcome.agreement["chest pain"]["votes"] == 1


def test_merged_confidence_averages_only_supporting_runs():
    client = ScriptedClient([
        [_finding("Asthma", 0.8)],
        [_finding("Asthma", 0.6)],
        [],
    ])
    outcome = run_ensemble(client, NOTE, n=3)
    assert outcome.agreement["asthma"]["avg_confidence"] == 0.7
    assert outcome.findings[0].confidence == 0.7


def test_ensemble_varies_reasoning_effort_across_runs():
    # Sampling params are rejected by current models, so effort is the knob
    # that makes the runs differ. It must actually reach the client.
    client = ScriptedClient([[], [], [], []])
    run_ensemble(client, NOTE, n=4)
    expected = [DEFAULT_ENSEMBLE_EFFORTS[i % len(DEFAULT_ENSEMBLE_EFFORTS)] for i in range(4)]
    assert client.efforts_seen == expected


def test_critique_rejection_is_scoped_to_the_sentence_naming_the_finding():
    findings = [_finding("Chest Pain", 0.9), _finding("Type 2 Diabetes", 0.9)]
    critique = (
        "Chest Pain is unsupported by the note. "
        "Type 2 Diabetes is well supported and accurate."
    )
    assert critique_rejected_indices(critique, findings) == {0}


def test_critique_with_no_rejection_markers_rejects_nothing():
    findings = [_finding("Chest Pain", 0.9)]
    assert critique_rejected_indices("Everything checks out.", findings) == set()
    assert critique_rejected_indices("", findings) == set()


if __name__ == "__main__":
    test_repeated_finding_within_one_run_counts_as_one_vote()
    test_majority_negation_is_not_overruled_by_a_confident_minority()
    test_merged_confidence_averages_only_supporting_runs()
    test_ensemble_varies_reasoning_effort_across_runs()
    test_critique_rejection_is_scoped_to_the_sentence_naming_the_finding()
    test_critique_with_no_rejection_markers_rejects_nothing()
    print("All orchestration tests passed.")
