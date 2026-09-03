"""
Runs the full pipeline offline using the mock LLM -- no API key, no network.
Real usage: swap MockLLMClient() for LLMClient() once you have
ANTHROPIC_API_KEY set (see .env.example).

    python demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "examples"))

from clinical_pipeline import ClinicalPipeline, LogStore, MockLLMClient
from sample_notes import SAMPLE_NOTES


def main():
    client = MockLLMClient()
    log_store = LogStore("demo_log.db")
    pipeline = ClinicalPipeline(client, log_store=log_store)

    print("=" * 70)
    print("1) SINGLE-CALL MODE on every sample note")
    print("=" * 70)
    for name, text in SAMPLE_NOTES.items():
        result = pipeline.process_note(text, mode="single")
        print(f"\n--- {name} ---")
        print(f"summary: {result.note_summary}")
        for vf in result.validated:
            f = vf.finding
            flag = "KEPT   " if vf.kept else "DROPPED"
            print(f"  [{flag}] {f.condition:<30} conf={f.confidence:<5} negated={f.negated}")
            for issue in vf.issues:
                print(f"           issue: {issue.issue_type} - {issue.detail}")

    print("\n" + "=" * 70)
    print("2) ENSEMBLE MODE (3 calls, majority vote) on the respiratory note")
    print("=" * 70)
    result = pipeline.process_note(SAMPLE_NOTES["note_2_respiratory"], mode="ensemble", ensemble_n=3)
    print("agreement per condition:", json.dumps(result.ensemble_agreement, indent=2))

    print("\n" + "=" * 70)
    print("3) CRITIQUE MODE on the headache note")
    print("=" * 70)
    result = pipeline.process_note(SAMPLE_NOTES["note_3_headache"], mode="critique")
    print("critique:", result.critique_notes)

    print("\n" + "=" * 70)
    print("4) RETRY MODE on the diabetes/hypertension note")
    print("=" * 70)
    result = pipeline.process_note(SAMPLE_NOTES["note_1_diabetes_htn"], mode="retry")
    print(f"total latency across calls: {result.latency_ms:.2f}ms")

    print("\n" + "=" * 70)
    print("5) OBSERVABILITY: aggregate stats from the SQLite log")
    print("=" * 70)
    print(json.dumps(log_store.stats(), indent=2))
    print(f"\n(Full audit trail written to {Path('demo_log.db').resolve()})")


if __name__ == "__main__":
    main()
