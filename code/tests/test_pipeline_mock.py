import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from clinical_pipeline import ClinicalPipeline, LogStore, MockLLMClient
from sample_notes import SAMPLE_NOTES


def _fresh_pipeline():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    return ClinicalPipeline(MockLLMClient(jitter_confidence=False), log_store=LogStore(tmp.name))


def test_single_mode_produces_findings_for_note_with_content():
    pipeline = _fresh_pipeline()
    result = pipeline.process_note(SAMPLE_NOTES["note_1_diabetes_htn"], mode="single")
    assert result.raw_findings_count > 0
    assert any(f.condition == "Type 2 Diabetes Mellitus" for f in result.kept_findings)


def test_single_mode_on_sparse_note_yields_no_findings():
    pipeline = _fresh_pipeline()
    result = pipeline.process_note(SAMPLE_NOTES["note_4_sparse"], mode="single")
    assert result.raw_findings_count == 0
    assert result.kept_findings == []


def test_ensemble_mode_runs_and_logs():
    pipeline = _fresh_pipeline()
    result = pipeline.process_note(SAMPLE_NOTES["note_2_respiratory"], mode="ensemble", ensemble_n=3)
    assert result.ensemble_agreement is not None
    assert result.raw_findings_count >= len(result.kept_findings)


def test_logging_persists_and_stats_computable():
    pipeline = _fresh_pipeline()
    for text in SAMPLE_NOTES.values():
        pipeline.process_note(text, mode="single")
    stats = pipeline.log_store.stats()
    assert stats["total_runs"] == len(SAMPLE_NOTES)
    assert stats["validation_pass_rate"] is not None


if __name__ == "__main__":
    test_single_mode_produces_findings_for_note_with_content()
    test_single_mode_on_sparse_note_yields_no_findings()
    test_ensemble_mode_runs_and_logs()
    test_logging_persists_and_stats_computable()
    print("All pipeline tests passed.")
