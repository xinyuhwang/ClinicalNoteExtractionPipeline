import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_pipeline.models import ClinicalFinding, Severity
from clinical_pipeline.validation import validate_all


NOTE = "Patient has a history of Type 2 diabetes, well controlled on metformin."


def test_grounded_high_confidence_finding_is_kept():
    findings = [
        ClinicalFinding(
            condition="Type 2 Diabetes",
            evidence="history of Type 2 diabetes",
            confidence=0.92,
            severity=Severity.MODERATE,
            negated=False,
        )
    ]
    validated = validate_all(findings, NOTE)
    assert validated[0].kept is True
    assert validated[0].issues == []


def test_low_confidence_finding_is_dropped():
    findings = [
        ClinicalFinding(
            condition="Something",
            evidence="well controlled on metformin",
            confidence=0.2,
            severity=Severity.UNKNOWN,
            negated=False,
        )
    ]
    validated = validate_all(findings, NOTE)
    assert validated[0].kept is False
    assert any(i.issue_type == "low_confidence" for i in validated[0].issues)


def test_hallucinated_evidence_is_dropped():
    findings = [
        ClinicalFinding(
            condition="Hallucinated Condition",
            evidence="patient has zebra flu and space sickness",
            confidence=0.8,
            severity=Severity.UNKNOWN,
            negated=False,
        )
    ]
    validated = validate_all(findings, NOTE)
    assert validated[0].kept is False
    assert any(i.issue_type == "evidence_not_grounded" for i in validated[0].issues)


def test_duplicate_keeps_higher_confidence():
    findings = [
        ClinicalFinding(condition="Type 2 Diabetes", evidence="history of Type 2 diabetes", confidence=0.6, severity=Severity.UNKNOWN, negated=False),
        ClinicalFinding(condition="type 2 diabetes", evidence="history of Type 2 diabetes", confidence=0.92, severity=Severity.UNKNOWN, negated=False),
    ]
    validated = validate_all(findings, NOTE)
    assert validated[0].kept is False
    assert validated[1].kept is True
    assert any(i.issue_type == "duplicate" for i in validated[0].issues)


def test_duplicate_prefers_validated_finding_over_more_confident_failure():
    # The ungrounded copy is more confident, but confidence must not let it win
    # the dedup: that would drop the grounded finding and keep nothing.
    findings = [
        ClinicalFinding(condition="Type 2 Diabetes", evidence="history of Type 2 diabetes", confidence=0.6, severity=Severity.UNKNOWN, negated=False),
        ClinicalFinding(condition="type 2 diabetes", evidence="zebra flu and space sickness", confidence=0.95, severity=Severity.UNKNOWN, negated=False),
    ]
    validated = validate_all(findings, NOTE)
    assert validated[0].kept is True
    assert validated[1].kept is False
    assert any(i.issue_type == "evidence_not_grounded" for i in validated[1].issues)
    assert any(i.issue_type == "duplicate" for i in validated[1].issues)


if __name__ == "__main__":
    test_grounded_high_confidence_finding_is_kept()
    test_low_confidence_finding_is_dropped()
    test_hallucinated_evidence_is_dropped()
    test_duplicate_keeps_higher_confidence()
    test_duplicate_prefers_validated_finding_over_more_confident_failure()
    print("All validation tests passed.")
