"""
Structured logging to SQLite. Every pipeline run writes:
  - one row in `runs`         (metadata: model, prompt version, latency, mode)
  - N rows in `findings`      (one per extracted finding, kept or not)
  - M rows in `issues`        (one per validation issue raised)

This gives you a queryable audit trail: validation failure rate over
time, confidence distributions, which prompt version performs best, etc.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import PipelineRunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    note_hash TEXT NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    raw_findings_count INTEGER NOT NULL,
    note_summary TEXT,
    critique_notes TEXT,
    ensemble_agreement_json TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    finding_index INTEGER NOT NULL,
    condition TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT,
    negated INTEGER NOT NULL,
    kept INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    finding_index INTEGER NOT NULL,
    issue_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    severity TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_issues_run ON issues(run_id);
"""


def note_hash(note_text: str) -> str:
    return hashlib.sha256(note_text.encode("utf-8")).hexdigest()[:16]


class LogStore:
    def __init__(self, db_path: str | Path = "clinical_pipeline_log.db"):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_run(self, run: PipelineRunResult) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs
                   (note_hash, mode, model, prompt_version, latency_ms,
                    raw_findings_count, note_summary, critique_notes, ensemble_agreement_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.note_hash, run.mode, run.model, run.prompt_version, run.latency_ms,
                    run.raw_findings_count, run.note_summary, run.critique_notes,
                    json.dumps(run.ensemble_agreement) if run.ensemble_agreement else None,
                ),
            )
            run_id = cur.lastrowid

            for vf in run.validated:
                f = vf.finding
                cur2 = conn.execute(
                    """INSERT INTO findings
                       (run_id, finding_index, condition, evidence, confidence, severity, negated, kept)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, vf.issues[0].finding_index if vf.issues else -1,
                     f.condition, f.evidence, f.confidence, f.severity.value, int(f.negated), int(vf.kept)),
                )
                for issue in vf.issues:
                    conn.execute(
                        """INSERT INTO issues (run_id, finding_index, issue_type, detail, severity)
                           VALUES (?, ?, ?, ?, ?)""",
                        (run_id, issue.finding_index, issue.issue_type, issue.detail, issue.severity),
                    )
            return run_id

    # -- observability queries -------------------------------------------------

    def stats(self) -> dict:
        with self._connect() as conn:
            total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            total_findings = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            kept_findings = conn.execute("SELECT COUNT(*) FROM findings WHERE kept=1").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM findings").fetchone()[0]
            avg_latency = conn.execute("SELECT AVG(latency_ms) FROM runs").fetchone()[0]
            by_issue = conn.execute(
                "SELECT issue_type, COUNT(*) FROM issues GROUP BY issue_type ORDER BY 2 DESC"
            ).fetchall()
            by_prompt_version = conn.execute(
                "SELECT prompt_version, COUNT(*), AVG(latency_ms) FROM runs GROUP BY prompt_version"
            ).fetchall()
            return {
                "total_runs": total_runs,
                "total_findings": total_findings,
                "kept_findings": kept_findings,
                "validation_pass_rate": round(kept_findings / total_findings, 3) if total_findings else None,
                "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
                "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
                "issues_by_type": dict(by_issue),
                "runs_by_prompt_version": {
                    row[0]: {"count": row[1], "avg_latency_ms": round(row[2], 1)} for row in by_prompt_version
                },
            }
