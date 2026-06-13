from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd

from fmeval.evaluation.result import SamplePrediction
from fmeval.storage.models import EvaluationResult, JobRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    job_id                  TEXT PRIMARY KEY,
    model_name              TEXT NOT NULL,
    benchmark_name          TEXT NOT NULL,
    modality                TEXT NOT NULL,
    metrics                 TEXT NOT NULL,
    timestamp               TEXT NOT NULL,
    execution_time_seconds  REAL NOT NULL,
    exp_id                  TEXT NOT NULL DEFAULT '',
    max_samples             INTEGER NOT NULL DEFAULT 0,
    notes                   TEXT NOT NULL DEFAULT ''
)
"""

_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    model_name     TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    modality       TEXT NOT NULL,
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    max_samples    INTEGER NOT NULL,
    exp_id         TEXT NOT NULL DEFAULT '',
    runner_type    TEXT NOT NULL,
    handle_json    TEXT,
    error_message  TEXT
)
"""

_SAMPLE_PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sample_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL,
    sample_idx       INTEGER NOT NULL,
    input_text       TEXT NOT NULL DEFAULT '',
    raw_prediction   TEXT NOT NULL,
    raw_target       TEXT NOT NULL,
    predicted_letter TEXT,
    correct_letter   TEXT NOT NULL,
    is_correct       INTEGER NOT NULL,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
)
"""

_SAMPLE_PREDICTIONS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sp_job_id ON sample_predictions(job_id)"
)

_MIGRATIONS = [
    "ALTER TABLE evaluation_results ADD COLUMN exp_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE evaluation_results ADD COLUMN max_samples INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE evaluation_results ADD COLUMN notes TEXT NOT NULL DEFAULT ''",
]


class ResultsRepository:
    """Thin SQLite wrapper for persisting EvaluationResult records.

    Thread-safe: uses check_same_thread=False so Streamlit can call across
    its internal threads without creating a separate connection per thread.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._conn() as conn:
            conn.execute(_SCHEMA)
            conn.execute(_SAMPLE_PREDICTIONS_SCHEMA)
            conn.execute(_SAMPLE_PREDICTIONS_INDEX)
            conn.execute(_JOBS_SCHEMA)
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, result: EvaluationResult) -> None:
        """Upsert a result (INSERT OR REPLACE on job_id primary key)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evaluation_results "
                "(job_id, model_name, benchmark_name, modality, metrics, "
                "timestamp, execution_time_seconds, exp_id, max_samples, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    result.job_id,
                    result.model_name,
                    result.benchmark_name,
                    result.modality,
                    json.dumps(result.metrics),
                    result.timestamp.isoformat(),
                    result.execution_time_seconds,
                    result.exp_id,
                    result.max_samples,
                    result.notes,
                ),
            )

    def update_run(
        self,
        job_id: str,
        *,
        exp_id: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update editable fields of a stored run; metrics stay immutable.

        Only the kwargs passed as non-None are written. Returns True if a
        row was updated, False if the job_id is unknown or nothing to set.
        """
        sets: list[str] = []
        params: list[str] = []
        if exp_id is not None:
            sets.append("exp_id = ?")
            params.append(exp_id)
        if notes is not None:
            sets.append("notes = ?")
            params.append(notes)
        if not sets:
            return False
        params.append(job_id)
        with self._conn() as conn:
            cursor = conn.execute(
                f"UPDATE evaluation_results SET {', '.join(sets)} WHERE job_id = ?",
                params,
            )
        return cursor.rowcount > 0

    def delete_run(self, job_id: str) -> bool:
        """Delete a run and its per-sample predictions in one transaction.

        Returns True if a result row was deleted.
        """
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM sample_predictions WHERE job_id = ?", (job_id,)
            )
            cursor = conn.execute(
                "DELETE FROM evaluation_results WHERE job_id = ?", (job_id,)
            )
        return cursor.rowcount > 0

    def query(
        self,
        model_name: str | None = None,
        benchmark_name: str | None = None,
        exp_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[EvaluationResult]:
        """Return results, optionally filtered by model, benchmark, exp_id, or date range."""
        clauses: list[str] = []
        params: list[str] = []
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if benchmark_name is not None:
            clauses.append("benchmark_name = ?")
            params.append(benchmark_name)
        if exp_id is not None:
            clauses.append("exp_id = ?")
            params.append(exp_id)
        if date_from is not None:
            clauses.append("timestamp >= ?")
            params.append(date_from.isoformat())
        if date_to is not None:
            clauses.append("timestamp <= ?")
            params.append(date_to.isoformat())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM evaluation_results {where} ORDER BY timestamp DESC"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_result(r) for r in rows]

    def get_by_job_id(self, job_id: str) -> EvaluationResult | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_results WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row_to_result(row) if row else None

    def list_exp_ids(self) -> list[str]:
        """Return sorted distinct exp_id values (excluding empty string)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT exp_id FROM evaluation_results WHERE exp_id != '' ORDER BY exp_id"
            ).fetchall()
        return [r[0] for r in rows]

    def export_csv(self, results: list[EvaluationResult]) -> str:
        """Serialize a list of results to a CSV string (no file written)."""
        if not results:
            return "job_id,exp_id,model_name,benchmark_name,modality,metrics,timestamp,execution_time_seconds,max_samples,notes\n"
        rows = [
            {
                "job_id": r.job_id,
                "exp_id": r.exp_id,
                "model_name": r.model_name,
                "benchmark_name": r.benchmark_name,
                "modality": r.modality,
                **r.metrics,
                "timestamp": r.timestamp.isoformat(),
                "execution_time_seconds": r.execution_time_seconds,
                "max_samples": r.max_samples,
                "notes": r.notes,
            }
            for r in results
        ]
        return pd.DataFrame(rows).to_csv(index=False)

    # ------------------------------------------------------------------
    # Jobs table — lets the service rediscover jobs after an app restart.
    # ------------------------------------------------------------------

    def save_job(self, record: JobRecord) -> None:
        """Upsert a job snapshot (INSERT OR REPLACE on job_id primary key)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs "
                "(job_id, model_name, benchmark_name, modality, status, "
                "created_at, max_samples, exp_id, runner_type, handle_json, "
                "error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.job_id,
                    record.model_name,
                    record.benchmark_name,
                    record.modality,
                    record.status,
                    record.created_at.isoformat(),
                    record.max_samples,
                    record.exp_id,
                    record.runner_type,
                    record.handle_json,
                    record.error_message,
                ),
            )

    def update_job_status(
        self, job_id: str, status: str, error_message: str | None = None
    ) -> None:
        """Record a job status transition (called by the service on poll)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ? WHERE job_id = ?",
                (status, error_message, job_id),
            )

    def load_jobs(self, limit: int = 100) -> list[JobRecord]:
        """Return the most recent job snapshots, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            JobRecord(
                job_id=row["job_id"],
                model_name=row["model_name"],
                benchmark_name=row["benchmark_name"],
                modality=row["modality"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                max_samples=row["max_samples"],
                exp_id=row["exp_id"],
                runner_type=row["runner_type"],
                handle_json=row["handle_json"],
                error_message=row["error_message"],
            )
            for row in rows
        ]

    def delete_job(self, job_id: str) -> None:
        """Remove a job snapshot (used when its run is deleted)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def save_sample_predictions(
        self, job_id: str, predictions: list[SamplePrediction]
    ) -> None:
        """Bulk-insert per-sample predictions for a completed job."""
        rows = [
            (
                job_id,
                sp.sample_idx,
                sp.input_text,
                sp.raw_prediction,
                sp.raw_target,
                sp.predicted_letter,
                sp.correct_letter,
                int(sp.is_correct),
                json.dumps(sp.metadata),
            )
            for sp in predictions
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO sample_predictions "
                "(job_id, sample_idx, input_text, raw_prediction, raw_target, "
                "predicted_letter, correct_letter, is_correct, metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def get_sample_predictions(self, job_id: str) -> list[SamplePrediction]:
        """Return all per-sample predictions for a job, ordered by sample_idx."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sample_predictions WHERE job_id = ? ORDER BY sample_idx",
                (job_id,),
            ).fetchall()
        return [
            SamplePrediction(
                sample_idx=row["sample_idx"],
                raw_prediction=row["raw_prediction"],
                raw_target=row["raw_target"],
                predicted_letter=row["predicted_letter"],
                correct_letter=row["correct_letter"],
                is_correct=bool(row["is_correct"]),
                input_text=row["input_text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> EvaluationResult:
        return EvaluationResult(
            job_id=row["job_id"],
            model_name=row["model_name"],
            benchmark_name=row["benchmark_name"],
            modality=row["modality"],
            metrics=json.loads(row["metrics"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            execution_time_seconds=row["execution_time_seconds"],
            exp_id=row["exp_id"],
            max_samples=row["max_samples"],
            notes=row["notes"],
        )
