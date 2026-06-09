from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd

from fmeval.evaluation.result import SamplePrediction
from fmeval.storage.models import EvaluationResult

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
    max_samples             INTEGER NOT NULL DEFAULT 0
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
                "INSERT OR REPLACE INTO evaluation_results VALUES (?,?,?,?,?,?,?,?,?)",
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
                ),
            )

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
            return "job_id,exp_id,model_name,benchmark_name,modality,metrics,timestamp,execution_time_seconds,max_samples\n"
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
            }
            for r in results
        ]
        return pd.DataFrame(rows).to_csv(index=False)

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
        )
