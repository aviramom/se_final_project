from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationResult:
    """Persisted record of one completed evaluation run."""

    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    metrics: dict[str, float]  # serialized as JSON in SQLite
    timestamp: datetime
    execution_time_seconds: float
    exp_id: str = ""   # experiment label; "" means auto-slug was used
    max_samples: int = 0
    notes: str = ""    # free-text user annotation; editable after the run


@dataclass
class JobRecord:
    """Persisted snapshot of an EvaluationJob.

    Lives in the storage layer (which must not import execution/), so status
    is a plain string and the runner handle is pre-serialized JSON — the
    service maps EvaluationJob <-> JobRecord and the runner owns handle
    (de)serialization.
    """

    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    status: str  # JobStatus.value
    created_at: datetime
    max_samples: int
    exp_id: str = ""
    runner_type: str = "unknown"  # "mock" | "slurm"
    handle_json: str | None = None  # serialized runner handle; None if not serializable
    error_message: str | None = None
