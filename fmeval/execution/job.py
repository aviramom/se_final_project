from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Status of an evaluation job. Inherits str so values serialize without .value."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EvaluationJob:
    """Mutable state for a single evaluation job tracked by EvaluationService."""

    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    status: JobStatus
    created_at: datetime
    max_samples: int
    exp_id: str = ""
    handle: Any = field(default=None, repr=False)  # runner-specific (Future, Slurm ID, …)
    error_message: str | None = None
