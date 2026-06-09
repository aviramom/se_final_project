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
