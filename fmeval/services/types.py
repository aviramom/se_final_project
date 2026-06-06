from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmeval.execution.job import EvaluationJob
    from fmeval.storage.repository import EvaluationResult


@dataclass
class EvaluationConfig:
    """Input to EvaluationService.run_evaluation — selects model and benchmark."""

    model_name: str
    benchmark_name: str


@dataclass
class DashboardData:
    """Read-path payload shaped for UI rendering of job statuses and result comparisons."""

    jobs: list["EvaluationJob"] = field(default_factory=list)
    results: list["EvaluationResult"] = field(default_factory=list)
