from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmeval.execution.job import EvaluationJob
    from fmeval.storage.repository import EvaluationResult


@dataclass
class EvaluationConfig:
    """Input to EvaluationService.run_evaluation — selects model and benchmark."""

    model_name: str
    benchmark_name: str
    max_samples: int = 50
    exp_id: str = ""  # free-text label; "" → auto-slug applied in service


@dataclass
class DashboardData:
    """Read-path payload shaped for UI rendering of job statuses and result comparisons."""

    jobs: list["EvaluationJob"] = field(default_factory=list)
    results: list["EvaluationResult"] = field(default_factory=list)


@dataclass
class ResultsFilter:
    """Multi-value filter bag passed to EvaluationService.query_results().

    None on any field means "no restriction on that dimension".
    An empty list is equivalent to None (no filter).
    """

    model_names: list[str] | None = None
    benchmark_names: list[str] | None = None
    exp_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


@dataclass
class GroupedResult:
    """Aggregated view of runs sharing the same exp_id × model × benchmark.

    mean_metrics and std_metrics share the same keys. std_metrics values are 0.0
    for groups with only one run.
    """

    exp_id: str
    model_name: str
    benchmark_name: str
    n_runs: int
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    run_timestamps: list[datetime]
