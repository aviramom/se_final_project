from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmeval.config.benchmark_registry import BenchmarkInfo, BenchmarkRegistry
    from fmeval.config.model_registry import ModelInfo, ModelRegistry
    from fmeval.execution.job import EvaluationJob
    from fmeval.execution.runner import Runner
    from fmeval.storage.repository import ResultsRepository

from fmeval.services.types import DashboardData, EvaluationConfig


class EvaluationService:
    """Orchestration layer — the only class the UI talks to.

    Owns the evaluation workflow end-to-end and delegates every piece of actual
    work to the layers below. Nothing below this layer ever calls back up into it.
    """

    def __init__(
        self,
        model_registry: "ModelRegistry",
        benchmark_registry: "BenchmarkRegistry",
        runner: "Runner",
        repository: "ResultsRepository",
    ) -> None: ...

    def list_models(self) -> list["ModelInfo"]:
        """Return all models available for selection in the UI."""
        ...

    def list_benchmarks(self) -> list["BenchmarkInfo"]:
        """Return all benchmarks available for selection in the UI."""
        ...

    def run_evaluation(self, config: EvaluationConfig) -> str:
        """Validate config, submit the job, persist it as queued, and return job_id.

        Never blocks — returns immediately after handing the job to the Runner.
        Raises ValueError if the model's modality does not match the benchmark.
        """
        ...

    def poll_jobs(self) -> list["EvaluationJob"]:
        """Check runner status for all in-progress jobs.

        On completion: parse output → compute metrics → persist result.
        On failure: record error message and mark job failed.
        Returns the updated list of all tracked jobs.
        """
        ...

    def get_dashboard_data(self) -> DashboardData:
        """Query the repository and return a DashboardData ready for the UI to render."""
        ...

    def export_csv(self, job_id: str) -> str:
        """Write results for job_id to a CSV file and return the file path."""
        ...
