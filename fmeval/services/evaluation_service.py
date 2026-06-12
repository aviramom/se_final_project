from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from datetime import datetime

from fmeval.config.benchmark_registry import BenchmarkInfo, BenchmarkRegistry
from fmeval.config.model_registry import ModelInfo, ModelRegistry
from fmeval.evaluation.result import SamplePrediction
from fmeval.execution.job import EvaluationJob, JobStatus
from fmeval.execution.mock_runner import MockRunner
from fmeval.execution.runner import Runner
from fmeval.services.types import (
    DashboardData,
    EvaluationConfig,
    GroupedResult,
    ResultsFilter,
)
from fmeval.storage.models import EvaluationResult
from fmeval.storage.repository import ResultsRepository


class EvaluationService:
    """Orchestration layer — the only class the UI talks to.

    Owns the evaluation workflow end-to-end and delegates every piece of actual
    work to the layers below. Nothing below this layer ever calls back up into it.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        benchmark_registry: BenchmarkRegistry,
        runner: Runner,
        repository: ResultsRepository,
    ) -> None:
        self._model_registry = model_registry
        self._benchmark_registry = benchmark_registry
        self._runner = runner
        self._repository = repository
        self._jobs: dict[str, EvaluationJob] = {}  # in-memory, session-scoped

    def list_models(self) -> list[ModelInfo]:
        """Return all models available for selection in the UI."""
        return self._model_registry.list()

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        """Return all benchmarks available for selection in the UI."""
        return self._benchmark_registry.list()

    def run_evaluation(self, config: EvaluationConfig) -> str:
        """Validate config, submit the job, track it, and return job_id.

        Never blocks — returns immediately after handing the job to the Runner.
        Raises ValueError if the model's modality does not match the benchmark.
        """
        model_info = self._model_registry.get_info(config.model_name)
        if model_info.requires_gpu and isinstance(self._runner, MockRunner):
            raise ValueError(
                f"'{model_info.display_name}' requires a GPU and cannot run locally. "
                f"Launch with FMEVAL_RUNNER=slurm, or select a Mock model for local testing."
            )

        model = self._model_registry.get(config.model_name)
        dataset = self._benchmark_registry.get(
            config.benchmark_name, max_samples=config.max_samples
        )

        if dataset.modality not in model.supported_modalities:
            raise ValueError(
                f"Model '{config.model_name}' does not support modality "
                f"'{dataset.modality}' (supports: {model.supported_modalities})."
            )

        exp_id = config.exp_id.strip() or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        job_id = str(uuid.uuid4())[:8]
        job = EvaluationJob(
            job_id=job_id,
            model_name=config.model_name,
            benchmark_name=config.benchmark_name,
            modality=dataset.modality,
            status=JobStatus.RUNNING,
            created_at=datetime.now(),
            max_samples=config.max_samples,
            exp_id=exp_id,
        )
        job.handle = self._runner.submit(job, dataset, model)
        self._jobs[job_id] = job
        return job_id

    def poll_jobs(self) -> list[EvaluationJob]:
        """Check runner status for all non-terminal jobs.

        On completion: persist result to repository.
        On failure: record error message and mark job failed.
        Returns the updated list of all tracked jobs.
        """
        for job in list(self._jobs.values()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                continue
            job.status = self._runner.get_status(job)
            if job.status == JobStatus.COMPLETED:
                self._finalize_job(job)
            elif job.status == JobStatus.FAILED:
                try:
                    self._runner.get_result(job)
                except RuntimeError as exc:
                    job.error_message = str(exc)
        return list(self._jobs.values())

    def _finalize_job(self, job: EvaluationJob) -> None:
        """Retrieve RunResult from runner and persist to repository."""
        try:
            run_result = self._runner.get_result(job)
            exec_time = (datetime.now() - job.created_at).total_seconds()
            self._repository.save(
                EvaluationResult(
                    job_id=job.job_id,
                    model_name=job.model_name,
                    benchmark_name=job.benchmark_name,
                    modality=job.modality,
                    metrics=run_result.metrics,
                    timestamp=run_result.timestamp,
                    execution_time_seconds=exec_time,
                    exp_id=job.exp_id,
                    max_samples=job.max_samples,
                )
            )
            self._repository.save_sample_predictions(
                job.job_id, run_result.sample_predictions
            )
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error_message = f"Finalization error: {exc}"

    def get_run_detail(self, job_id: str) -> list[SamplePrediction]:
        """Return per-sample predictions for a completed run, ordered by sample_idx."""
        return self._repository.get_sample_predictions(job_id)

    def get_dashboard_data(self) -> DashboardData:
        """Query the repository and return a DashboardData ready for the UI to render."""
        return DashboardData(
            jobs=list(self._jobs.values()),
            results=self._repository.query(),
        )

    def list_exp_ids(self) -> list[str]:
        """Return sorted distinct exp_ids stored in the repository."""
        return self._repository.list_exp_ids()

    def query_results(self, filters: ResultsFilter) -> list[EvaluationResult]:
        """Return results matching the given filter.

        Date filters are pushed to the DB; multi-value list filters are applied
        in Python so the repository stays single-value per dimension.
        """
        results = self._repository.query(
            date_from=filters.date_from,
            date_to=filters.date_to,
        )
        if filters.model_names:
            results = [r for r in results if r.model_name in filters.model_names]
        if filters.benchmark_names:
            results = [r for r in results if r.benchmark_name in filters.benchmark_names]
        if filters.exp_ids:
            results = [r for r in results if r.exp_id in filters.exp_ids]
        return results

    def group_results(self, results: list[EvaluationResult]) -> list[GroupedResult]:
        """Aggregate results by (exp_id, model_name, benchmark_name), computing mean ± std.

        Groups with a single run get std = 0.0 for all metrics.
        n_samples and n_unparseable are summed rather than averaged.
        """
        buckets: dict[tuple[str, str, str], list[EvaluationResult]] = defaultdict(list)
        for r in results:
            buckets[(r.exp_id, r.model_name, r.benchmark_name)].append(r)

        SUM_KEYS = {"n_samples", "n_unparseable"}

        grouped: list[GroupedResult] = []
        for (exp_id, model_name, benchmark_name), runs in sorted(buckets.items()):
            all_keys: set[str] = set().union(*(r.metrics.keys() for r in runs))
            mean_metrics: dict[str, float] = {}
            std_metrics: dict[str, float] = {}
            for key in all_keys:
                vals = [r.metrics[key] for r in runs if key in r.metrics]
                if key in SUM_KEYS:
                    mean_metrics[key] = sum(vals)
                    std_metrics[key] = 0.0
                else:
                    mean_metrics[key] = statistics.mean(vals)
                    std_metrics[key] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            grouped.append(
                GroupedResult(
                    exp_id=exp_id,
                    model_name=model_name,
                    benchmark_name=benchmark_name,
                    n_runs=len(runs),
                    mean_metrics=mean_metrics,
                    std_metrics=std_metrics,
                    run_timestamps=[r.timestamp for r in runs],
                )
            )
        return grouped

    def export_csv(self, job_id: str | None = None) -> str:
        """Return CSV string of results. Pass job_id to filter to one run."""
        if job_id:
            result = self._repository.get_by_job_id(job_id)
            results = [result] if result else []
        else:
            results = self._repository.query()
        return self._repository.export_csv(results)
