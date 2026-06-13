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
    COUNT_METRIC_KEYS,
    CategoryBreakdown,
    CategoryBreakdownRow,
    DashboardData,
    EvaluationConfig,
    GroupedResult,
    ResultsFilter,
    RunComparison,
    SampleDiff,
)
from fmeval.storage.models import EvaluationResult, JobRecord
from fmeval.storage.repository import ResultsRepository

# Preferred display order for the dashboard metric selector.
_HEADLINE_METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
]


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
        self._jobs: dict[str, EvaluationJob] = {}  # in-memory; seeded from DB
        self._restore_jobs()

    def _restore_jobs(self) -> None:
        """Reload persisted jobs so an app restart doesn't lose track of them.

        Terminal jobs are restored as-is (handle-less — they're never polled).
        Non-terminal jobs are reattached when the active runner can rebuild
        their handle (Slurm); otherwise they're marked failed as orphaned
        (Mock futures die with the process, and a runner switch invalidates
        any stored handle).
        """
        for record in self._repository.load_jobs():
            try:
                status = JobStatus(record.status)
            except ValueError:
                status = JobStatus.FAILED
            job = EvaluationJob(
                job_id=record.job_id,
                model_name=record.model_name,
                benchmark_name=record.benchmark_name,
                modality=record.modality,
                status=status,
                created_at=record.created_at,
                max_samples=record.max_samples,
                exp_id=record.exp_id,
                error_message=record.error_message,
            )
            if status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                reattachable = (
                    record.runner_type == self._runner.runner_type
                    and record.handle_json is not None
                )
                if reattachable:
                    job.handle = self._runner.deserialize_handle(record.handle_json)  # type: ignore[arg-type]
                if not reattachable or job.handle is None:
                    job.status = JobStatus.FAILED
                    job.error_message = "Orphaned by app restart."
                    self._repository.update_job_status(
                        job.job_id, job.status.value, job.error_message
                    )
            self._jobs[job.job_id] = job

    def _job_to_record(self, job: EvaluationJob) -> JobRecord:
        """Snapshot an EvaluationJob for persistence (handles serialized by the runner)."""
        return JobRecord(
            job_id=job.job_id,
            model_name=job.model_name,
            benchmark_name=job.benchmark_name,
            modality=job.modality,
            status=job.status.value,
            created_at=job.created_at,
            max_samples=job.max_samples,
            exp_id=job.exp_id,
            runner_type=self._runner.runner_type,
            handle_json=self._runner.serialize_handle(job.handle),
            error_message=job.error_message,
        )

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
        dataset_params = config.dataset_params()
        dataset = self._benchmark_registry.get(
            config.benchmark_name,
            max_samples=config.max_samples,
            dataset_params=dataset_params,
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
            dataset_params=dataset_params,
        )
        job.handle = self._runner.submit(job, dataset, model)
        self._jobs[job_id] = job
        self._repository.save_job(self._job_to_record(job))
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
            prev_status = job.status
            job.status = self._runner.get_status(job)
            if job.status == JobStatus.COMPLETED:
                self._finalize_job(job)  # may flip status to FAILED
            elif job.status == JobStatus.FAILED:
                log = self._runner.get_error_log(job)
                job.error_message = log or "Job failed — no error log available."
            if job.status != prev_status:
                self._repository.update_job_status(
                    job.job_id, job.status.value, job.error_message
                )
        return list(self._jobs.values())

    def _finalize_job(self, job: EvaluationJob) -> None:
        """Retrieve RunResult from runner and persist to repository."""
        try:
            run_result = self._runner.get_result(job)
            # For a Slurm job finalized after an app restart this includes the
            # downtime — acceptable POC inaccuracy.
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
            results = [
                r for r in results if r.benchmark_name in filters.benchmark_names
            ]
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

    # ------------------------------------------------------------------
    # Run editing & deletion
    # ------------------------------------------------------------------

    def update_run(
        self,
        job_id: str,
        *,
        exp_id: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Edit a run's label/notes; metrics are immutable by design."""
        updated = self._repository.update_run(job_id, exp_id=exp_id, notes=notes)
        if updated and exp_id is not None:
            # Keep the jobs table and any in-memory job in sync with the new label.
            job = self._jobs.get(job_id)
            if job is not None:
                job.exp_id = exp_id
                self._repository.save_job(self._job_to_record(job))
        return updated

    def delete_run(self, job_id: str) -> bool:
        """Delete a run, its sample predictions, and its job record."""
        deleted = self._repository.delete_run(job_id)
        self._repository.delete_job(job_id)
        self._jobs.pop(job_id, None)
        return deleted

    def delete_runs(self, job_ids: list[str]) -> int:
        """Bulk delete; returns how many stored results were actually removed."""
        return sum(1 for job_id in job_ids if self.delete_run(job_id))

    # ------------------------------------------------------------------
    # Analytics — run-vs-run diff, category breakdowns, metric discovery
    # ------------------------------------------------------------------

    def compare_runs(self, job_id_a: str, job_id_b: str) -> RunComparison:
        """Join two runs' sample predictions on sample_idx and score agreement.

        Raises ValueError if either run is missing or the benchmarks differ —
        comparing answers across different question sets is meaningless.
        """
        result_a = self._repository.get_by_job_id(job_id_a)
        result_b = self._repository.get_by_job_id(job_id_b)
        if result_a is None or result_b is None:
            missing = job_id_a if result_a is None else job_id_b
            raise ValueError(f"Run '{missing}' not found.")
        if result_a.benchmark_name != result_b.benchmark_name:
            raise ValueError(
                f"Cannot compare runs on different benchmarks "
                f"('{result_a.benchmark_name}' vs '{result_b.benchmark_name}')."
            )

        preds_a = {
            p.sample_idx: p for p in self._repository.get_sample_predictions(job_id_a)
        }
        preds_b = {
            p.sample_idx: p for p in self._repository.get_sample_predictions(job_id_b)
        }
        common = sorted(preds_a.keys() & preds_b.keys())

        diffs: list[SampleDiff] = []
        counts = {
            "both_correct": 0,
            "only_a_correct": 0,
            "only_b_correct": 0,
            "both_wrong": 0,
        }
        for idx in common:
            pa, pb = preds_a[idx], preds_b[idx]
            diff = SampleDiff(
                sample_idx=idx,
                input_text=pa.input_text,
                correct_letter=pa.correct_letter,
                raw_target=pa.raw_target,
                predicted_a=pa.predicted_letter,
                predicted_b=pb.predicted_letter,
                raw_prediction_a=pa.raw_prediction,
                raw_prediction_b=pb.raw_prediction,
                a_correct=pa.is_correct,
                b_correct=pb.is_correct,
                metadata=pa.metadata,
            )
            counts[diff.agreement] += 1
            diffs.append(diff)

        return RunComparison(
            result_a=result_a,
            result_b=result_b,
            n_common=len(common),
            both_correct=counts["both_correct"],
            only_a_correct=counts["only_a_correct"],
            only_b_correct=counts["only_b_correct"],
            both_wrong=counts["both_wrong"],
            diffs=diffs,
        )

    def list_metadata_keys(self, job_id: str) -> list[str]:
        """Return sorted union of metadata keys across a run's sample predictions."""
        keys: set[str] = set()
        for pred in self._repository.get_sample_predictions(job_id):
            keys.update(pred.metadata.keys())
        return sorted(keys)

    def get_category_breakdown(self, job_id: str, key: str) -> CategoryBreakdown:
        """Slice a run's accuracy by one sample-metadata key (e.g. 'difficulty').

        Samples lacking the key are skipped — partial metadata coverage is
        common across benchmarks.
        """
        result = self._repository.get_by_job_id(job_id)
        buckets: dict[str, list[bool]] = defaultdict(list)
        for pred in self._repository.get_sample_predictions(job_id):
            if key in pred.metadata:
                buckets[str(pred.metadata[key])].append(pred.is_correct)
        rows = [
            CategoryBreakdownRow(
                value=value,
                n_samples=len(outcomes),
                n_correct=sum(outcomes),
                accuracy=sum(outcomes) / len(outcomes),
            )
            for value, outcomes in sorted(buckets.items())
        ]
        return CategoryBreakdown(
            job_id=job_id,
            exp_id=result.exp_id if result else "",
            model_name=result.model_name if result else "",
            key=key,
            rows=rows,
        )

    def get_category_breakdowns(
        self, job_ids: list[str], key: str
    ) -> list[CategoryBreakdown]:
        """Per-category slices for several runs; runs lacking the key yield no rows."""
        breakdowns = [self.get_category_breakdown(job_id, key) for job_id in job_ids]
        return [b for b in breakdowns if b.rows]

    def list_metric_keys(self, results: list[EvaluationResult]) -> list[str]:
        """Sorted union of metric keys across results, excluding raw counts.

        Feeds the dashboard metric selector; headline metrics first, then the rest
        alphabetically (per-class keys like f1_A stay available but don't crowd
        the top of the list).
        """
        keys: set[str] = set()
        for r in results:
            keys.update(r.metrics.keys())
        keys -= COUNT_METRIC_KEYS
        headline = [k for k in _HEADLINE_METRIC_ORDER if k in keys]
        rest = sorted(keys - set(headline))
        return headline + rest
