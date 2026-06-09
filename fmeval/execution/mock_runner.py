from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from fmeval.core.datasets.base import Dataset
from fmeval.core.metrics.mcq_metrics import MCQMetrics
from fmeval.core.models.base import ModelWrapper
from fmeval.evaluation.pipeline import LocalEvaluationPipeline
from fmeval.evaluation.result import RunResult
from fmeval.execution.job import EvaluationJob, JobStatus
from fmeval.execution.runner import Runner


class MockRunner(Runner):
    """Runs LocalEvaluationPipeline in a background thread pool.

    Used for local CPU execution during development and demos. Slurm execution
    is a future concern; swap this runner in EvaluationService when ready.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mock_runner"
        )
        self._futures: dict[str, Future[RunResult]] = {}
        self._lock = threading.Lock()

    def submit(self, job: EvaluationJob, dataset: Dataset, model: ModelWrapper) -> Any:
        """Schedule pipeline.run(dataset) in the thread pool; return the Future."""
        pipeline = LocalEvaluationPipeline(model=model, metric=MCQMetrics(), verbose=False)
        future: Future[RunResult] = self._executor.submit(pipeline.run, dataset)
        with self._lock:
            self._futures[job.job_id] = future
        return future

    def get_status(self, job: EvaluationJob) -> JobStatus:
        with self._lock:
            future = self._futures.get(job.job_id)
        if future is None:
            return JobStatus.FAILED
        if future.done():
            return JobStatus.FAILED if future.exception() is not None else JobStatus.COMPLETED
        if future.running():
            return JobStatus.RUNNING
        return JobStatus.QUEUED  # pending in executor queue

    def get_result(self, job: EvaluationJob) -> RunResult:
        with self._lock:
            future = self._futures.get(job.job_id)
        if future is None or not future.done():
            raise RuntimeError(f"Job {job.job_id} is not completed yet.")
        exc = future.exception()
        if exc is not None:
            raise RuntimeError(f"Job {job.job_id} failed: {exc}") from exc
        return future.result()
