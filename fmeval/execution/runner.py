from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fmeval.core.datasets.base import Dataset
from fmeval.core.models.base import ModelWrapper
from fmeval.evaluation.result import RunResult
from fmeval.execution.job import EvaluationJob, JobStatus


class Runner(ABC):
    """Abstract execution backend.

    Concrete subclasses (MockRunner, SlurmRunner) own how a job is submitted and
    polled. EvaluationService calls these three methods and never inspects job.handle.
    """

    @abstractmethod
    def submit(self, job: EvaluationJob, dataset: Dataset, model: ModelWrapper) -> Any:
        """Start the job and return a runner-specific handle stored on job.handle."""
        ...

    @abstractmethod
    def get_status(self, job: EvaluationJob) -> JobStatus:
        """Return current status by inspecting job.handle."""
        ...

    @abstractmethod
    def get_result(self, job: EvaluationJob) -> RunResult:
        """Return the completed RunResult.

        Raises RuntimeError if the job is not yet completed or has failed.
        """
        ...
