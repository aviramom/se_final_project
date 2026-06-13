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

    # Stable identifier persisted with each job so a restarted app knows which
    # runner produced a stored handle (and whether it can be reattached).
    runner_type: str = "unknown"

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

    def get_error_log(self, job: EvaluationJob) -> str | None:
        """Return a brief error log snippet for a failed job, or None if unavailable.

        Default implementation returns None. Override in runners that can surface
        stderr output (e.g. SlurmRunner fetches the .err file from the cluster).
        """
        return None

    def serialize_handle(self, handle: Any) -> str | None:
        """Serialize a runner handle to JSON for persistence, or None.

        Default returns None — handles that only make sense inside this
        process (e.g. MockRunner's Futures) cannot survive a restart.
        """
        return None

    def deserialize_handle(self, handle_json: str) -> Any:
        """Rebuild a handle from serialize_handle() output.

        Default returns None; runners that support reattach override both.
        """
        return None
