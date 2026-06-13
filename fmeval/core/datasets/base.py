"""Dataset abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Literal

from fmeval.core.metrics.base import Metric
from fmeval.core.metrics.mcq_metrics import MCQMetrics
from fmeval.core.sample import Sample


class Dataset(ABC):
    """Wraps a benchmark in its original format and yields standardized Samples.

    One subclass per benchmark — adding a new benchmark means adding one file
    here and registering it in the BenchmarkRegistry.  Nothing else changes.

    Never alters the raw benchmark files; all standardization happens inside
    __iter__ in the subclass.
    """

    @property
    @abstractmethod
    def modality(self) -> Literal["text", "time_series", "multimodal"]:
        """Modality tag used by EvaluationService for compatibility checks and
        Metric selection.  Current research targets 'multimodal' exclusively;
        the broader type keeps future pure-text or pure-TS datasets possible."""
        ...

    @property
    def metric(self) -> Metric:
        """Evaluation method for this benchmark.

        Defaults to MCQMetrics — the answer format shared by the multiple-choice
        benchmarks.  A subclass whose answer format differs (e.g. free class
        labels in UCR ICL) overrides this to return the right Metric.  The
        runners read it so the pipeline is never tied to one metric.
        """
        return MCQMetrics()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier stored in results and used as the registry key."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Yield one Sample per benchmark record.  Prefer lazy / streaming reads
        so arbitrarily large datasets do not exhaust memory."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples (used for progress tracking)."""
        ...
