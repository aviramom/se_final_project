"""Metric abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class Metric(ABC):
    """Computes evaluation scores from raw predictions and targets.

    Must not know which model produced the predictions or which benchmark
    supplied the targets — it receives only the strings.

    compute() returns a dict so a single Metric subclass can report multiple
    related scores in one pass (e.g., MCQMetrics returns accuracy, F1, recall,
    precision, and per-class breakdowns together).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'mcq_metrics'."""
        ...

    @property
    @abstractmethod
    def applicable_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        ...

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> dict[str, float]:
        """Return a flat dict of metric_name → score.

        Both lists must be the same length.  Both predictions and targets are
        raw text strings exactly as produced/stored by the model and dataset.
        Metric subclasses are responsible for any parsing needed (e.g., letter
        extraction for MCQ).
        """
        ...
