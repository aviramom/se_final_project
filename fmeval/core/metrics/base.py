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

    @abstractmethod
    def label_predictions(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> tuple[list[str | None], list[str | None]]:
        """Extract the canonical answer token from each prediction and target.

        Returns (predicted_labels, true_labels), aligned per sample.  This is
        the per-sample counterpart to compute(): the pipeline uses it to record
        each sample's predicted/correct answer and whether it matched, without
        the pipeline needing to know how a given metric parses an answer (MCQ
        letters, class labels, etc.).  A None entry means the string could not
        be parsed into a valid answer.
        """
        ...
