"""Public surface of the metrics sub-package."""

from fmeval.core.metrics.base import Metric
from fmeval.core.metrics.classification_metrics import (
    ClassificationMetrics,
    extract_label,
)
from fmeval.core.metrics.mcq_metrics import MCQMetrics, extract_letter

__all__ = [
    "Metric",
    "MCQMetrics",
    "extract_letter",
    "ClassificationMetrics",
    "extract_label",
]
