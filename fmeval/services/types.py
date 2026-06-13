from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmeval.execution.job import EvaluationJob
    from fmeval.storage.repository import EvaluationResult

# Metric keys that are counts, not rates — excluded from the metric selector
# and never averaged as scores.
COUNT_METRIC_KEYS = {"n_samples", "n_unparseable"}


@dataclass
class EvaluationConfig:
    """Input to EvaluationService.run_evaluation — selects model and benchmark."""

    model_name: str
    benchmark_name: str
    max_samples: int = 50
    exp_id: str = ""  # free-text label; "" → auto-slug applied in service
    # Few-shot ICL options (used only by benchmarks that support them, e.g. UCR);
    # ignored by others. Threaded to the dataset factory as dataset_params.
    num_shots: int = 1
    picking_strategy: str = "random"  # "first" | "random" | "reversed"
    random_seed: int = 0

    def dataset_params(self) -> dict:
        """Benchmark-specific construction hints forwarded to the dataset factory."""
        return {
            "num_shots": self.num_shots,
            "picking_strategy": self.picking_strategy,
            "random_seed": self.random_seed,
        }


@dataclass
class DashboardData:
    """Read-path payload shaped for UI rendering of job statuses and result comparisons."""

    jobs: list["EvaluationJob"] = field(default_factory=list)
    results: list["EvaluationResult"] = field(default_factory=list)


@dataclass
class ResultsFilter:
    """Multi-value filter bag passed to EvaluationService.query_results().

    None on any field means "no restriction on that dimension".
    An empty list is equivalent to None (no filter).
    """

    model_names: list[str] | None = None
    benchmark_names: list[str] | None = None
    exp_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


@dataclass
class GroupedResult:
    """Aggregated view of runs sharing the same exp_id × model × benchmark.

    mean_metrics and std_metrics share the same keys. std_metrics values are 0.0
    for groups with only one run.
    """

    exp_id: str
    model_name: str
    benchmark_name: str
    n_runs: int
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    run_timestamps: list[datetime]


@dataclass
class SampleDiff:
    """One sample present in both runs of a comparison, with both predictions."""

    sample_idx: int
    input_text: str
    correct_letter: str
    raw_target: str
    predicted_a: str | None  # extracted letter, None if unparseable
    predicted_b: str | None
    raw_prediction_a: str
    raw_prediction_b: str
    a_correct: bool
    b_correct: bool
    metadata: dict = field(default_factory=dict)

    @property
    def agreement(self) -> str:
        """Four-way agreement bucket used for grouping/coloring in the UI."""
        if self.a_correct and self.b_correct:
            return "both_correct"
        if self.a_correct:
            return "only_a_correct"
        if self.b_correct:
            return "only_b_correct"
        return "both_wrong"


@dataclass
class RunComparison:
    """Question-by-question comparison of two runs on the same benchmark.

    diffs holds ALL common samples (joined on sample_idx); the UI filters
    to disagreements. Counts are over the common samples only — the two
    runs may have used different max_samples.
    """

    result_a: "EvaluationResult"
    result_b: "EvaluationResult"
    n_common: int
    both_correct: int
    only_a_correct: int
    only_b_correct: int
    both_wrong: int
    diffs: list[SampleDiff] = field(default_factory=list)


@dataclass
class CategoryBreakdownRow:
    """Accuracy within one value of a metadata key (e.g. difficulty='easy')."""

    value: str
    n_samples: int
    n_correct: int
    accuracy: float


@dataclass
class CategoryBreakdown:
    """Per-category accuracy slices of one run, computed from sample metadata."""

    job_id: str
    exp_id: str
    model_name: str
    key: str  # the metadata key sliced on, e.g. "category" or "difficulty"
    rows: list[CategoryBreakdownRow] = field(default_factory=list)
