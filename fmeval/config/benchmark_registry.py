from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fmeval.core.datasets.base import Dataset
from fmeval.core.datasets.tsexam1 import TimeSeriesExam1Dataset


@dataclass
class BenchmarkInfo:
    """UI-facing metadata for a registered benchmark."""

    name: str          # registry key used in EvaluationConfig
    display_name: str  # shown in dropdowns
    modality: str


class BenchmarkRegistry:
    """Maps benchmark names to factories that produce Dataset instances.

    The factory receives max_samples so the dataset is constructed with the
    right size limit without the registry needing to store config state.
    """

    def __init__(self) -> None:
        self._registry: dict[str, tuple[BenchmarkInfo, Callable[[int | None], Dataset]]] = {}

    def register(
        self,
        info: BenchmarkInfo,
        factory: Callable[[int | None], Dataset],
    ) -> None:
        self._registry[info.name] = (info, factory)

    def list(self) -> list[BenchmarkInfo]:
        return [info for info, _ in self._registry.values()]

    def get(self, name: str, max_samples: int | None = None) -> Dataset:
        if name not in self._registry:
            available = list(self._registry)
            raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
        _, factory = self._registry[name]
        return factory(max_samples)


def build_default_benchmark_registry() -> BenchmarkRegistry:
    """Return a registry pre-populated with the benchmarks available for the POC."""
    registry = BenchmarkRegistry()
    registry.register(
        BenchmarkInfo(
            name="tsexam1",
            display_name="TimeSeriesExam1 (HuggingFace)",
            modality="multimodal",
        ),
        factory=lambda max_s: TimeSeriesExam1Dataset(max_samples=max_s),
    )
    return registry
