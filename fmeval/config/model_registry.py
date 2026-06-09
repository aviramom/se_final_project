from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.mock_model import MockModel


@dataclass
class ModelInfo:
    """UI-facing metadata for a registered model."""

    name: str          # registry key used in EvaluationConfig
    display_name: str  # shown in dropdowns
    modalities: list[str]


class ModelRegistry:
    """Maps model names to factories that produce fresh ModelWrapper instances."""

    def __init__(self) -> None:
        self._registry: dict[str, tuple[ModelInfo, Callable[[], ModelWrapper]]] = {}

    def register(self, info: ModelInfo, factory: Callable[[], ModelWrapper]) -> None:
        """Register a model. factory() is called each time get() is invoked."""
        self._registry[info.name] = (info, factory)

    def list(self) -> list[ModelInfo]:
        return [info for info, _ in self._registry.values()]

    def get(self, name: str) -> ModelWrapper:
        if name not in self._registry:
            available = list(self._registry)
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        _, factory = self._registry[name]
        return factory()


def build_default_model_registry() -> ModelRegistry:
    """Return a registry pre-populated with the models available for the POC."""
    registry = ModelRegistry()
    registry.register(
        ModelInfo(
            name="mock_always_a",
            display_name="Mock Model (always A)",
            modalities=["text", "time_series", "multimodal"],
        ),
        factory=lambda: MockModel("A"),
    )
    registry.register(
        ModelInfo(
            name="mock_always_b",
            display_name="Mock Model (always B)",
            modalities=["text", "time_series", "multimodal"],
        ),
        factory=lambda: MockModel("B"),
    )
    registry.register(
        ModelInfo(
            name="mock_always_c",
            display_name="Mock Model (always C)",
            modalities=["text", "time_series", "multimodal"],
        ),
        factory=lambda: MockModel("C"),
    )
    return registry
