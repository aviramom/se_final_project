from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.chatts_model import ChatTSModel
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
    registry.register(
        ModelInfo(
            name="chatts-8b",
            display_name="ChatTS-8B (ByteDance Research)",
            modalities=["multimodal"],
        ),
        factory=lambda: ChatTSModel(
            checkpoint_path=os.environ.get(
                "CHATTS_MODEL_PATH", "bytedance-research/ChatTS-8B"
            )
        ),
    )
    return registry
