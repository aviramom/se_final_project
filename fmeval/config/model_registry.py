from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.chatts_model import ChatTSModel
from fmeval.core.models.mock_model import MockModel
from fmeval.core.models.qwen_vl_model import QwenVLModel
from fmeval.core.models.random_label_model import RandomLabelModel


@dataclass
class ModelInfo:
    """UI-facing metadata for a registered model."""

    name: str          # registry key used in EvaluationConfig
    display_name: str  # shown in dropdowns
    modalities: list[str]
    requires_gpu: bool = False  # True → cannot run under MockRunner (local)


class ModelRegistry:
    """Maps model names to factories that produce fresh ModelWrapper instances."""

    def __init__(self) -> None:
        self._registry: dict[str, tuple[ModelInfo, Callable[[], ModelWrapper]]] = {}

    def register(self, info: ModelInfo, factory: Callable[[], ModelWrapper]) -> None:
        """Register a model. factory() is called each time get() is invoked."""
        self._registry[info.name] = (info, factory)

    def list(self) -> list[ModelInfo]:
        return [info for info, _ in self._registry.values()]

    def get_info(self, name: str) -> ModelInfo:
        if name not in self._registry:
            available = list(self._registry)
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        info, _ = self._registry[name]
        return info

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
            name="random_label",
            display_name="Random Label (chance baseline)",
            modalities=["text", "time_series", "multimodal"],
        ),
        factory=lambda: RandomLabelModel(),
    )
    registry.register(
        ModelInfo(
            name="chatts-8b",
            display_name="ChatTS-8B (ByteDance Research)",
            modalities=["multimodal"],
            requires_gpu=True,
        ),
        factory=lambda: ChatTSModel(
            checkpoint_path=os.environ.get(
                "CHATTS_MODEL_PATH", "bytedance-research/ChatTS-8B"
            )
        ),
    )
    registry.register(
        ModelInfo(
            name="qwen3-vl-8b",
            display_name="Qwen3-VL-8B-Instruct (vision)",
            modalities=["multimodal"],
            requires_gpu=True,
        ),
        factory=lambda: QwenVLModel(
            checkpoint_path=os.environ.get(
                "QWEN_VL_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"
            )
        ),
    )
    return registry
