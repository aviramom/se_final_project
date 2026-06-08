"""ModelWrapper abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from fmeval.core.sample import Sample


class ModelWrapper(ABC):
    """Wraps one HuggingFace model family behind a uniform interface.

    Handles model-specific loading, input formatting, and inference.
    Never knows about metrics, benchmark formats, or evaluation logic.

    The combined / separate split:
        input_mode == "combined"  → format_input calls SampleFormatter.to_combined
                                    before further processing; the model sees a single
                                    text string with TS values inlined.
        input_mode == "separate"  → format_input passes the Sample as-is; the model
                                    receives both input_text (with <TS_N> tokens) and
                                    input_ts (raw arrays) and handles fusion itself.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier, e.g. 'llama3-8b'."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        ...

    @property
    @abstractmethod
    def input_mode(self) -> Literal["combined", "separate"]:
        ...

    @abstractmethod
    def format_input(self, sample: Sample) -> Any:
        """Convert one Sample into the model's expected input format.

        Must call SampleFormatter.to_combined or to_separate based on
        self.input_mode before applying any model-specific prompt template.
        Returns a single formatted input (not a batch).
        """
        ...

    @abstractmethod
    def predict(self, inputs: list[Any]) -> list[str]:
        """Run inference on a batch of formatted inputs (output of format_input).

        Always returns list[str] — all models produce text output.
        This is the code that runs on the cluster; in mock mode it runs locally.
        """
        ...
