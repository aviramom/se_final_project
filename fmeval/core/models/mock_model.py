"""MockModel — always answers 'A)'.

Useful for:
- End-to-end pipeline tests without loading any real weights.
- Establishing a baseline: a model that always picks A) will have predictable
  accuracy equal to the fraction of questions where A is correct.
- Verifying that metrics, result serialization, and breakdown logic are wired
  correctly before a real model is integrated.
"""

from __future__ import annotations

from typing import Any, Literal

from fmeval.core.datasets.formatters import SampleFormatter
from fmeval.core.models.base import ModelWrapper
from fmeval.core.sample import Sample


class MockModel(ModelWrapper):
    """Trivial model that returns 'A)' for every input.

    Parameters
    ----------
    answer:
        The fixed answer letter to always return (default 'A').
        Useful for testing specific answer distributions.
    """

    def __init__(self, answer: str = "A") -> None:
        letter = answer.strip().upper().rstrip(")")
        if letter not in "ABCD" or len(letter) != 1:
            raise ValueError(f"MockModel answer must be one of A/B/C/D, got '{answer}'")
        self._answer = f"{letter})"

    @property
    def model_name(self) -> str:
        return f"mock_always_{self._answer[0].lower()}"

    @property
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["text", "time_series", "multimodal"]

    @property
    def input_mode(self) -> Literal["combined", "separate"]:
        return "combined"

    def format_input(self, sample: Sample) -> str:
        """Serialize TS inline and return the combined text prompt."""
        return SampleFormatter.to_combined(sample).input_text

    def predict(self, inputs: list[Any]) -> list[str]:
        """Return the fixed answer for every input in the batch."""
        return [self._answer] * len(inputs)
