"""RandomLabelModel — a CPU baseline that guesses a uniformly random class.

Reads the option list from the prompt's ``Return ONLY the label as one of: [...]``
line and returns one option at random.  Two uses:

- A genuine chance-level baseline for classification benchmarks (its balanced
  accuracy hovers around 1 / num_classes), useful for sanity-checking that a real
  model is actually learning from the support set.
- A GPU-free smoke test for the UCR ICL path: unlike the always-"A)" MockModel
  (built for MCQ letters), it emits valid class labels, so it exercises the
  positive branch of ClassificationMetrics' label extraction end-to-end.
"""

from __future__ import annotations

import random
import re
from typing import Any, Literal

from fmeval.core.datasets.formatters import SampleFormatter
from fmeval.core.models.base import ModelWrapper
from fmeval.core.sample import Sample

_OPTIONS_RE = re.compile(r"Return ONLY the label as one of:\s*\[([^\]]+)\]")


class RandomLabelModel(ModelWrapper):
    """Returns a uniformly random option parsed from each prompt.

    Parameters
    ----------
    seed:
        Seed for reproducible choices across runs.
    """

    def __init__(self, seed: int | None = 0) -> None:
        self._rng = random.Random(seed)

    @property
    def model_name(self) -> str:
        return "random_label"

    @property
    def supported_modalities(
        self,
    ) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["text", "time_series", "multimodal"]

    @property
    def input_mode(self) -> Literal["combined", "separate", "image"]:
        return "combined"

    def format_input(self, sample: Sample) -> str:
        """Serialize TS inline and return the combined prompt text."""
        return SampleFormatter.to_combined(sample).input_text

    def predict(self, inputs: list[Any]) -> list[str]:
        """Return a random parsed option per prompt (empty string if none found)."""
        outputs = []
        for prompt in inputs:
            options = self._parse_options(prompt)
            outputs.append(self._rng.choice(options) if options else "")
        return outputs

    @staticmethod
    def _parse_options(prompt: str) -> list[str]:
        match = _OPTIONS_RE.search(prompt)
        if not match:
            return []
        return [opt.strip() for opt in match.group(1).split(",") if opt.strip()]
