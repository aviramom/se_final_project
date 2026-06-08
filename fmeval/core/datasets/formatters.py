"""SampleFormatter — converts between canonical (separate) and combined forms."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Protocol, runtime_checkable

import numpy as np

from fmeval.core.sample import Sample

_TS_TOKEN_RE = re.compile(r"<TS_(\d+)>")


@runtime_checkable
class TSSerializer(Protocol):
    """Strategy for turning a numpy array into its text representation."""

    def serialize(self, ts: np.ndarray) -> str:
        ...


class DefaultTSSerializer:
    """Renders an array as a bracketed, comma-separated float list.

    Example: [1.2, 3.45, -0.7]
    Precision is capped at 4 decimal places to keep prompts readable.
    """

    def serialize(self, ts: np.ndarray) -> str:
        values = ", ".join(f"{v:.4g}" for v in ts.flatten())
        return f"[{values}]"


class SampleFormatter:
    """Stateless converter between the two forms a model may need.

    Canonical (separate) form — what every Dataset yields:
        input_text  : prompt with <TS_N> placeholders
        input_ts    : list of raw numpy arrays, index-matched to the tokens

    Combined form — what LLMs typically consume:
        input_text  : prompt with placeholders replaced by serialized values
        input_ts    : [] (arrays are now inside the text)

    The formatter never mutates the original Sample; it always returns a new one.
    """

    _TOKEN_RE = _TS_TOKEN_RE

    @staticmethod
    def to_combined(
        sample: Sample,
        serializer: TSSerializer = DefaultTSSerializer(),
    ) -> Sample:
        """Replace every <TS_N> token with its serialized array value.

        Returns a new Sample with an empty input_ts list — the time series
        information is now embedded in input_text.

        Raises ValueError if a token references an index that does not exist
        in input_ts.
        """
        n_arrays = len(sample.input_ts)

        def _replace(match: re.Match) -> str:
            idx = int(match.group(1))
            if idx >= n_arrays:
                raise ValueError(
                    f"SampleFormatter.to_combined: token <TS_{idx}> found in "
                    f"input_text but input_ts has only {n_arrays} array(s)."
                )
            return serializer.serialize(sample.input_ts[idx])

        combined_text = SampleFormatter._TOKEN_RE.sub(_replace, sample.input_text)
        return Sample(
            input_text=combined_text,
            input_ts=[],
            output=sample.output,
            metadata=deepcopy(sample.metadata),
        )

    @staticmethod
    def to_separate(sample: Sample) -> Sample:
        """Identity — the canonical form is already the separate form."""
        return sample
