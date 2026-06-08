"""Intermediate base class for all multimodal (text + time-series) datasets."""

from __future__ import annotations

import re
from typing import Literal

from fmeval.core.datasets.base import Dataset
from fmeval.core.sample import Sample

_TS_TOKEN_RE = re.compile(r"<TS_(\d+)>")


class MultimodalDataset(Dataset):
    """Base class for benchmarks that combine natural language and time series.

    Fixes modality to 'multimodal' and provides _validate_sample(), which
    subclasses should call before yielding each Sample to catch mismatches
    between <TS_N> tokens in input_text and entries in input_ts early.
    """

    @property
    def modality(self) -> Literal["text", "time_series", "multimodal"]:
        return "multimodal"

    def _validate_sample(self, sample: Sample) -> None:
        """Raise ValueError if placeholder indices don't align with input_ts.

        Specifically checks:
        - Every <TS_N> token in input_text has a corresponding array at index N
          in input_ts.
        - input_ts is non-empty (all multimodal samples must carry at least one
          time series).
        """
        if not sample.input_ts:
            raise ValueError(
                f"[{self.name}] Sample has empty input_ts; multimodal datasets "
                "must include at least one time series array."
            )

        indices_in_text = {
            int(m.group(1)) for m in _TS_TOKEN_RE.finditer(sample.input_text)
        }
        max_ts_index = len(sample.input_ts) - 1

        out_of_range = {i for i in indices_in_text if i > max_ts_index}
        if out_of_range:
            raise ValueError(
                f"[{self.name}] input_text references <TS_{max(out_of_range)}> "
                f"but input_ts has only {len(sample.input_ts)} array(s) "
                f"(valid indices 0–{max_ts_index})."
            )
