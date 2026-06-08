"""Standardized data unit flowing between Dataset and ModelWrapper."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    """One benchmark record in canonical (separate) form.

    input_text contains the natural-language prompt.  Wherever a time series
    belongs inline, the text carries a placeholder token <TS_0>, <TS_1>, etc.
    The matching arrays live in input_ts at the same indices.

    SampleFormatter.to_combined() replaces the tokens with serialized values
    for models that need a single text string.  Models with a dedicated TS
    encoder receive the sample as-is.
    """

    input_text: str
    input_ts: list[np.ndarray]
    output: str
    metadata: dict = field(default_factory=dict)
