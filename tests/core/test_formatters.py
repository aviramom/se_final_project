"""Tests for fmeval.core.datasets.formatters."""

import numpy as np
import pytest

from fmeval.core.datasets.formatters import DefaultTSSerializer, SampleFormatter
from fmeval.core.sample import Sample


def _make_sample(text: str, arrays: list) -> Sample:
    return Sample(
        input_text=text,
        input_ts=[np.array(a, dtype=np.float32) for a in arrays],
        output="answer",
    )


# ── DefaultTSSerializer ────────────────────────────────────────────────────────

def test_default_serializer_basic():
    s = DefaultTSSerializer()
    result = s.serialize(np.array([1.0, 2.5, -3.0]))
    assert result == "[1, 2.5, -3]"


def test_default_serializer_single_value():
    s = DefaultTSSerializer()
    assert s.serialize(np.array([42.0])) == "[42]"


# ── SampleFormatter.to_combined ───────────────────────────────────────────────

def test_to_combined_replaces_single_token():
    sample = _make_sample("Values: <TS_0>. What trend?", [[1.0, 2.0, 3.0]])
    combined = SampleFormatter.to_combined(sample)
    assert "<TS_0>" not in combined.input_text
    assert "[1, 2, 3]" in combined.input_text
    assert combined.input_ts == []


def test_to_combined_replaces_multiple_tokens():
    sample = _make_sample(
        "Series A: <TS_0>. Series B: <TS_1>. Compare.",
        [[1.0, 2.0], [10.0, 20.0]],
    )
    combined = SampleFormatter.to_combined(sample)
    assert "<TS_0>" not in combined.input_text
    assert "<TS_1>" not in combined.input_text
    assert "[1, 2]" in combined.input_text
    assert "[10, 20]" in combined.input_text


def test_to_combined_preserves_output_and_metadata():
    sample = _make_sample("<TS_0>", [[1.0]])
    sample.metadata["id"] = "test-1"
    combined = SampleFormatter.to_combined(sample)
    assert combined.output == "answer"
    assert combined.metadata["id"] == "test-1"


def test_to_combined_does_not_mutate_original():
    sample = _make_sample("prompt <TS_0> end", [[1.0, 2.0]])
    original_text = sample.input_text
    SampleFormatter.to_combined(sample)
    assert sample.input_text == original_text
    assert len(sample.input_ts) == 1


def test_to_combined_raises_on_out_of_range_token():
    sample = _make_sample("refer to <TS_2>", [[1.0], [2.0]])  # only TS_0 and TS_1
    with pytest.raises(ValueError, match="TS_2"):
        SampleFormatter.to_combined(sample)


def test_to_combined_repeated_token_uses_same_array():
    sample = _make_sample("<TS_0> and again <TS_0>", [[5.0, 6.0]])
    combined = SampleFormatter.to_combined(sample)
    assert combined.input_text.count("[5, 6]") == 2


# ── SampleFormatter.to_separate ───────────────────────────────────────────────

def test_to_separate_is_identity():
    sample = _make_sample("<TS_0>", [[1.0, 2.0]])
    result = SampleFormatter.to_separate(sample)
    assert result is sample  # exact same object, not a copy


# ── Custom serializer ─────────────────────────────────────────────────────────

def test_custom_serializer_is_used():
    class SpaceSerializer:
        def serialize(self, ts: np.ndarray) -> str:
            return " ".join(str(v) for v in ts.flatten())

    sample = _make_sample("<TS_0>", [[1.0, 2.0, 3.0]])
    combined = SampleFormatter.to_combined(sample, serializer=SpaceSerializer())
    assert "1.0 2.0 3.0" in combined.input_text
