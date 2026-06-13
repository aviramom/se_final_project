"""Tests for fmeval.core.models.mock_model.MockModel."""

import numpy as np
import pytest

from fmeval.core.models.mock_model import MockModel
from fmeval.core.sample import Sample


def _sample(text="<TS_0> question?", arrays=None):
    if arrays is None:
        arrays = [np.array([1.0, 2.0, 3.0])]
    return Sample(input_text=text, input_ts=arrays, output="A) correct")


class TestMockModel:
    def test_default_answer_is_a(self):
        m = MockModel()
        assert m.predict(["any input"]) == ["A)"]

    def test_custom_answer_b(self):
        m = MockModel("B")
        assert m.predict(["x", "y"]) == ["B)", "B)"]

    def test_custom_answer_with_paren(self):
        m = MockModel("C)")
        assert m.predict(["x"]) == ["C)"]

    def test_invalid_answer_raises(self):
        with pytest.raises(ValueError, match="A/B/C/D"):
            MockModel("E")

    def test_model_name_reflects_answer(self):
        assert MockModel("A").model_name == "mock_always_a"
        assert MockModel("B").model_name == "mock_always_b"

    def test_supported_modalities_includes_multimodal(self):
        assert "multimodal" in MockModel().supported_modalities

    def test_input_mode_is_combined(self):
        assert MockModel().input_mode == "combined"

    def test_format_input_returns_string(self):
        m = MockModel()
        result = m.format_input(_sample())
        assert isinstance(result, str)

    def test_format_input_inlines_ts(self):
        m = MockModel()
        result = m.format_input(_sample())
        assert "<TS_0>" not in result    # placeholder must be replaced
        assert "[" in result            # serialised array present

    def test_predict_batch_length_matches(self):
        m = MockModel()
        inputs = ["a", "b", "c", "d", "e"]
        assert len(m.predict(inputs)) == 5

    def test_predict_empty_batch(self):
        assert MockModel().predict([]) == []
