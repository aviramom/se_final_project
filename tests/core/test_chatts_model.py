"""Unit tests for ChatTSModel. No GPU or heavy deps required.

The module-level import of ChatTSModel is intentional: it ensures the full
fmeval.core.datasets and HuggingFace 'datasets' import chains complete BEFORE
any sys.modules patches are applied. This prevents importlib.util.find_spec
from receiving a MagicMock for 'torch' and raising ValueError.

Fixtures then patch sys.modules["torch"] only around the constructor call (and
keep it active for the test body so that `import torch` inside predict() works).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Module-level import — triggers dataset package imports before any mocking.
from fmeval.core.models.chatts_model import ChatTSModel
from fmeval.core.sample import Sample

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _BatchEncoding(dict):
    """Minimal dict with .to() — mimics HuggingFace BatchEncoding."""

    def to(self, device):
        return self


def _make_batch_encoding(batch_size: int, input_seq_len: int = 10) -> _BatchEncoding:
    ids = MagicMock()
    ids.shape = (batch_size, input_seq_len)
    return _BatchEncoding({"input_ids": ids, "attention_mask": MagicMock()})


def _build_torch_mock() -> types.ModuleType:
    m = types.ModuleType("torch")
    m.__spec__ = None  # safe value if importlib.util.find_spec is called
    m.float16 = "float16"
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    m.inference_mode = MagicMock(return_value=ctx)
    return m


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def chatts():
    """ChatTSModel instance backed entirely by lightweight mocks."""
    param = MagicMock()
    param.device = "cpu"

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.parameters.side_effect = lambda: iter([param])

    mock_tokenizer = MagicMock()
    # Return the user message content so format_input tests can inspect actual text.
    mock_tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: msgs[0]["content"]
    mock_tokenizer.decode.return_value = "The answer is B)"

    mock_processor = MagicMock()

    tf_mock = MagicMock()
    tf_mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    tf_mock.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    tf_mock.AutoProcessor.from_pretrained.return_value = mock_processor

    torch_mock = _build_torch_mock()

    with patch.dict("sys.modules", {"torch": torch_mock, "transformers": tf_mock}):
        instance = ChatTSModel(checkpoint_path="/fake/path")
        instance._load_if_needed()  # trigger lazy load while mocks are active

        # Wire processor and model generate for predict() tests.
        def proc_side_effect(text=None, timeseries=None, **kw):
            return _make_batch_encoding(batch_size=len(text) if text else 1)

        instance._processor.side_effect = proc_side_effect

        def generate_side_effect(input_ids=None, **kw):
            n = input_ids.shape[0] if input_ids is not None else 1
            return [MagicMock() for _ in range(n)]

        instance._model.generate.side_effect = generate_side_effect

        yield instance


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_model_name_is_chatts_8b(chatts):
    assert chatts.model_name == "chatts-8b"


def test_model_name_is_lowercase(chatts):
    assert chatts.model_name == chatts.model_name.lower()


def test_input_mode_is_separate(chatts):
    assert chatts.input_mode == "separate"


def test_supported_modalities_includes_multimodal(chatts):
    assert "multimodal" in chatts.supported_modalities


# ---------------------------------------------------------------------------
# format_input — placeholder stripping and text handling
# ---------------------------------------------------------------------------


def _single_ts_sample() -> Sample:
    return Sample(
        input_text=(
            "Time series: <TS_0>\n\n"
            "Question: How does it trend?\n\n"
            "Options:\nA) Up\nB) Down"
        ),
        input_ts=[np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        output="A) Up",
        metadata={"id": 1},
    )


def _two_ts_sample() -> Sample:
    return Sample(
        input_text=(
            "Time series 1: <TS_0>\nTime series 2: <TS_1>\n\n"
            "Question: Compare them."
        ),
        input_ts=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        output="B) Increase",
        metadata={"id": 2},
    )


def test_format_input_returns_dict(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert isinstance(result, dict)


def test_format_input_has_text_and_timeseries_keys(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert "text" in result
    assert "timeseries" in result


def test_format_input_text_is_string(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert isinstance(result["text"], str)


def test_format_input_strips_single_ts_placeholder(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert "<TS_0>" not in result["text"]


def test_format_input_strips_both_ts_placeholders(chatts):
    result = chatts.format_input(_two_ts_sample())
    assert "<TS_0>" not in result["text"]
    assert "<TS_1>" not in result["text"]


def test_format_input_replaces_placeholder_with_marker(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert "<ts><ts/>" in result["text"]


def test_format_input_replaces_both_placeholders(chatts):
    result = chatts.format_input(_two_ts_sample())
    assert result["text"].count("<ts><ts/>") == 2


def test_format_input_preserves_question_text(chatts):
    result = chatts.format_input(_single_ts_sample())
    assert "How does it trend?" in result["text"]
    assert "A) Up" in result["text"]


def test_format_input_timeseries_single(chatts):
    sample = _single_ts_sample()
    result = chatts.format_input(sample)
    assert len(result["timeseries"]) == 1
    np.testing.assert_array_equal(result["timeseries"][0], sample.input_ts[0])


def test_format_input_timeseries_two(chatts):
    sample = _two_ts_sample()
    result = chatts.format_input(sample)
    assert len(result["timeseries"]) == 2
    np.testing.assert_array_equal(result["timeseries"][0], sample.input_ts[0])
    np.testing.assert_array_equal(result["timeseries"][1], sample.input_ts[1])


def test_format_input_does_not_mutate_sample(chatts):
    sample = _single_ts_sample()
    original_text = sample.input_text
    chatts.format_input(sample)
    assert sample.input_text == original_text


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_list(chatts):
    fmt = chatts.format_input(_single_ts_sample())
    result = chatts.predict([fmt])
    assert isinstance(result, list)


def test_predict_single_input_returns_one_string(chatts):
    fmt = chatts.format_input(_single_ts_sample())
    result = chatts.predict([fmt])
    assert len(result) == 1
    assert isinstance(result[0], str)


def test_predict_batch_length_matches_input(chatts):
    fmts = [chatts.format_input(_single_ts_sample()) for _ in range(4)]
    result = chatts.predict(fmts)
    assert len(result) == 4
