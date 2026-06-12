"""Unit tests for QwenVLModel. No GPU, matplotlib render, or heavy deps required.

The module-level import of QwenVLModel is intentional: it ensures the full
fmeval.core.datasets and HuggingFace 'datasets' import chains complete BEFORE
any sys.modules patches are applied. This prevents importlib.util.find_spec
from receiving a MagicMock for 'torch' and raising ValueError.

Fixtures then patch sys.modules["torch"] only around the constructor call (and
keep it active for the test body so that `import torch` inside predict() works).
_plot_ts is replaced with a lightweight mock on each fixture instance so that
matplotlib and PIL are never imported during test runs.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Module-level import — triggers dataset package imports before any mocking.
from fmeval.core.models.qwen_vl_model import QwenVLModel
from fmeval.core.sample import Sample

# ---------------------------------------------------------------------------
# Mock helpers (mirrors test_chatts_model.py conventions)
# ---------------------------------------------------------------------------


class _BatchEncoding(dict):
    """Minimal dict with .to() — mimics HuggingFace BatchEncoding."""

    def to(self, device):
        return self


def _make_batch_encoding(input_seq_len: int = 10) -> _BatchEncoding:
    ids = MagicMock()
    ids.shape = (1, input_seq_len)
    return _BatchEncoding({"input_ids": ids, "attention_mask": MagicMock()})


def _build_torch_mock() -> types.ModuleType:
    m = types.ModuleType("torch")
    m.__spec__ = None
    m.bfloat16 = "bfloat16"
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    m.inference_mode = MagicMock(return_value=ctx)
    return m


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def qwen():
    """QwenVLModel instance backed entirely by lightweight mocks."""
    param = MagicMock()
    param.device = "cpu"

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.parameters.side_effect = lambda: iter([param])

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "formatted prompt"
    mock_processor.decode.return_value = "C)"

    tf_mock = MagicMock()
    tf_mock.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
    tf_mock.AutoProcessor.from_pretrained.return_value = mock_processor

    torch_mock = _build_torch_mock()

    qwen_vl_utils_mock = MagicMock()
    qwen_vl_utils_mock.process_vision_info.return_value = ([], [])

    with patch.dict(
        "sys.modules",
        {
            "torch": torch_mock,
            "transformers": tf_mock,
            "qwen_vl_utils": qwen_vl_utils_mock,
        },
    ):
        instance = QwenVLModel(checkpoint_path="fake-model/qwen3-vl")
        instance._load_if_needed()

        # Replace _plot_ts so matplotlib / PIL are never imported.
        instance._plot_ts = MagicMock(return_value=MagicMock(name="PIL_Image"))

        # Wire processor callable for predict() tests.
        instance._processor.side_effect = lambda **kw: _make_batch_encoding()

        # Wire model.generate to return a list of one fake token sequence.
        fake_out = MagicMock(name="out_ids")
        instance._model.generate.return_value = [fake_out]

        yield instance


# ---------------------------------------------------------------------------
# Shared test samples
# ---------------------------------------------------------------------------


def _no_ts_sample() -> Sample:
    return Sample(
        input_text="What is 2+2? A) 3 B) 4 C) 5 D) 6",
        input_ts=[],
        output="B) 4",
        metadata={},
    )


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
            "First: <TS_0>\nSecond: <TS_1>\n\nWhich grows faster? A) First B) Second"
        ),
        input_ts=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 5.0], dtype=np.float32),
        ],
        output="B) Second",
        metadata={"id": 2},
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_model_name(qwen):
    assert qwen.model_name == "qwen3-vl-8b"


def test_model_name_is_lowercase(qwen):
    assert qwen.model_name == qwen.model_name.lower()


def test_input_mode_is_image(qwen):
    assert qwen.input_mode == "image"


def test_supported_modalities_includes_multimodal(qwen):
    assert "multimodal" in qwen.supported_modalities


# ---------------------------------------------------------------------------
# format_input — structure
# ---------------------------------------------------------------------------


def test_format_input_returns_dict(qwen):
    result = qwen.format_input(_single_ts_sample())
    assert isinstance(result, dict)


def test_format_input_has_messages_key(qwen):
    result = qwen.format_input(_single_ts_sample())
    assert "messages" in result


def test_format_input_messages_has_one_user_turn(qwen):
    result = qwen.format_input(_single_ts_sample())
    messages = result["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_format_input_content_is_list(qwen):
    result = qwen.format_input(_single_ts_sample())
    assert isinstance(result["messages"][0]["content"], list)


# ---------------------------------------------------------------------------
# format_input — image blocks
# ---------------------------------------------------------------------------


def test_format_input_single_ts_produces_one_image_block(qwen):
    content = qwen.format_input(_single_ts_sample())["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 1


def test_format_input_two_ts_produces_two_image_blocks(qwen):
    content = qwen.format_input(_two_ts_sample())["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 2


def test_format_input_no_ts_produces_no_image_blocks(qwen):
    content = qwen.format_input(_no_ts_sample())["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 0


def test_format_input_calls_plot_ts_once_per_ts(qwen):
    qwen.format_input(_two_ts_sample())
    assert qwen._plot_ts.call_count == 2


def test_format_input_image_block_has_image_key(qwen):
    content = qwen.format_input(_single_ts_sample())["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image")
    assert "image" in image_block


# ---------------------------------------------------------------------------
# format_input — text blocks
# ---------------------------------------------------------------------------


def test_format_input_preserves_text_outside_placeholders(qwen):
    content = qwen.format_input(_single_ts_sample())["messages"][0]["content"]
    all_text = "".join(b["text"] for b in content if b["type"] == "text")
    assert "How does it trend?" in all_text
    assert "A) Up" in all_text


def test_format_input_no_ts_tokens_in_content_text(qwen):
    content = qwen.format_input(_single_ts_sample())["messages"][0]["content"]
    all_text = "".join(b["text"] for b in content if b["type"] == "text")
    assert "<TS_0>" not in all_text


def test_format_input_no_ts_sample_has_single_text_block(qwen):
    content = qwen.format_input(_no_ts_sample())["messages"][0]["content"]
    text_blocks = [b for b in content if b["type"] == "text"]
    assert len(text_blocks) == 1
    assert "What is 2+2?" in text_blocks[0]["text"]


def test_format_input_interleaves_text_before_image(qwen):
    # For "First: <TS_0>\nSecond: <TS_1>...", the first content block is text
    content = qwen.format_input(_two_ts_sample())["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"


def test_format_input_does_not_mutate_sample(qwen):
    sample = _single_ts_sample()
    original_text = sample.input_text
    qwen.format_input(sample)
    assert sample.input_text == original_text


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_list(qwen):
    result = qwen.predict([qwen.format_input(_single_ts_sample())])
    assert isinstance(result, list)


def test_predict_single_input_returns_one_string(qwen):
    result = qwen.predict([qwen.format_input(_single_ts_sample())])
    assert len(result) == 1
    assert isinstance(result[0], str)


def test_predict_batch_length_matches_input(qwen):
    fmts = [qwen.format_input(_single_ts_sample()) for _ in range(4)]
    result = qwen.predict(fmts)
    assert len(result) == 4


def test_predict_returns_decoded_string(qwen):
    result = qwen.predict([qwen.format_input(_single_ts_sample())])
    # mock_processor.decode returns "C)"
    assert result[0] == "C)"
