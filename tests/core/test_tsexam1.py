"""Tests for fmeval.core.datasets.tsexam1.TimeSeriesExam1Dataset.

All tests use a mocked HuggingFace dataset so no network access is required.
To run a live integration test against the real HF dataset, call
    TimeSeriesExam1Dataset(max_samples=5)
manually and inspect the output.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fmeval.core.datasets.tsexam1 import TimeSeriesExam1Dataset
from fmeval.core.sample import Sample

# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_TS_SINGLE = list(range(1024))           # single float list
_TS_A = [float(i) for i in range(1024)]
_TS_B = [float(i * 2) for i in range(1024)]

_FORMAT_HINT = (
    "Please answer the question and provide the correct option letter, "
    "e.g., A), B), C), D), and option content at the end of your answer."
)

_FAKE_ROWS = [
    # Row 0: single TS, 3 options, answer is index 2
    {
        "id": 1, "tid": 10,
        "question": "How does the amplitude change?",
        "options": ["Increase", "Remain the same", "Decrease"],
        "answer": "Decrease",
        "question_type": "multiple-choice",
        "ts": _TS_SINGLE, "ts1": None, "ts2": None,
        "difficulty": "hard",
        "format_hint": _FORMAT_HINT,
        "relevant_concepts": ["Sine Wave"],
        "question_hint": "",
        "category": "Pattern Recognition",
        "subcategory": "Cycle Recognition",
    },
    # Row 1: two TS, 2 options, answer is index 0
    {
        "id": 2, "tid": 91,
        "question": "Do these two series share the same distribution?",
        "options": ["No, they differ", "Yes, they are the same"],
        "answer": "No, they differ",
        "question_type": "binary",
        "ts": None, "ts1": _TS_A, "ts2": _TS_B,
        "difficulty": "hard",
        "format_hint": _FORMAT_HINT,
        "relevant_concepts": ["Gaussian White Noise"],
        "question_hint": "",
        "category": "Similarity Analysis",
        "subcategory": "Distributional",
    },
    # Row 2: single TS, 4 options, answer is index 1
    {
        "id": 3, "tid": 5,
        "question": "What pattern is observed?",
        "options": ["SquareWave", "SineWave", "SawtoothWave", "No Pattern"],
        "answer": "SineWave",
        "question_type": "multiple_choice",
        "ts": _TS_SINGLE, "ts1": None, "ts2": None,
        "difficulty": "easy",
        "format_hint": _FORMAT_HINT,
        "relevant_concepts": ["Sine Wave"],
        "question_hint": "",
        "category": "Pattern Recognition",
        "subcategory": "Wave Recognition",
    },
]


def _make_dataset(rows=None, max_samples=None):
    if rows is None:
        rows = _FAKE_ROWS
    with patch("fmeval.core.datasets.tsexam1.load_dataset", return_value=rows):
        return TimeSeriesExam1Dataset(max_samples=max_samples)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_name():
    ds = _make_dataset()
    assert ds.name == "tsexam1"


def test_modality_is_multimodal():
    ds = _make_dataset()
    assert ds.modality == "multimodal"


def test_len():
    ds = _make_dataset()
    assert len(ds) == len(_FAKE_ROWS)


def test_len_with_max_samples():
    ds = _make_dataset(max_samples=2)
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# Single-TS rows
# ---------------------------------------------------------------------------

def test_single_ts_row_has_one_array():
    ds = _make_dataset()
    samples = list(ds)
    s0 = samples[0]
    assert len(s0.input_ts) == 1


def test_single_ts_row_placeholder_in_text():
    ds = _make_dataset()
    s = list(ds)[0]
    assert "<TS_0>" in s.input_text
    assert "<TS_1>" not in s.input_text


def test_single_ts_header_label():
    ds = _make_dataset()
    s = list(ds)[0]
    assert s.input_text.startswith("Time series: <TS_0>")


def test_single_ts_array_values():
    ds = _make_dataset()
    s = list(ds)[0]
    np.testing.assert_array_equal(s.input_ts[0], np.array(_TS_SINGLE, dtype=np.float32))


# ---------------------------------------------------------------------------
# Two-TS rows
# ---------------------------------------------------------------------------

def test_two_ts_row_has_two_arrays():
    ds = _make_dataset()
    s1 = list(ds)[1]
    assert len(s1.input_ts) == 2


def test_two_ts_both_placeholders_in_text():
    ds = _make_dataset()
    s = list(ds)[1]
    assert "<TS_0>" in s.input_text
    assert "<TS_1>" in s.input_text


def test_two_ts_header_labels():
    ds = _make_dataset()
    s = list(ds)[1]
    assert "Time series 1: <TS_0>" in s.input_text
    assert "Time series 2: <TS_1>" in s.input_text


def test_two_ts_arrays_are_correct():
    ds = _make_dataset()
    s = list(ds)[1]
    np.testing.assert_array_equal(s.input_ts[0], np.array(_TS_A, dtype=np.float32))
    np.testing.assert_array_equal(s.input_ts[1], np.array(_TS_B, dtype=np.float32))


# ---------------------------------------------------------------------------
# Options and answer formatting
# ---------------------------------------------------------------------------

def test_output_contains_correct_letter_and_text():
    ds = _make_dataset()
    samples = list(ds)
    assert samples[0].output == "C) Decrease"     # index 2 → C
    assert samples[1].output == "A) No, they differ"  # index 0 → A
    assert samples[2].output == "B) SineWave"     # index 1 → B


def test_options_appear_in_input_text():
    ds = _make_dataset()
    s = list(ds)[0]
    assert "A) Increase" in s.input_text
    assert "B) Remain the same" in s.input_text
    assert "C) Decrease" in s.input_text


def test_question_appears_in_input_text():
    ds = _make_dataset()
    s = list(ds)[0]
    assert "How does the amplitude change?" in s.input_text


def test_format_hint_appears_in_input_text():
    ds = _make_dataset()
    s = list(ds)[0]
    assert _FORMAT_HINT in s.input_text


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_fields_present():
    ds = _make_dataset()
    s = list(ds)[0]
    for key in ("id", "tid", "question_type", "difficulty", "category",
                "subcategory", "relevant_concepts", "answer_letter", "answer_text",
                "num_options"):
        assert key in s.metadata, f"Missing metadata key: {key}"


def test_metadata_answer_letter_matches_output():
    ds = _make_dataset()
    samples = list(ds)
    assert samples[0].metadata["answer_letter"] == "C"
    assert samples[1].metadata["answer_letter"] == "A"
    assert samples[2].metadata["answer_letter"] == "B"


def test_metadata_num_options():
    ds = _make_dataset()
    samples = list(ds)
    assert samples[0].metadata["num_options"] == 3
    assert samples[1].metadata["num_options"] == 2
    assert samples[2].metadata["num_options"] == 4


# ---------------------------------------------------------------------------
# max_samples
# ---------------------------------------------------------------------------

def test_max_samples_limits_iteration():
    ds = _make_dataset(max_samples=1)
    samples = list(ds)
    assert len(samples) == 1


def test_max_samples_larger_than_dataset():
    ds = _make_dataset(max_samples=100)
    assert len(list(ds)) == len(_FAKE_ROWS)


# ---------------------------------------------------------------------------
# Sample type
# ---------------------------------------------------------------------------

def test_yields_sample_instances():
    ds = _make_dataset()
    for s in ds:
        assert isinstance(s, Sample)
