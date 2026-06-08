"""Tests for fmeval.core.datasets.template.JSONLMultimodalDataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fmeval.core.datasets.template import JSONLMultimodalDataset
from fmeval.core.sample import Sample


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _write_jsonl(records: list[dict]) -> Path:
    """Write records to a temp JSONL file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        tmp.write(json.dumps(rec) + "\n")
    tmp.close()
    return Path(tmp.name)


VALID_RECORDS = [
    {
        "context": "The weekly readings are <TS_0>. Is there an anomaly?",
        "ts": [[12.3, 14.1, 9.8, 25.4, 13.0, 11.7, 10.2]],
        "answer": "Day 4 shows an anomalous spike.",
    },
    {
        "context": "Compare series A <TS_0> and B <TS_1>.",
        "ts": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "answer": "Both series increase monotonically.",
    },
]


# ── Basic loading ──────────────────────────────────────────────────────────────

def test_len_matches_record_count():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    assert len(ds) == len(VALID_RECORDS)


def test_iter_yields_correct_sample_count():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    samples = list(ds)
    assert len(samples) == len(VALID_RECORDS)


def test_sample_fields_are_correct():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    first = next(iter(ds))

    assert isinstance(first, Sample)
    assert first.input_text == VALID_RECORDS[0]["context"]
    assert first.output == VALID_RECORDS[0]["answer"]
    assert len(first.input_ts) == 1
    np.testing.assert_array_almost_equal(
        first.input_ts[0],
        np.array(VALID_RECORDS[0]["ts"][0], dtype=np.float32),
    )


def test_second_sample_has_two_ts_arrays():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    samples = list(ds)
    assert len(samples[1].input_ts) == 2


# ── max_samples ────────────────────────────────────────────────────────────────

def test_max_samples_limits_iteration():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path, max_samples=1)
    assert len(list(ds)) == 1


def test_max_samples_limits_len():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path, max_samples=1)
    assert len(ds) == 1


# ── Modality ───────────────────────────────────────────────────────────────────

def test_modality_is_multimodal():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    assert ds.modality == "multimodal"


# ── name derives from filename ─────────────────────────────────────────────────

def test_name_is_stem_of_file():
    path = _write_jsonl(VALID_RECORDS)
    ds = JSONLMultimodalDataset(path)
    assert ds.name == path.stem


# ── Validation: mismatched placeholder ────────────────────────────────────────

def test_validation_raises_on_out_of_range_token():
    bad_records = [
        {
            "context": "Data: <TS_0> and <TS_2>",  # TS_2 but only one array
            "ts": [[1.0, 2.0]],
            "answer": "n/a",
        }
    ]
    path = _write_jsonl(bad_records)
    ds = JSONLMultimodalDataset(path)
    with pytest.raises(ValueError, match="TS_2"):
        list(ds)


def test_validation_raises_on_empty_ts():
    bad_records = [
        {
            "context": "No time series here.",
            "ts": [],  # empty — not allowed for multimodal
            "answer": "n/a",
        }
    ]
    path = _write_jsonl(bad_records)
    ds = JSONLMultimodalDataset(path)
    with pytest.raises(ValueError, match="empty input_ts"):
        list(ds)


# ── Malformed JSONL ────────────────────────────────────────────────────────────

def test_missing_key_raises_value_error():
    bad = [{"context": "hello", "ts": [[1.0]]}]  # no "answer"
    path = _write_jsonl(bad)
    ds = JSONLMultimodalDataset(path)
    with pytest.raises(ValueError, match="missing required key"):
        list(ds)


def test_blank_lines_are_skipped():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    tmp.write("\n")
    tmp.write(json.dumps(VALID_RECORDS[0]) + "\n")
    tmp.write("\n")
    tmp.close()
    ds = JSONLMultimodalDataset(Path(tmp.name))
    assert len(list(ds)) == 1
