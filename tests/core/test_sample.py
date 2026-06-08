"""Tests for fmeval.core.sample.Sample."""

import numpy as np
import pytest

from fmeval.core.sample import Sample


def test_sample_construction_basic():
    ts = [np.array([1.0, 2.0, 3.0])]
    s = Sample(input_text="Describe <TS_0>.", input_ts=ts, output="upward trend")
    assert s.input_text == "Describe <TS_0>."
    assert len(s.input_ts) == 1
    np.testing.assert_array_equal(s.input_ts[0], [1.0, 2.0, 3.0])
    assert s.output == "upward trend"


def test_sample_metadata_defaults_to_empty_dict():
    s = Sample(input_text="<TS_0>", input_ts=[np.zeros(5)], output="ok")
    assert s.metadata == {}


def test_sample_metadata_is_stored():
    meta = {"source": "etth1", "split": "test"}
    s = Sample(input_text="<TS_0>", input_ts=[np.zeros(5)], output="ok", metadata=meta)
    assert s.metadata["source"] == "etth1"


def test_sample_multiple_ts_arrays():
    ts = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    s = Sample(input_text="<TS_0> and <TS_1>", input_ts=ts, output="both rise")
    assert len(s.input_ts) == 2


def test_sample_metadata_instances_are_independent():
    s1 = Sample(input_text="<TS_0>", input_ts=[np.zeros(3)], output="a")
    s2 = Sample(input_text="<TS_0>", input_ts=[np.zeros(3)], output="b")
    s1.metadata["key"] = "value"
    assert "key" not in s2.metadata
