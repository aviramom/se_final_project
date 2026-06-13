"""Tests for LocalEvaluationPipeline + RunResult."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

# sklearn warns when predictions contain classes absent from targets (e.g.
# mock model predicts A on a B-only subset). Expected in tests; suppress.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

from fmeval.core.datasets.base import Dataset
from fmeval.core.metrics.mcq_metrics import MCQMetrics
from fmeval.core.models.mock_model import MockModel
from fmeval.core.sample import Sample
from fmeval.evaluation.pipeline import LocalEvaluationPipeline
from fmeval.evaluation.result import RunResult, SamplePrediction


# ── Minimal in-memory dataset for testing ─────────────────────────────────────

def _make_samples(answers: list[str], difficulty="medium", category="Test") -> list[Sample]:
    """Create samples where each answer is a letter like 'A) opt'."""
    return [
        Sample(
            input_text=f"Q{i}: <TS_0> What?",
            input_ts=[np.array([float(i), float(i + 1)])],
            output=ans,
            metadata={
                "id": i,
                "difficulty": difficulty,
                "category": category,
                "subcategory": "sub",
                "question_type": "binary",
                "num_options": 2,
                "answer_letter": ans[0],
                "answer_text": ans[3:],
                "tid": i,
                "relevant_concepts": [],
            },
        )
        for i, ans in enumerate(answers)
    ]


class InMemoryDataset(Dataset):
    def __init__(self, samples: list[Sample], name: str = "test_ds"):
        self._samples = samples
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def modality(self):
        return "multimodal"

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        yield from self._samples


# ── Pipeline basic ─────────────────────────────────────────────────────────────

def _make_pipeline(**kwargs) -> LocalEvaluationPipeline:
    return LocalEvaluationPipeline(
        model=MockModel("A"),
        metric=MCQMetrics(),
        verbose=False,
        **kwargs,
    )


def _run(answers: list[str], **kwargs) -> RunResult:
    ds = InMemoryDataset(_make_samples(answers))
    return _make_pipeline(**kwargs).run(ds)


def test_run_returns_run_result():
    result = _run(["A) yes", "B) no"])
    assert isinstance(result, RunResult)


def test_num_samples_correct():
    result = _run(["A) yes", "B) no", "C) maybe"])
    assert result.num_samples == 3


def test_model_name_in_result():
    result = _run(["A) yes"])
    assert result.model_name == "mock_always_a"


def test_dataset_name_in_result():
    ds = InMemoryDataset(_make_samples(["A) yes"]), name="my_ds")
    result = _make_pipeline().run(ds)
    assert result.dataset_name == "my_ds"


def test_sample_predictions_count():
    result = _run(["A) x", "B) y", "C) z"])
    assert len(result.sample_predictions) == 3


def test_sample_predictions_are_correct_type():
    result = _run(["A) x"])
    assert isinstance(result.sample_predictions[0], SamplePrediction)


def test_sample_prediction_input_text_populated():
    samples = _make_samples(["A) yes", "B) no"])
    ds = InMemoryDataset(samples)
    result = _make_pipeline().run(ds)
    for i, sp in enumerate(result.sample_predictions):
        assert sp.input_text == samples[i].input_text
        assert sp.input_text != ""


def test_empty_dataset_raises():
    ds = InMemoryDataset([])
    with pytest.raises(ValueError, match="empty"):
        _make_pipeline().run(ds)


# ── Correctness checks ─────────────────────────────────────────────────────────

def test_all_correct_when_answers_are_A():
    # Mock always returns A); all targets start with A)
    result = _run(["A) yes", "A) no", "A) maybe"])
    assert result.metrics["accuracy"] == pytest.approx(1.0)


def test_partial_correct():
    # 2 A targets (correct) + 1 B target (wrong) → 2/3 accuracy
    result = _run(["A) yes", "A) no", "B) other"])
    assert result.metrics["accuracy"] == pytest.approx(2 / 3)


def test_is_correct_set_per_sample():
    result = _run(["A) yes", "B) no"])   # mock answers A) for both
    assert result.sample_predictions[0].is_correct is True
    assert result.sample_predictions[1].is_correct is False


def test_predicted_letter_is_a_for_mock():
    result = _run(["A) yes", "B) no"])
    for sp in result.sample_predictions:
        assert sp.predicted_letter == "A"


def test_correct_letter_extracted_from_target():
    result = _run(["C) text"])
    assert result.sample_predictions[0].correct_letter == "C"


# ── Aggregate metrics keys ─────────────────────────────────────────────────────

def test_all_metric_keys_present():
    result = _run(["A) x"] * 5 + ["B) y"] * 5)
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                "precision_macro", "recall_macro", "n_samples", "n_unparseable"):
        assert key in result.metrics, f"Missing: {key}"


# ── RunResult.to_dataframe ────────────────────────────────────────────────────

def test_to_dataframe_shape():
    result = _run(["A) x", "B) y", "C) z"])
    df = result.to_dataframe()
    assert len(df) == 3
    assert "is_correct" in df.columns
    assert "difficulty" in df.columns   # promoted from metadata


def test_to_dataframe_metadata_promotion():
    result = _run(["A) x"])
    df = result.to_dataframe()
    assert "category" in df.columns
    assert df.iloc[0]["category"] == "Test"


# ── RunResult.breakdown_by ────────────────────────────────────────────────────

def test_breakdown_by_difficulty():
    easy = _make_samples(["A) x", "A) y"], difficulty="easy")
    hard = _make_samples(["B) x", "B) y"], difficulty="hard")
    ds = InMemoryDataset(easy + hard)
    result = _make_pipeline().run(ds)
    bd = result.breakdown_by("difficulty")
    assert set(bd["difficulty"]) == {"easy", "hard"}


def test_breakdown_accuracy_easy_vs_hard():
    easy = _make_samples(["A) x", "A) y"], difficulty="easy")   # A correct → 100%
    hard = _make_samples(["B) x", "B) y"], difficulty="hard")   # B correct → 0%
    ds = InMemoryDataset(easy + hard)
    result = _make_pipeline().run(ds)
    bd = result.breakdown_by("difficulty").set_index("difficulty")
    assert bd.loc["easy", "accuracy"] == pytest.approx(1.0)
    assert bd.loc["hard", "accuracy"] == pytest.approx(0.0)


def test_breakdown_by_unknown_key_raises():
    result = _run(["A) x"])
    with pytest.raises(KeyError, match="nonexistent"):
        result.breakdown_by("nonexistent")


# ── RunResult.to_json ─────────────────────────────────────────────────────────

def test_to_json_is_valid_json():
    import json
    result = _run(["A) x", "B) y"])
    parsed = json.loads(result.to_json())
    assert parsed["model_name"] == "mock_always_a"
    assert parsed["num_samples"] == 2
    assert len(parsed["sample_predictions"]) == 2


# ── RunResult.summary ─────────────────────────────────────────────────────────

def test_summary_contains_model_name():
    result = _run(["A) x"])
    s = result.summary()
    assert "mock_always_a" in s


def test_summary_contains_accuracy():
    result = _run(["A) x"])
    s = result.summary()
    assert "accuracy" in s


# ── Batching ──────────────────────────────────────────────────────────────────

def test_small_batch_size_gives_same_results():
    answers = ["A) x"] * 5 + ["B) y"] * 5
    r1 = _run(answers, batch_size=2)
    r2 = _run(answers, batch_size=100)
    assert r1.metrics["accuracy"] == pytest.approx(r2.metrics["accuracy"])


# ── run_config ────────────────────────────────────────────────────────────────

def test_run_config_contains_expected_keys():
    result = _run(["A) x"])
    for key in ("model_name", "dataset_name", "input_mode", "batch_size", "metric"):
        assert key in result.run_config
