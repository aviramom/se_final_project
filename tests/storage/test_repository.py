from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from fmeval.evaluation.result import SamplePrediction
from fmeval.storage.models import EvaluationResult
from fmeval.storage.repository import ResultsRepository


def _make_prediction(idx: int = 0, **kwargs: object) -> SamplePrediction:
    defaults: dict[str, object] = dict(
        sample_idx=idx,
        raw_prediction="A) yes",
        raw_target="A) yes",
        predicted_letter="A",
        correct_letter="A",
        is_correct=True,
        input_text=f"Question {idx}: <TS_0> What?",
        metadata={"difficulty": "easy", "category": "trend"},
    )
    defaults.update(kwargs)
    return SamplePrediction(**defaults)  # type: ignore[arg-type]


def _make_result(**kwargs: object) -> EvaluationResult:
    defaults: dict[str, object] = dict(
        job_id="job-001",
        model_name="mock-model",
        benchmark_name="ts-exam-1",
        modality="multimodal",
        metrics={"accuracy": 0.8, "f1_macro": 0.75},
        timestamp=datetime(2026, 6, 9, 12, 0, 0),
        execution_time_seconds=5.0,
        exp_id="seed_42",
        max_samples=50,
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path) -> ResultsRepository:
    return ResultsRepository(tmp_path / "test.db")


def test_save_and_get_by_job_id(repo: ResultsRepository) -> None:
    result = _make_result()
    repo.save(result)
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert fetched.job_id == "job-001"
    assert fetched.exp_id == "seed_42"
    assert fetched.max_samples == 50
    assert fetched.metrics["accuracy"] == pytest.approx(0.8)


def test_save_upsert(repo: ResultsRepository) -> None:
    repo.save(_make_result(metrics={"accuracy": 0.5}))
    repo.save(_make_result(metrics={"accuracy": 0.9}))  # same job_id
    results = repo.query()
    assert len(results) == 1
    assert results[0].metrics["accuracy"] == pytest.approx(0.9)


def test_query_no_filter(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", exp_id="exp-a"))
    repo.save(_make_result(job_id="j2", exp_id="exp-b"))
    assert len(repo.query()) == 2


def test_query_filter_by_exp_id(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", exp_id="exp-a"))
    repo.save(_make_result(job_id="j2", exp_id="exp-b"))
    results = repo.query(exp_id="exp-a")
    assert len(results) == 1
    assert results[0].exp_id == "exp-a"


def test_query_filter_by_model(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", model_name="model-x"))
    repo.save(_make_result(job_id="j2", model_name="model-y"))
    results = repo.query(model_name="model-x")
    assert len(results) == 1
    assert results[0].model_name == "model-x"


def test_query_filter_by_date_range(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", timestamp=datetime(2026, 1, 1)))
    repo.save(_make_result(job_id="j2", timestamp=datetime(2026, 6, 1)))
    results = repo.query(date_from=datetime(2026, 3, 1))
    assert len(results) == 1
    assert results[0].job_id == "j2"


def test_list_exp_ids(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", exp_id="beta"))
    repo.save(_make_result(job_id="j2", exp_id="alpha"))
    repo.save(_make_result(job_id="j3", exp_id=""))  # blank should be excluded
    exp_ids = repo.list_exp_ids()
    assert exp_ids == ["alpha", "beta"]  # sorted, blank excluded


def test_list_exp_ids_empty(repo: ResultsRepository) -> None:
    assert repo.list_exp_ids() == []


def test_migration_idempotent(tmp_path: Path) -> None:
    """Opening the same DB twice must not raise (migration runs twice)."""
    db = tmp_path / "test.db"
    ResultsRepository(db)
    ResultsRepository(db)  # second open triggers ALTER TABLE which should be no-op


def test_export_csv_includes_exp_id(repo: ResultsRepository) -> None:
    repo.save(_make_result(exp_id="my-exp"))
    csv = repo.export_csv(repo.query())
    assert "exp_id" in csv
    assert "my-exp" in csv


def test_get_by_job_id_missing(repo: ResultsRepository) -> None:
    assert repo.get_by_job_id("nonexistent") is None


# ---------------------------------------------------------------------------
# sample_predictions — save / retrieve
# ---------------------------------------------------------------------------

def test_save_and_get_sample_predictions(repo: ResultsRepository) -> None:
    predictions = [_make_prediction(0), _make_prediction(1, is_correct=False)]
    repo.save_sample_predictions("job-001", predictions)
    fetched = repo.get_sample_predictions("job-001")
    assert len(fetched) == 2
    assert fetched[0].sample_idx == 0
    assert fetched[0].input_text == "Question 0: <TS_0> What?"
    assert fetched[1].is_correct is False
    assert fetched[1].predicted_letter == "A"


def test_get_sample_predictions_empty_for_unknown_job(repo: ResultsRepository) -> None:
    assert repo.get_sample_predictions("nonexistent") == []


def test_sample_prediction_metadata_survives_serialization(repo: ResultsRepository) -> None:
    pred = _make_prediction(0, metadata={"difficulty": "hard", "category": "periodicity", "num_options": 4})
    repo.save_sample_predictions("job-001", [pred])
    fetched = repo.get_sample_predictions("job-001")
    assert fetched[0].metadata["difficulty"] == "hard"
    assert fetched[0].metadata["num_options"] == 4


def test_get_sample_predictions_ordered_by_sample_idx(repo: ResultsRepository) -> None:
    # Insert in reverse order; should come back sorted by sample_idx
    repo.save_sample_predictions("job-001", [_make_prediction(2), _make_prediction(0), _make_prediction(1)])
    fetched = repo.get_sample_predictions("job-001")
    assert [p.sample_idx for p in fetched] == [0, 1, 2]


def test_unparseable_prediction_stored_as_none(repo: ResultsRepository) -> None:
    pred = _make_prediction(0, predicted_letter=None, is_correct=False)
    repo.save_sample_predictions("job-001", [pred])
    fetched = repo.get_sample_predictions("job-001")
    assert fetched[0].predicted_letter is None
