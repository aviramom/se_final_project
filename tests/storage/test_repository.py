from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from fmeval.evaluation.result import SamplePrediction
from fmeval.storage.models import EvaluationResult, JobRecord
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


# ---------------------------------------------------------------------------
# notes — roundtrip + legacy-DB migration
# ---------------------------------------------------------------------------

def test_notes_roundtrip(repo: ResultsRepository) -> None:
    repo.save(_make_result(notes="baseline run, seed 42"))
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert fetched.notes == "baseline run, seed 42"


def test_notes_default_empty(repo: ResultsRepository) -> None:
    repo.save(_make_result())
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert fetched.notes == ""


def test_notes_migration_on_legacy_db(tmp_path: Path) -> None:
    """A DB created before the notes column existed must upgrade on open."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE evaluation_results (
            job_id TEXT PRIMARY KEY, model_name TEXT NOT NULL,
            benchmark_name TEXT NOT NULL, modality TEXT NOT NULL,
            metrics TEXT NOT NULL, timestamp TEXT NOT NULL,
            execution_time_seconds REAL NOT NULL,
            exp_id TEXT NOT NULL DEFAULT '',
            max_samples INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        "INSERT INTO evaluation_results VALUES (?,?,?,?,?,?,?,?,?)",
        ("old-job", "m", "b", "multimodal", '{"accuracy": 0.5}',
         "2026-01-01T00:00:00", 1.0, "e", 10),
    )
    conn.commit()
    conn.close()

    repo = ResultsRepository(db)
    fetched = repo.get_by_job_id("old-job")
    assert fetched is not None
    assert fetched.notes == ""


def test_export_csv_includes_notes(repo: ResultsRepository) -> None:
    repo.save(_make_result(notes="my special note"))
    csv = repo.export_csv(repo.query())
    assert "notes" in csv
    assert "my special note" in csv


# ---------------------------------------------------------------------------
# update_run / delete_run
# ---------------------------------------------------------------------------

def test_update_run_exp_id_only(repo: ResultsRepository) -> None:
    repo.save(_make_result(notes="keep me"))
    assert repo.update_run("job-001", exp_id="renamed") is True
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert fetched.exp_id == "renamed"
    assert fetched.notes == "keep me"  # untouched


def test_update_run_notes_only(repo: ResultsRepository) -> None:
    repo.save(_make_result(exp_id="keep-exp"))
    assert repo.update_run("job-001", notes="new note") is True
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert fetched.notes == "new note"
    assert fetched.exp_id == "keep-exp"


def test_update_run_both_fields(repo: ResultsRepository) -> None:
    repo.save(_make_result())
    assert repo.update_run("job-001", exp_id="new-exp", notes="new note") is True
    fetched = repo.get_by_job_id("job-001")
    assert fetched is not None
    assert (fetched.exp_id, fetched.notes) == ("new-exp", "new note")


def test_update_run_unknown_job_returns_false(repo: ResultsRepository) -> None:
    assert repo.update_run("nonexistent", exp_id="x") is False


def test_update_run_nothing_to_set_returns_false(repo: ResultsRepository) -> None:
    repo.save(_make_result())
    assert repo.update_run("job-001") is False


def test_delete_run_removes_result_and_predictions(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1"))
    repo.save(_make_result(job_id="j2"))
    repo.save_sample_predictions("j1", [_make_prediction(0)])
    repo.save_sample_predictions("j2", [_make_prediction(0)])

    assert repo.delete_run("j1") is True
    assert repo.get_by_job_id("j1") is None
    assert repo.get_sample_predictions("j1") == []
    # other job untouched
    assert repo.get_by_job_id("j2") is not None
    assert len(repo.get_sample_predictions("j2")) == 1


def test_delete_run_unknown_job_returns_false(repo: ResultsRepository) -> None:
    assert repo.delete_run("nonexistent") is False


# ---------------------------------------------------------------------------
# jobs table — save / load / status updates
# ---------------------------------------------------------------------------

def _make_job_record(**kwargs: object) -> JobRecord:
    defaults: dict[str, object] = dict(
        job_id="job-001",
        model_name="mock-model",
        benchmark_name="ts-exam-1",
        modality="multimodal",
        status="running",
        created_at=datetime(2026, 6, 12, 10, 0, 0),
        max_samples=50,
        exp_id="seed_42",
        runner_type="slurm",
        handle_json='{"slurm_job_id": "123"}',
        error_message=None,
    )
    defaults.update(kwargs)
    return JobRecord(**defaults)  # type: ignore[arg-type]


def test_save_and_load_jobs_roundtrip(repo: ResultsRepository) -> None:
    repo.save_job(_make_job_record())
    jobs = repo.load_jobs()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.job_id == "job-001"
    assert j.status == "running"
    assert j.runner_type == "slurm"
    assert j.handle_json == '{"slurm_job_id": "123"}'
    assert j.error_message is None
    assert j.created_at == datetime(2026, 6, 12, 10, 0, 0)


def test_save_job_upserts(repo: ResultsRepository) -> None:
    repo.save_job(_make_job_record(status="queued"))
    repo.save_job(_make_job_record(status="running"))
    jobs = repo.load_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == "running"


def test_update_job_status(repo: ResultsRepository) -> None:
    repo.save_job(_make_job_record(status="running"))
    repo.update_job_status("job-001", "failed", "out of memory")
    j = repo.load_jobs()[0]
    assert j.status == "failed"
    assert j.error_message == "out of memory"


def test_load_jobs_newest_first(repo: ResultsRepository) -> None:
    repo.save_job(_make_job_record(job_id="old", created_at=datetime(2026, 1, 1)))
    repo.save_job(_make_job_record(job_id="new", created_at=datetime(2026, 6, 1)))
    jobs = repo.load_jobs()
    assert [j.job_id for j in jobs] == ["new", "old"]


def test_delete_job(repo: ResultsRepository) -> None:
    repo.save_job(_make_job_record())
    repo.delete_job("job-001")
    assert repo.load_jobs() == []
