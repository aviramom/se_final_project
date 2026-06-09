from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fmeval.services.evaluation_service import EvaluationService
from fmeval.services.types import EvaluationConfig, GroupedResult, ResultsFilter
from fmeval.storage.models import EvaluationResult
from fmeval.storage.repository import ResultsRepository


def _make_result(**kwargs: object) -> EvaluationResult:
    defaults: dict[str, object] = dict(
        job_id="job-001",
        model_name="mock-model",
        benchmark_name="ts-exam-1",
        modality="multimodal",
        metrics={
            "accuracy": 0.8,
            "balanced_accuracy": 0.75,
            "f1_macro": 0.70,
            "f1_weighted": 0.72,
            "n_samples": 50,
            "n_unparseable": 2,
        },
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


# ---------------------------------------------------------------------------
# query_results — filter routing
# ---------------------------------------------------------------------------

class TestQueryResults:
    def _service(self, repo: ResultsRepository) -> EvaluationService:
        return EvaluationService(
            model_registry=MagicMock(),
            benchmark_registry=MagicMock(),
            runner=MagicMock(),
            repository=repo,
        )

    def test_no_filter_returns_all(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save(_make_result(job_id="j2", exp_id="exp-b"))
        svc = self._service(repo)
        assert len(svc.query_results(ResultsFilter())) == 2

    def test_filter_by_exp_ids(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1", exp_id="a"))
        repo.save(_make_result(job_id="j2", exp_id="b"))
        repo.save(_make_result(job_id="j3", exp_id="c"))
        svc = self._service(repo)
        results = svc.query_results(ResultsFilter(exp_ids=["a", "c"]))
        assert {r.exp_id for r in results} == {"a", "c"}

    def test_filter_by_model_names(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1", model_name="x"))
        repo.save(_make_result(job_id="j2", model_name="y"))
        svc = self._service(repo)
        results = svc.query_results(ResultsFilter(model_names=["x"]))
        assert all(r.model_name == "x" for r in results)

    def test_filter_by_benchmark_names(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1", benchmark_name="bm-1"))
        repo.save(_make_result(job_id="j2", benchmark_name="bm-2"))
        svc = self._service(repo)
        results = svc.query_results(ResultsFilter(benchmark_names=["bm-2"]))
        assert all(r.benchmark_name == "bm-2" for r in results)

    def test_empty_list_means_no_restriction(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1", exp_id="a"))
        repo.save(_make_result(job_id="j2", exp_id="b"))
        svc = self._service(repo)
        # empty list → treated as None → no filter
        results = svc.query_results(ResultsFilter(exp_ids=[]))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# group_results — aggregation
# ---------------------------------------------------------------------------

class TestGroupResults:
    def _service(self) -> EvaluationService:
        return EvaluationService(
            model_registry=MagicMock(),
            benchmark_registry=MagicMock(),
            runner=MagicMock(),
            repository=MagicMock(),
        )

    def test_single_run_group_std_is_zero(self) -> None:
        svc = self._service()
        results = [_make_result(job_id="j1", exp_id="e1")]
        groups = svc.group_results(results)
        assert len(groups) == 1
        assert groups[0].std_metrics["accuracy"] == 0.0
        assert groups[0].n_runs == 1

    def test_two_runs_same_group(self) -> None:
        svc = self._service()
        results = [
            _make_result(job_id="j1", exp_id="e1", metrics={"accuracy": 0.8, "n_samples": 50, "n_unparseable": 1}),
            _make_result(job_id="j2", exp_id="e1", metrics={"accuracy": 0.6, "n_samples": 50, "n_unparseable": 3}),
        ]
        groups = svc.group_results(results)
        assert len(groups) == 1
        g = groups[0]
        assert g.n_runs == 2
        assert g.mean_metrics["accuracy"] == pytest.approx(0.7)
        assert g.std_metrics["accuracy"] > 0
        # n_samples and n_unparseable should be summed
        assert g.mean_metrics["n_samples"] == 100
        assert g.mean_metrics["n_unparseable"] == 4

    def test_two_different_groups(self) -> None:
        svc = self._service()
        results = [
            _make_result(job_id="j1", exp_id="e1"),
            _make_result(job_id="j2", exp_id="e2"),
        ]
        groups = svc.group_results(results)
        assert len(groups) == 2
        exp_ids = {g.exp_id for g in groups}
        assert exp_ids == {"e1", "e2"}

    def test_empty_input(self) -> None:
        svc = self._service()
        assert svc.group_results([]) == []


# ---------------------------------------------------------------------------
# list_exp_ids
# ---------------------------------------------------------------------------

def test_list_exp_ids_delegates_to_repo(repo: ResultsRepository) -> None:
    repo.save(_make_result(job_id="j1", exp_id="beta"))
    repo.save(_make_result(job_id="j2", exp_id="alpha"))
    svc = EvaluationService(
        model_registry=MagicMock(),
        benchmark_registry=MagicMock(),
        runner=MagicMock(),
        repository=repo,
    )
    assert svc.list_exp_ids() == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# exp_id auto-slug in run_evaluation
# ---------------------------------------------------------------------------

def test_run_evaluation_auto_slug_when_blank(tmp_path: Path) -> None:
    """Blank exp_id in config should produce a 'run-YYYYMMDD-HHMMSS' slug."""
    from fmeval.execution.job import EvaluationJob, JobStatus
    from fmeval.evaluation.result import RunResult

    mock_model = MagicMock()
    mock_model.supported_modalities = ["multimodal"]
    model_registry = MagicMock()
    model_registry.get.return_value = mock_model

    mock_dataset = MagicMock()
    mock_dataset.modality = "multimodal"
    benchmark_registry = MagicMock()
    benchmark_registry.get.return_value = mock_dataset

    mock_runner = MagicMock()
    mock_runner.submit.return_value = MagicMock()

    repo = ResultsRepository(tmp_path / "test.db")
    svc = EvaluationService(
        model_registry=model_registry,
        benchmark_registry=benchmark_registry,
        runner=mock_runner,
        repository=repo,
    )

    job_id = svc.run_evaluation(EvaluationConfig(
        model_name="m", benchmark_name="b", max_samples=10, exp_id=""
    ))
    job = svc._jobs[job_id]
    assert job.exp_id.startswith("run-")
    assert len(job.exp_id) == len("run-20260609-120000")


# ---------------------------------------------------------------------------
# get_run_detail — delegates to repository
# ---------------------------------------------------------------------------

def test_get_run_detail_returns_predictions(tmp_path: Path) -> None:
    """get_run_detail should return whatever the repository has for that job."""
    from fmeval.evaluation.result import SamplePrediction

    repo = ResultsRepository(tmp_path / "test.db")
    predictions = [
        SamplePrediction(
            sample_idx=0, raw_prediction="A) yes", raw_target="A) yes",
            predicted_letter="A", correct_letter="A", is_correct=True,
            input_text="Q0: <TS_0> What?",
        ),
        SamplePrediction(
            sample_idx=1, raw_prediction="B) no", raw_target="A) yes",
            predicted_letter="B", correct_letter="A", is_correct=False,
            input_text="Q1: <TS_0> What?",
        ),
    ]
    repo.save_sample_predictions("job-abc", predictions)

    svc = EvaluationService(
        model_registry=MagicMock(),
        benchmark_registry=MagicMock(),
        runner=MagicMock(),
        repository=repo,
    )
    detail = svc.get_run_detail("job-abc")
    assert len(detail) == 2
    assert detail[0].input_text == "Q0: <TS_0> What?"
    assert detail[1].is_correct is False


def test_get_run_detail_empty_for_unknown_job(tmp_path: Path) -> None:
    repo = ResultsRepository(tmp_path / "test.db")
    svc = EvaluationService(
        model_registry=MagicMock(),
        benchmark_registry=MagicMock(),
        runner=MagicMock(),
        repository=repo,
    )
    assert svc.get_run_detail("no-such-job") == []


def test_run_evaluation_preserves_exp_id(tmp_path: Path) -> None:
    """Non-blank exp_id should be preserved exactly."""
    mock_model = MagicMock()
    mock_model.supported_modalities = ["multimodal"]
    model_registry = MagicMock()
    model_registry.get.return_value = mock_model

    mock_dataset = MagicMock()
    mock_dataset.modality = "multimodal"
    benchmark_registry = MagicMock()
    benchmark_registry.get.return_value = mock_dataset

    mock_runner = MagicMock()
    mock_runner.submit.return_value = MagicMock()

    repo = ResultsRepository(tmp_path / "test.db")
    svc = EvaluationService(
        model_registry=model_registry,
        benchmark_registry=benchmark_registry,
        runner=mock_runner,
        repository=repo,
    )

    job_id = svc.run_evaluation(EvaluationConfig(
        model_name="m", benchmark_name="b", max_samples=10, exp_id="my-experiment"
    ))
    assert svc._jobs[job_id].exp_id == "my-experiment"
