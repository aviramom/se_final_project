from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fmeval.services.evaluation_service import EvaluationService
from fmeval.services.types import EvaluationConfig, ResultsFilter
from fmeval.storage.models import EvaluationResult, JobRecord
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
    mock_runner.runner_type = "mock"
    mock_runner.serialize_handle.return_value = None

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
    mock_runner.runner_type = "mock"
    mock_runner.serialize_handle.return_value = None

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


# ---------------------------------------------------------------------------
# Shared helpers for the new feature tests
# ---------------------------------------------------------------------------

def _service_with_repo(repo: ResultsRepository, runner: MagicMock | None = None) -> EvaluationService:
    if runner is None:
        runner = MagicMock()
        runner.runner_type = "mock"
        runner.serialize_handle.return_value = None
    return EvaluationService(
        model_registry=MagicMock(),
        benchmark_registry=MagicMock(),
        runner=runner,
        repository=repo,
    )


def _make_prediction(idx: int, *, correct: bool, letter: str = "A", **kwargs: object):
    from fmeval.evaluation.result import SamplePrediction

    defaults: dict[str, object] = dict(
        sample_idx=idx,
        raw_prediction=f"{letter}) answer",
        raw_target="A) answer",
        predicted_letter=letter,
        correct_letter="A",
        is_correct=correct,
        input_text=f"Q{idx}: <TS_0> What?",
        metadata={"difficulty": "easy" if idx % 2 == 0 else "hard"},
    )
    defaults.update(kwargs)
    return SamplePrediction(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# update_run / delete_run
# ---------------------------------------------------------------------------

class TestEditDelete:
    def test_update_run_changes_exp_id_and_notes(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        svc = _service_with_repo(repo)
        assert svc.update_run("j1", exp_id="renamed", notes="a note") is True
        fetched = repo.get_by_job_id("j1")
        assert fetched is not None
        assert (fetched.exp_id, fetched.notes) == ("renamed", "a note")

    def test_update_run_unknown_job(self, repo: ResultsRepository) -> None:
        svc = _service_with_repo(repo)
        assert svc.update_run("nope", exp_id="x") is False

    def test_delete_run_removes_everything(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save_sample_predictions("j1", [_make_prediction(0, correct=True)])
        svc = _service_with_repo(repo)
        assert svc.delete_run("j1") is True
        assert repo.get_by_job_id("j1") is None
        assert repo.get_sample_predictions("j1") == []
        assert "j1" not in svc._jobs

    def test_delete_runs_bulk_returns_count(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save(_make_result(job_id="j2"))
        svc = _service_with_repo(repo)
        assert svc.delete_runs(["j1", "j2", "ghost"]) == 2


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def _seed_two_runs(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="run-a", model_name="model-a"))
        repo.save(_make_result(job_id="run-b", model_name="model-b"))
        # A: correct on 0,1 — wrong on 2,3. B: correct on 0,2 — wrong on 1,3.
        repo.save_sample_predictions("run-a", [
            _make_prediction(0, correct=True),
            _make_prediction(1, correct=True),
            _make_prediction(2, correct=False, letter="B"),
            _make_prediction(3, correct=False, letter="C"),
        ])
        repo.save_sample_predictions("run-b", [
            _make_prediction(0, correct=True),
            _make_prediction(1, correct=False, letter="D"),
            _make_prediction(2, correct=True),
            _make_prediction(3, correct=False, letter="B"),
        ])

    def test_agreement_counts(self, repo: ResultsRepository) -> None:
        self._seed_two_runs(repo)
        svc = _service_with_repo(repo)
        cmp = svc.compare_runs("run-a", "run-b")
        assert cmp.n_common == 4
        assert cmp.both_correct == 1     # sample 0
        assert cmp.only_a_correct == 1   # sample 1
        assert cmp.only_b_correct == 1   # sample 2
        assert cmp.both_wrong == 1       # sample 3
        assert len(cmp.diffs) == 4

    def test_benchmark_mismatch_raises(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="run-a", benchmark_name="bm-1"))
        repo.save(_make_result(job_id="run-b", benchmark_name="bm-2"))
        svc = _service_with_repo(repo)
        with pytest.raises(ValueError, match="different benchmarks"):
            svc.compare_runs("run-a", "run-b")

    def test_missing_run_raises(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="run-a"))
        svc = _service_with_repo(repo)
        with pytest.raises(ValueError, match="not found"):
            svc.compare_runs("run-a", "ghost")

    def test_differing_sample_counts_use_intersection(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="run-a"))
        repo.save(_make_result(job_id="run-b"))
        repo.save_sample_predictions("run-a", [
            _make_prediction(0, correct=True),
            _make_prediction(1, correct=True),
            _make_prediction(2, correct=True),
        ])
        repo.save_sample_predictions("run-b", [_make_prediction(0, correct=True)])
        svc = _service_with_repo(repo)
        cmp = svc.compare_runs("run-a", "run-b")
        assert cmp.n_common == 1
        assert cmp.both_correct == 1


# ---------------------------------------------------------------------------
# category breakdown + metadata/metric key discovery
# ---------------------------------------------------------------------------

class TestBreakdowns:
    def test_breakdown_by_difficulty(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save_sample_predictions("j1", [
            _make_prediction(0, correct=True),    # easy
            _make_prediction(2, correct=True),    # easy
            _make_prediction(1, correct=False, letter="B"),  # hard
        ])
        svc = _service_with_repo(repo)
        bd = svc.get_category_breakdown("j1", "difficulty")
        rows = {r.value: r for r in bd.rows}
        assert rows["easy"].n_samples == 2
        assert rows["easy"].accuracy == pytest.approx(1.0)
        assert rows["hard"].n_samples == 1
        assert rows["hard"].accuracy == pytest.approx(0.0)

    def test_breakdown_skips_samples_missing_key(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save_sample_predictions("j1", [
            _make_prediction(0, correct=True, metadata={"category": "trend"}),
            _make_prediction(1, correct=False, letter="B", metadata={}),
        ])
        svc = _service_with_repo(repo)
        bd = svc.get_category_breakdown("j1", "category")
        assert len(bd.rows) == 1
        assert bd.rows[0].n_samples == 1

    def test_list_metadata_keys_union(self, repo: ResultsRepository) -> None:
        repo.save_sample_predictions("j1", [
            _make_prediction(0, correct=True, metadata={"a": 1}),
            _make_prediction(1, correct=True, metadata={"b": 2}),
        ])
        svc = _service_with_repo(repo)
        assert svc.list_metadata_keys("j1") == ["a", "b"]

    def test_get_category_breakdowns_skips_runs_without_key(self, repo: ResultsRepository) -> None:
        repo.save(_make_result(job_id="j1"))
        repo.save(_make_result(job_id="j2"))
        repo.save_sample_predictions("j1", [_make_prediction(0, correct=True)])
        repo.save_sample_predictions("j2", [
            _make_prediction(0, correct=True, metadata={})
        ])
        svc = _service_with_repo(repo)
        breakdowns = svc.get_category_breakdowns(["j1", "j2"], "difficulty")
        assert [b.job_id for b in breakdowns] == ["j1"]

    def test_list_metric_keys_excludes_counts_and_orders_headline_first(self) -> None:
        service = EvaluationService(
            model_registry=MagicMock(),
            benchmark_registry=MagicMock(),
            runner=MagicMock(),
            repository=MagicMock(),
        )
        results = [_make_result(metrics={
            "accuracy": 0.8, "f1_macro": 0.7, "n_samples": 50,
            "n_unparseable": 2, "f1_A": 0.9,
        })]
        keys = service.list_metric_keys(results)
        assert "n_samples" not in keys
        assert "n_unparseable" not in keys
        assert keys[0] == "accuracy"
        assert "f1_A" in keys


# ---------------------------------------------------------------------------
# job persistence — restore across restarts
# ---------------------------------------------------------------------------

class TestJobRestore:
    def _record(self, **kwargs: object) -> JobRecord:
        defaults: dict[str, object] = dict(
            job_id="job-1",
            model_name="m",
            benchmark_name="b",
            modality="multimodal",
            status="running",
            created_at=datetime(2026, 6, 12, 9, 0, 0),
            max_samples=10,
            exp_id="e",
            runner_type="slurm",
            handle_json='{"slurm_job_id": "99"}',
            error_message=None,
        )
        defaults.update(kwargs)
        return JobRecord(**defaults)  # type: ignore[arg-type]

    def test_terminal_jobs_restored_as_is(self, repo: ResultsRepository) -> None:
        repo.save_job(self._record(status="completed"))
        svc = _service_with_repo(repo)
        job = svc._jobs["job-1"]
        assert job.status.value == "completed"

    def test_running_slurm_job_reattached(self, repo: ResultsRepository) -> None:
        repo.save_job(self._record(status="running", runner_type="slurm"))
        runner = MagicMock()
        runner.runner_type = "slurm"
        runner.deserialize_handle.return_value = {"slurm_job_id": "99"}
        svc = _service_with_repo(repo, runner=runner)
        job = svc._jobs["job-1"]
        assert job.status.value == "running"
        assert job.handle == {"slurm_job_id": "99"}
        runner.deserialize_handle.assert_called_once_with('{"slurm_job_id": "99"}')

    def test_running_mock_job_orphaned(self, repo: ResultsRepository) -> None:
        repo.save_job(self._record(status="running", runner_type="mock", handle_json=None))
        svc = _service_with_repo(repo)  # mock runner
        job = svc._jobs["job-1"]
        assert job.status.value == "failed"
        assert "Orphaned" in (job.error_message or "")
        # persisted too
        assert repo.load_jobs()[0].status == "failed"

    def test_runner_type_mismatch_orphaned(self, repo: ResultsRepository) -> None:
        repo.save_job(self._record(status="running", runner_type="slurm"))
        svc = _service_with_repo(repo)  # active runner is mock
        assert svc._jobs["job-1"].status.value == "failed"

    def test_run_evaluation_persists_job(self, repo: ResultsRepository) -> None:
        mock_model = MagicMock()
        mock_model.supported_modalities = ["multimodal"]
        model_registry = MagicMock()
        model_registry.get.return_value = mock_model
        mock_dataset = MagicMock()
        mock_dataset.modality = "multimodal"
        benchmark_registry = MagicMock()
        benchmark_registry.get.return_value = mock_dataset
        runner = MagicMock()
        runner.runner_type = "mock"
        runner.serialize_handle.return_value = None

        svc = EvaluationService(
            model_registry=model_registry,
            benchmark_registry=benchmark_registry,
            runner=runner,
            repository=repo,
        )
        job_id = svc.run_evaluation(
            EvaluationConfig(model_name="m", benchmark_name="b", max_samples=5)
        )
        stored = repo.load_jobs()
        assert len(stored) == 1
        assert stored[0].job_id == job_id
        assert stored[0].runner_type == "mock"
