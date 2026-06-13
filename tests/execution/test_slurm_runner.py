"""Tests for SlurmRunner.

All SSH and SCP calls are mocked via unittest.mock so these run fully offline.
The tests cover:
  - sbatch script generation (pure function, no I/O)
  - sbatch output parsing
  - submit() flow: directory creation, file uploads, sbatch submission
  - get_status() with various squeue outputs and file-existence fallback
  - get_result() JSON parsing
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from fmeval.execution.job import EvaluationJob, JobStatus
from fmeval.execution.slurm_config import SlurmConfig
from fmeval.execution.slurm_runner import SlurmHandle, SlurmRunner, _parse_run_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> SlurmConfig:
    return SlurmConfig(
        host="slurm.example.com",
        user="testuser",
        remote_work_dir="/home/testuser/fmeval_jobs",
        ssh_key_path="/home/me/.ssh/id_rsa",
        partition="gpu",
        time_limit="01:00:00",
        gpus_per_node=1,
        cpus_per_task=4,
        mem_gb=16,
        python_bin="/home/testuser/.venv/bin/python",
        fmeval_dir="/home/testuser/fmeval",
        env_setup_commands=["module load cuda/12.1"],
    )


@pytest.fixture()
def runner(cfg: SlurmConfig) -> SlurmRunner:
    return SlurmRunner(cfg)


@pytest.fixture()
def job() -> EvaluationJob:
    return EvaluationJob(
        job_id="abcd1234-5678-0000-0000-000000000000",
        model_name="mock_always_a",
        benchmark_name="tsexam1",
        modality="multimodal",
        status=JobStatus.QUEUED,
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        max_samples=50,
    )


@pytest.fixture()
def handle() -> SlurmHandle:
    return SlurmHandle(
        slurm_job_id="99999",
        remote_job_dir="/home/testuser/fmeval_jobs/abcd1234-5678-0000-0000-000000000000",
        result_json="/home/testuser/fmeval_jobs/abcd1234-5678-0000-0000-000000000000/result.json",
    )


# ---------------------------------------------------------------------------
# _parse_sbatch_output
# ---------------------------------------------------------------------------


class TestParseSbatchOutput:
    def test_standard_output(self, runner: SlurmRunner) -> None:
        assert (
            runner._parse_sbatch_output("Submitted batch job 12345678\n") == "12345678"
        )

    def test_extra_whitespace(self, runner: SlurmRunner) -> None:
        assert (
            runner._parse_sbatch_output("  Submitted batch job   987654  \n")
            == "987654"
        )

    def test_no_job_id_raises(self, runner: SlurmRunner) -> None:
        with pytest.raises(RuntimeError, match="Could not parse"):
            runner._parse_sbatch_output("Error: something went wrong\n")


# ---------------------------------------------------------------------------
# _build_sbatch_script
# ---------------------------------------------------------------------------


class TestBuildSbatchScript:
    def test_contains_required_directives(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=fmeval_abcd1234" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --mem=16G" in script
        assert "module load cuda/12.1" in script
        assert "export PYTHONPATH=/home/testuser/fmeval:$PYTHONPATH" in script

    def test_contains_worker_invocation(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--dataset tsexam1" in script
        assert "--model mock_always_a" in script
        assert "--max-samples 50" in script
        assert "--output /jobs/abc/result.json" in script

    def test_no_max_samples_omits_flag(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        job.max_samples = 0  # falsy → no --max-samples flag
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--max-samples" not in script

    def test_few_shot_params_forwarded(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        job.dataset_params = {
            "num_shots": 3,
            "picking_strategy": "first",
            "random_seed": 7,
        }
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="icl_ucr_GunPoint",
            model_name="random_label",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--num-shots 3" in script
        assert "--picking-strategy first" in script
        assert "--random-seed 7" in script

    def test_no_dataset_params_omits_few_shot_flags(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        # Default job (empty dataset_params) → no few-shot flags emitted.
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--num-shots" not in script
        assert "--picking-strategy" not in script
        assert "--random-seed" not in script

    def test_no_partition_omits_directive(
        self, cfg: SlurmConfig, job: EvaluationJob
    ) -> None:
        cfg.partition = None
        runner = SlurmRunner(cfg)
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--partition" not in script

    def test_cpu_only_omits_gres(self, cfg: SlurmConfig, job: EvaluationJob) -> None:
        cfg.gpus_per_node = 0
        runner = SlurmRunner(cfg)
        script = runner._build_sbatch_script(
            job=job,
            dataset_name="tsexam1",
            model_name="mock_always_a",
            remote_job_dir="/jobs/abc",
            result_json="/jobs/abc/result.json",
        )
        assert "--gres" not in script


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_calls_ssh_and_scp_in_order(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        mock_dataset = MagicMock()
        mock_dataset.name = "tsexam1"
        mock_model = MagicMock()
        mock_model.model_name = "mock_always_a"

        with (
            patch.object(
                runner, "_ssh", return_value="Submitted batch job 55555\n"
            ) as mock_ssh,
            patch.object(runner, "_scp_to_remote") as mock_scp,
        ):
            handle = runner.submit(job, mock_dataset, mock_model)

        # mkdir -p must be the first SSH call
        assert mock_ssh.call_args_list[0] == call(
            f"mkdir -p /home/testuser/fmeval_jobs/{job.job_id}"
        )
        # Two SCP uploads: worker.py and job.sh
        assert mock_scp.call_count == 2

        assert handle.slurm_job_id == "55555"
        assert handle.remote_job_dir == f"/home/testuser/fmeval_jobs/{job.job_id}"
        assert handle.result_json.endswith("result.json")

    def test_submit_returns_slurm_handle(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        mock_dataset = MagicMock(name="tsexam1")
        mock_dataset.name = "tsexam1"
        mock_model = MagicMock()
        mock_model.model_name = "mock_always_a"

        with (
            patch.object(runner, "_ssh", return_value="Submitted batch job 42\n"),
            patch.object(runner, "_scp_to_remote"),
        ):
            handle = runner.submit(job, mock_dataset, mock_model)

        assert isinstance(handle, SlurmHandle)
        assert handle.slurm_job_id == "42"


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------


class TestGetStatus:
    @pytest.mark.parametrize(
        "slurm_state,expected",
        [
            ("PENDING", JobStatus.QUEUED),
            ("CONFIGURING", JobStatus.QUEUED),
            ("RUNNING", JobStatus.RUNNING),
            ("COMPLETING", JobStatus.RUNNING),
            ("COMPLETED", JobStatus.COMPLETED),
            ("FAILED", JobStatus.FAILED),
            ("CANCELLED", JobStatus.FAILED),
            ("TIMEOUT", JobStatus.FAILED),
        ],
    )
    def test_known_states(
        self,
        runner: SlurmRunner,
        job: EvaluationJob,
        handle: SlurmHandle,
        slurm_state: str,
        expected: JobStatus,
    ) -> None:
        job.handle = handle
        with patch.object(runner, "_ssh", return_value=slurm_state + "\n"):
            assert runner.get_status(job) == expected

    def test_job_not_in_queue_result_exists_returns_completed(
        self, runner: SlurmRunner, job: EvaluationJob, handle: SlurmHandle
    ) -> None:
        job.handle = handle
        # First call: squeue returns empty (job left the queue)
        # Second call: file-existence check returns EXISTS
        with patch.object(runner, "_ssh", side_effect=["", "EXISTS\n"]):
            assert runner.get_status(job) == JobStatus.COMPLETED

    def test_job_not_in_queue_result_missing_returns_failed(
        self, runner: SlurmRunner, job: EvaluationJob, handle: SlurmHandle
    ) -> None:
        job.handle = handle
        with patch.object(runner, "_ssh", side_effect=["", "MISSING\n"]):
            assert runner.get_status(job) == JobStatus.FAILED

    def test_no_handle_returns_failed(
        self, runner: SlurmRunner, job: EvaluationJob
    ) -> None:
        job.handle = None
        assert runner.get_status(job) == JobStatus.FAILED


# ---------------------------------------------------------------------------
# get_result()
# ---------------------------------------------------------------------------

SAMPLE_RESULT_JSON = {
    "model_name": "mock_always_a",
    "dataset_name": "tsexam1",
    "timestamp": "2024-01-01T12:00:00",
    "num_samples": 10,
    "metrics": {"accuracy": 0.25, "n_samples": 10.0, "n_unparseable": 0.0},
    "run_config": {"model_name": "mock_always_a"},
    "sample_predictions": [
        {
            "sample_idx": 0,
            "raw_prediction": "A",
            "raw_target": "B",
            "predicted_letter": "A",
            "correct_letter": "B",
            "is_correct": False,
            "input_text": "What is X?",
            "metadata": {"difficulty": "easy"},
        }
    ],
}


class TestGetResult:
    def test_parses_run_result_correctly(
        self, runner: SlurmRunner, job: EvaluationJob, handle: SlurmHandle, tmp_path
    ) -> None:
        job.handle = handle
        json_file = tmp_path / "result.json"
        json_file.write_text(json.dumps(SAMPLE_RESULT_JSON))

        def fake_scp_from_remote(remote_path: str, local_path: str, **_) -> None:
            import shutil

            shutil.copy(json_file, local_path)

        with patch.object(runner, "_scp_from_remote", side_effect=fake_scp_from_remote):
            result = runner.get_result(job)

        assert result.model_name == "mock_always_a"
        assert result.dataset_name == "tsexam1"
        assert result.num_samples == 10
        assert result.metrics["accuracy"] == pytest.approx(0.25)
        assert len(result.sample_predictions) == 1
        assert result.sample_predictions[0].metadata == {"difficulty": "easy"}

    def test_no_handle_raises(self, runner: SlurmRunner, job: EvaluationJob) -> None:
        job.handle = None
        with pytest.raises(RuntimeError, match="no Slurm handle"):
            runner.get_result(job)


# ---------------------------------------------------------------------------
# _parse_run_result (unit test for the JSON → RunResult converter)
# ---------------------------------------------------------------------------


class TestParseRunResult:
    def test_full_round_trip(self) -> None:
        result = _parse_run_result(SAMPLE_RESULT_JSON)
        assert result.model_name == "mock_always_a"
        assert result.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert result.sample_predictions[0].input_text == "What is X?"
        assert result.sample_predictions[0].metadata["difficulty"] == "easy"

    def test_missing_optional_fields_use_defaults(self) -> None:
        minimal = {
            "model_name": "m",
            "dataset_name": "d",
            "timestamp": "2024-06-01T00:00:00",
            "num_samples": 1,
            "metrics": {},
            "sample_predictions": [],
        }
        result = _parse_run_result(minimal)
        assert result.run_config == {}
        assert result.sample_predictions == []


# ---------------------------------------------------------------------------
# Handle serialization — reattach support across app restarts
# ---------------------------------------------------------------------------


class TestHandleSerialization:
    def test_roundtrip(self, runner: SlurmRunner) -> None:
        handle = SlurmHandle(
            slurm_job_id="12345",
            remote_job_dir="/home/testuser/fmeval_jobs/abc",
            result_json="/home/testuser/fmeval_jobs/abc/result.json",
        )
        serialized = runner.serialize_handle(handle)
        assert serialized is not None
        assert runner.deserialize_handle(serialized) == handle

    def test_serialize_none_returns_none(self, runner: SlurmRunner) -> None:
        assert runner.serialize_handle(None) is None

    def test_serialize_foreign_handle_returns_none(self, runner: SlurmRunner) -> None:
        assert runner.serialize_handle(object()) is None

    def test_serialized_form_is_json(self, runner: SlurmRunner) -> None:
        handle = SlurmHandle("1", "/dir", "/dir/result.json")
        serialized = runner.serialize_handle(handle)
        assert serialized is not None
        assert json.loads(serialized)["slurm_job_id"] == "1"

    def test_runner_type_is_slurm(self, runner: SlurmRunner) -> None:
        assert runner.runner_type == "slurm"
