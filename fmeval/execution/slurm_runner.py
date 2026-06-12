from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fmeval.core.datasets.base import Dataset
from fmeval.core.models.base import ModelWrapper
from fmeval.evaluation.result import RunResult, SamplePrediction
from fmeval.execution.job import EvaluationJob, JobStatus
from fmeval.execution.runner import Runner
from fmeval.execution.slurm_config import SlurmConfig

logger = logging.getLogger(__name__)

# Map Slurm job state strings to our JobStatus enum.
_SLURM_STATE_MAP: dict[str, JobStatus] = {
    "PENDING": JobStatus.QUEUED,
    "CONFIGURING": JobStatus.QUEUED,
    "SUSPENDED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "CANCELLED": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "PREEMPTED": JobStatus.FAILED,
}


@dataclass
class SlurmHandle:
    """Runner-specific handle stored on EvaluationJob.handle for Slurm jobs.

    slurm_job_id:   The numeric Slurm job ID returned by sbatch.
    remote_job_dir: {remote_work_dir}/{fmeval_job_id}/ on the cluster.
    result_json:    Path on the cluster where the worker writes result.json.
    """

    slurm_job_id: str
    remote_job_dir: str
    result_json: str


class SlurmRunner(Runner):
    """Submits evaluation jobs to a Slurm cluster over SSH.

    Job lifecycle:
      submit()     — uploads worker.py + job.sh, runs sbatch, stores SlurmHandle
      get_status() — polls squeue; if job has left the queue checks result file existence
      get_result() — SCPs result.json back, parses into RunResult

    SSH authentication must be key-based (no interactive password prompt).
    Configure ssh_key_path in SlurmConfig or ensure the key is loaded in ssh-agent.
    """

    def __init__(self, config: SlurmConfig) -> None:
        self._cfg = config
        # The cluster_worker.py file lives next to this module and is uploaded per-job.
        self._worker_local_path = Path(__file__).parent / "cluster_worker.py"

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------

    def submit(self, job: EvaluationJob, dataset: Dataset, model: ModelWrapper) -> SlurmHandle:
        """Upload scripts, submit via sbatch, return a SlurmHandle with the job ID."""
        remote_job_dir = f"{self._cfg.remote_work_dir}/{job.job_id}"
        result_json = f"{remote_job_dir}/result.json"

        # Create the job directory on the cluster.
        self._ssh(f"mkdir -p {remote_job_dir}")

        # Upload the worker script.
        self._scp_to_remote(str(self._worker_local_path), f"{remote_job_dir}/worker.py")

        # Build and upload the sbatch script.
        sbatch_content = self._build_sbatch_script(
            job=job,
            dataset_name=dataset.name,
            model_name=model.model_name,
            remote_job_dir=remote_job_dir,
            result_json=result_json,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            sbatch_local = f.name
            f.write(sbatch_content)
        try:
            self._scp_to_remote(sbatch_local, f"{remote_job_dir}/job.sh")
        finally:
            Path(sbatch_local).unlink(missing_ok=True)

        # Submit.
        stdout = self._ssh(f"sbatch {remote_job_dir}/job.sh")
        slurm_job_id = self._parse_sbatch_output(stdout)

        logger.info(
            "Submitted Slurm job %s for fmeval job %s (dataset=%s model=%s)",
            slurm_job_id, job.job_id, dataset.name, model.model_name,
        )

        return SlurmHandle(
            slurm_job_id=slurm_job_id,
            remote_job_dir=remote_job_dir,
            result_json=result_json,
        )

    def get_status(self, job: EvaluationJob) -> JobStatus:
        """Poll squeue; fall back to checking whether result.json exists."""
        handle: SlurmHandle | None = job.handle
        if handle is None:
            return JobStatus.FAILED

        try:
            # -h suppresses the header; -o '%T' prints only the state column.
            stdout = self._ssh(
                f"squeue -j {handle.slurm_job_id} -h -o '%T' 2>/dev/null || true"
            )
            state = stdout.strip().upper()
        except RuntimeError as exc:
            logger.warning("squeue failed for job %s: %s", job.job_id, exc)
            state = ""

        if state in _SLURM_STATE_MAP:
            return _SLURM_STATE_MAP[state]

        # Job has left the Slurm queue — decide by whether output was written.
        try:
            out = self._ssh(
                f"test -f {handle.result_json} && echo EXISTS || echo MISSING"
            )
            if "EXISTS" in out:
                return JobStatus.COMPLETED
        except RuntimeError as exc:
            logger.warning("File-existence check failed for job %s: %s", job.job_id, exc)

        return JobStatus.FAILED

    def get_result(self, job: EvaluationJob) -> RunResult:
        """SCP result.json from the cluster and parse it into a RunResult."""
        handle: SlurmHandle | None = job.handle
        if handle is None:
            raise RuntimeError(f"Job {job.job_id} has no Slurm handle — was it submitted?")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_json = Path(tmpdir) / "result.json"
            self._scp_from_remote(handle.result_json, str(local_json))
            with open(local_json) as f:
                data = json.load(f)

        return _parse_run_result(data)

    # ------------------------------------------------------------------
    # SSH / SCP helpers
    # ------------------------------------------------------------------

    def _ssh_base_args(self) -> list[str]:
        """Build the ssh command prefix (with key path if configured)."""
        args = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
        ]
        if self._cfg.ssh_key_path:
            args += ["-i", self._cfg.ssh_key_path]
        args.append(f"{self._cfg.user}@{self._cfg.host}")
        return args

    def _scp_base_args(self) -> list[str]:
        """Build the scp command prefix (with key path if configured)."""
        args = [
            "scp",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
        ]
        if self._cfg.ssh_key_path:
            args += ["-i", self._cfg.ssh_key_path]
        return args

    def _ssh(self, command: str, timeout: int = 60) -> str:
        """Run a shell command on the cluster; return stdout. Raise on non-zero exit."""
        cmd = self._ssh_base_args() + [command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (exit {result.returncode})\n"
                f"  command : {command}\n"
                f"  stderr  : {result.stderr.strip()}"
            )
        return result.stdout

    def _scp_to_remote(self, local_path: str, remote_path: str, timeout: int = 120) -> None:
        """Upload a local file to {user}@{host}:{remote_path}."""
        dest = f"{self._cfg.user}@{self._cfg.host}:{remote_path}"
        cmd = self._scp_base_args() + [local_path, dest]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP upload failed: {local_path} → {remote_path}\n"
                f"  stderr: {result.stderr.strip()}"
            )

    def _scp_from_remote(self, remote_path: str, local_path: str, timeout: int = 120) -> None:
        """Download {user}@{host}:{remote_path} to a local path."""
        src = f"{self._cfg.user}@{self._cfg.host}:{remote_path}"
        cmd = self._scp_base_args() + [src, local_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"SCP download failed: {remote_path} → {local_path}\n"
                f"  stderr: {result.stderr.strip()}"
            )

    # ------------------------------------------------------------------
    # sbatch script generation
    # ------------------------------------------------------------------

    def _build_sbatch_script(
        self,
        job: EvaluationJob,
        dataset_name: str,
        model_name: str,
        remote_job_dir: str,
        result_json: str,
    ) -> str:
        cfg = self._cfg
        lines: list[str] = ["#!/bin/bash"]

        # SBATCH directives — use first 8 chars of job_id to keep job names short.
        lines += [
            f"#SBATCH --job-name=fmeval_{job.job_id[:8]}",
            f"#SBATCH --output={remote_job_dir}/slurm_%j.out",
            f"#SBATCH --error={remote_job_dir}/slurm_%j.err",
            f"#SBATCH --time={cfg.time_limit}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
            f"#SBATCH --mem={cfg.mem_gb}G",
        ]
        if cfg.partition:
            lines.append(f"#SBATCH --partition={cfg.partition}")
        if cfg.gpus_per_node > 0:
            gres = f"gpu:{cfg.gpu_type}:{cfg.gpus_per_node}" if cfg.gpu_type else f"gpu:{cfg.gpus_per_node}"
            lines.append(f"#SBATCH --gres={gres}")
        for directive in cfg.extra_sbatch_directives:
            lines.append(directive)

        lines.append("")

        # Optional environment setup (module loads, conda activate, etc.)
        for cmd in cfg.env_setup_commands:
            lines.append(cmd)

        # Prepend fmeval repo to PYTHONPATH if an editable install path is given.
        if cfg.fmeval_dir:
            lines.append(f"export PYTHONPATH={cfg.fmeval_dir}:$PYTHONPATH")

        lines.append("")

        # Build the Python invocation, one argument per line for readability.
        cmd_parts = [
            f"{cfg.python_bin} {remote_job_dir}/worker.py",
            f"    --dataset {dataset_name}",
            f"    --model {model_name}",
        ]
        if job.max_samples:
            cmd_parts.append(f"    --max-samples {job.max_samples}")
        cmd_parts.append(f"    --output {result_json}")

        lines.append(" \\\n".join(cmd_parts))
        lines += ["", 'echo "FMEVAL_EXIT_CODE:$?"', ""]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sbatch_output(stdout: str) -> str:
        """Extract the numeric Slurm job ID from sbatch's stdout.

        sbatch prints: "Submitted batch job 12345678"
        """
        for token in stdout.split():
            if token.isdigit():
                return token
        raise RuntimeError(
            f"Could not parse a numeric Slurm job ID from sbatch output: {stdout!r}"
        )


# ------------------------------------------------------------------
# RunResult reconstruction from JSON
# ------------------------------------------------------------------

def _parse_run_result(data: dict[str, Any]) -> RunResult:
    """Reconstruct a RunResult from the JSON dict written by cluster_worker.py."""
    sample_predictions = [
        SamplePrediction(
            sample_idx=sp["sample_idx"],
            raw_prediction=sp["raw_prediction"],
            raw_target=sp["raw_target"],
            predicted_letter=sp["predicted_letter"],
            correct_letter=sp["correct_letter"],
            is_correct=sp["is_correct"],
            input_text=sp.get("input_text", ""),
            metadata=sp.get("metadata", {}),
        )
        for sp in data.get("sample_predictions", [])
    ]
    return RunResult(
        model_name=data["model_name"],
        dataset_name=data["dataset_name"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        num_samples=data["num_samples"],
        metrics=data["metrics"],
        sample_predictions=sample_predictions,
        run_config=data.get("run_config", {}),
    )
