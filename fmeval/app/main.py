"""Streamlit entry point for the Foundation Model Evaluation Platform.

Run with:
    streamlit run fmeval/app/main.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from fmeval.config import build_default_benchmark_registry, build_default_model_registry
from fmeval.execution.mock_runner import MockRunner
from fmeval.execution.runner import Runner
from fmeval.services.evaluation_service import EvaluationService
from fmeval.storage.repository import ResultsRepository


def _build_runner() -> Runner:
    if os.environ.get("FMEVAL_RUNNER", "mock").lower() == "slurm":
        from fmeval.execution.slurm_config import SlurmConfig
        from fmeval.execution.slurm_runner import SlurmRunner

        env_cmds = ["module load anaconda/anaconda"]
        # Forward CHATTS_MODEL_PATH into the sbatch script so the cluster worker
        # loads weights from a pre-downloaded path instead of fetching from HF Hub.
        chatts_path = os.environ.get("CHATTS_MODEL_PATH")
        if chatts_path:
            env_cmds.append(f"export CHATTS_MODEL_PATH={chatts_path}")

        cfg = SlurmConfig(
            host=os.environ.get("SLURM_HOST", "slurm.bgu.ac.il"),
            user=os.environ.get("SLURM_USER", "aviramom"),
            remote_work_dir=os.environ.get("SLURM_WORK_DIR", "/home/aviramom/fmeval_jobs"),
            ssh_key_path=os.environ.get("SLURM_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
            partition=os.environ.get("SLURM_PARTITION", "cpu"),
            time_limit=os.environ.get("SLURM_TIME_LIMIT", "01:00:00"),
            gpus_per_node=int(os.environ.get("SLURM_GPUS", "0")),
            gpu_type=os.environ.get("SLURM_GPU_TYPE"),  # e.g. "rtx_3090"
            cpus_per_task=int(os.environ.get("SLURM_CPUS", "2")),
            mem_gb=int(os.environ.get("SLURM_MEM_GB", "16")),
            python_bin=os.environ.get(
                "SLURM_PYTHON_BIN", "/home/aviramom/fmeval_project/.venv/bin/python"
            ),
            fmeval_dir=os.environ.get("SLURM_FMEVAL_DIR", "/home/aviramom/fmeval_project"),
            env_setup_commands=env_cmds,
        )
        return SlurmRunner(cfg)
    return MockRunner()


@st.cache_resource
def get_service() -> EvaluationService:
    """Create a single EvaluationService shared across all Streamlit rerenders."""
    db_path = Path(__file__).parent.parent.parent / "data" / "results.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return EvaluationService(
        model_registry=build_default_model_registry(),
        benchmark_registry=build_default_benchmark_registry(),
        runner=_build_runner(),
        repository=ResultsRepository(db_path),
    )


def main() -> None:
    st.set_page_config(
        page_title="FM Eval Platform",
        page_icon="📊",
        layout="wide",
    )
    st.title("Foundation Model Evaluation Platform")

    service = get_service()

    tab_config, tab_dashboard = st.tabs(["Run Evaluation", "Dashboard"])

    with tab_config:
        from fmeval.app.config_page import render_config_page

        render_config_page(service)

    with tab_dashboard:
        from fmeval.app.dashboard_page import render_dashboard_page

        render_dashboard_page(service)


if __name__ == "__main__":
    main()
