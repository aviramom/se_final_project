from __future__ import annotations

import pandas as pd
import streamlit as st

from fmeval.services.evaluation_service import EvaluationService
from fmeval.services.types import EvaluationConfig


def render_config_page(service: EvaluationService) -> None:
    """Config tab: select model + benchmark, set parameters, launch evaluation."""
    st.header("Run a New Evaluation")

    models = service.list_models()
    benchmarks = service.list_benchmarks()

    if not models:
        st.warning("No models registered.")
        return
    if not benchmarks:
        st.warning("No benchmarks registered.")
        return

    model_map = {m.display_name: m.name for m in models}
    benchmark_map = {b.display_name: b.name for b in benchmarks}

    col_left, col_right = st.columns(2)
    with col_left:
        selected_model_display = st.selectbox("Model", list(model_map))
    with col_right:
        selected_benchmark_display = st.selectbox("Benchmark", list(benchmark_map))

    max_samples = st.slider(
        "Max samples",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of dataset samples to evaluate. Lower = faster.",
    )

    exp_id_input = st.text_input(
        "Experiment ID",
        value="",
        placeholder="e.g. seed_42, ablation-lr-1e-4  (leave blank for auto)",
        help="Tag this run with a label. Runs sharing the same ID are grouped in the dashboard.",
        max_chars=64,
    )

    if st.button("Run Evaluation", type="primary", use_container_width=True):
        config = EvaluationConfig(
            model_name=model_map[selected_model_display],
            benchmark_name=benchmark_map[selected_benchmark_display],
            max_samples=max_samples,
            exp_id=exp_id_input.strip(),
        )
        try:
            job_id = service.run_evaluation(config)
            # Reflect the resolved exp_id (may be auto-slug if input was blank)
            job = next((j for j in service.poll_jobs() if j.job_id == job_id), None)
            resolved_exp = job.exp_id if job else exp_id_input.strip()
            st.success(f"Evaluation started — Job ID: `{job_id}` · Exp: `{resolved_exp}`")
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    st.divider()
    st.subheader("Recent Jobs (this session)")

    jobs = service.poll_jobs()  # advances state while refreshing the table
    if not jobs:
        st.info("No jobs yet. Run an evaluation above.")
    else:
        rows = [
            {
                "Job ID": j.job_id,
                "Exp ID": j.exp_id,
                "Model": j.model_name,
                "Benchmark": j.benchmark_name,
                "Samples": j.max_samples,
                "Status": j.status.value,
                "Started": j.created_at.strftime("%H:%M:%S"),
                "Error": j.error_message or "",
            }
            for j in sorted(jobs, key=lambda x: x.created_at, reverse=True)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
