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

    # Group benchmarks for a two-step picker: the 95 UCR datasets collapse into a
    # handful of category groups instead of one giant flat dropdown.
    groups: dict[str, list] = {}
    for b in benchmarks:
        groups.setdefault(b.group, []).append(b)

    col_left, col_right = st.columns(2)
    with col_left:
        selected_model_display = st.selectbox("Model", list(model_map))
    with col_right:
        selected_group = st.selectbox("Benchmark group", sorted(groups))

    group_benchmarks = sorted(groups[selected_group], key=lambda b: b.short_name)
    if len(group_benchmarks) == 1:
        selected_benchmark = group_benchmarks[0]
    else:
        short_map = {b.short_name: b for b in group_benchmarks}
        selected_short = st.selectbox("Dataset", list(short_map))
        selected_benchmark = short_map[selected_short]

    max_samples = st.slider(
        "Max samples",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of dataset samples to evaluate. Lower = faster.",
    )

    # Few-shot ICL controls — shown only for benchmarks that support them.
    num_shots, picking_strategy, random_seed = 1, "random", 0
    if selected_benchmark.supports_few_shot:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            num_shots = st.number_input(
                "Shots per class (k)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Labeled support examples per class included in the prompt.",
            )
        with fc2:
            picking_strategy = st.selectbox(
                "Support selection",
                ["random", "first", "reversed"],
                help="How support examples are drawn from the training split.",
            )
        with fc3:
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=10_000,
                value=0,
                step=1,
                help="Seeds support sampling and the test subsample for reproducibility.",
            )

    exp_id_input = st.text_input(
        "Experiment ID",
        value="",
        placeholder="e.g. seed_42, ablation-lr-1e-4  (leave blank for auto)",
        help="Tag this run with a label. Runs sharing the same ID are grouped in the dashboard.",
        max_chars=64,
    )

    if st.button("Run Evaluation", type="primary", width="stretch"):
        config = EvaluationConfig(
            model_name=model_map[selected_model_display],
            benchmark_name=selected_benchmark.name,
            max_samples=max_samples,
            exp_id=exp_id_input.strip(),
            num_shots=int(num_shots),
            picking_strategy=picking_strategy,
            random_seed=int(random_seed),
        )
        try:
            job_id = service.run_evaluation(config)
            # Reflect the resolved exp_id (may be auto-slug if input was blank)
            job = next((j for j in service.poll_jobs() if j.job_id == job_id), None)
            resolved_exp = job.exp_id if job else exp_id_input.strip()
            st.success(
                f"Evaluation started — Job ID: `{job_id}` · Exp: `{resolved_exp}`"
            )
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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
