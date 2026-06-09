from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from fmeval.services.evaluation_service import EvaluationService
from fmeval.services.types import ResultsFilter
from fmeval.storage.models import EvaluationResult

_HEADLINE_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Acc",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
}


def _init_session_state() -> None:
    defaults: dict[str, object] = {
        "dash_filter_exp_ids": [],
        "dash_filter_models": [],
        "dash_filter_benchmarks": [],
        "dash_date_from": None,
        "dash_date_to": None,
        "dash_view_mode": "Individual runs",
        "selected_job_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_dashboard_page(service: EvaluationService) -> None:
    """Dashboard tab: filter panel, job statuses, comparison chart and table, CSV export."""
    st.header("Dashboard")

    _init_session_state()

    # --- Top action row ---
    col_refresh, col_export = st.columns([1, 3])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            service.poll_jobs()
            st.rerun()
    with col_export:
        csv_str = service.export_csv()
        st.download_button(
            label="Export CSV",
            data=csv_str,
            file_name="evaluation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --- Jobs table (session-scoped, always shown) ---
    st.subheader("Jobs (this session)")
    session_jobs = service.poll_jobs()
    if session_jobs:
        job_rows = [
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
            for j in sorted(session_jobs, key=lambda x: x.created_at, reverse=True)
        ]
        st.dataframe(pd.DataFrame(job_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No jobs in this session. Go to **Run Evaluation** to start one.")

    st.divider()

    # --- Filter panel ---
    st.subheader("Results")
    with st.expander("Filters", expanded=True):
        available_exp_ids = service.list_exp_ids()
        available_models = [m.name for m in service.list_models()]
        available_benchmarks = [b.name for b in service.list_benchmarks()]

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.multiselect(
                "Experiment IDs",
                options=available_exp_ids,
                key="dash_filter_exp_ids",
                placeholder="All experiments",
            )
        with fcol2:
            st.multiselect(
                "Models",
                options=available_models,
                key="dash_filter_models",
                placeholder="All models",
            )
        with fcol3:
            st.multiselect(
                "Benchmarks",
                options=available_benchmarks,
                key="dash_filter_benchmarks",
                placeholder="All benchmarks",
            )

        dcol1, dcol2, dcol3 = st.columns([2, 2, 1])
        with dcol1:
            date_from = st.date_input("From date", value=None, key="dash_date_from")
        with dcol2:
            date_to = st.date_input("To date", value=None, key="dash_date_to")
        with dcol3:
            st.write("")  # vertical alignment spacer
            if st.button("Clear Filters", use_container_width=True):
                for key in (
                    "dash_filter_exp_ids",
                    "dash_filter_models",
                    "dash_filter_benchmarks",
                    "dash_date_from",
                    "dash_date_to",
                ):
                    st.session_state[key] = [] if "filter" in key else None
                st.rerun()

    # Build filter and fetch results
    from datetime import datetime

    date_from_dt = (
        datetime.combine(date_from, datetime.min.time())
        if date_from
        else None
    )
    date_to_dt = (
        datetime.combine(date_to, datetime.max.time().replace(microsecond=0))
        if date_to
        else None
    )

    filters = ResultsFilter(
        exp_ids=st.session_state.dash_filter_exp_ids or None,
        model_names=st.session_state.dash_filter_models or None,
        benchmark_names=st.session_state.dash_filter_benchmarks or None,
        date_from=date_from_dt,
        date_to=date_to_dt,
    )
    filtered = service.query_results(filters)

    if not filtered:
        any_filter_active = any([
            st.session_state.dash_filter_exp_ids,
            st.session_state.dash_filter_models,
            st.session_state.dash_filter_benchmarks,
            date_from,
            date_to,
        ])
        if any_filter_active:
            st.info("No results match the current filters. Adjust the filter panel above.")
        else:
            st.info("No completed results yet. Run an evaluation to see results here.")
        return

    # If a run is selected for detail, show that view instead of chart + table
    if st.session_state.selected_job_id:
        result_map = {r.job_id: r for r in filtered}
        selected_result = result_map.get(st.session_state.selected_job_id)
        if selected_result:
            _render_run_detail(service, selected_result)
            return
        # Job no longer in filtered set (filter changed) — clear selection
        st.session_state.selected_job_id = None

    # --- View mode toggle ---
    view_mode = st.radio(
        "View",
        ["Individual runs", "Grouped by Exp ID"],
        horizontal=True,
        key="dash_view_mode",
    )

    if view_mode == "Individual runs":
        _render_individual(filtered)
    else:
        _render_grouped(service, filtered)


def _render_individual(results: list) -> None:
    """Chart and table for individual (ungrouped) results. Click a row to see sample details."""
    rows = [
        {
            "Exp ID": r.exp_id,
            "Model": r.model_name,
            "Benchmark": r.benchmark_name,
            "Accuracy": round(r.metrics.get("accuracy", 0.0), 4),
            "Balanced Acc": round(r.metrics.get("balanced_accuracy", 0.0), 4),
            "F1 Macro": round(r.metrics.get("f1_macro", 0.0), 4),
            "F1 Weighted": round(r.metrics.get("f1_weighted", 0.0), 4),
            "Samples": int(r.metrics.get("n_samples", 0)),
            "Unparseable": int(r.metrics.get("n_unparseable", 0)),
            "Run At": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Exec (s)": round(r.execution_time_seconds, 1),
            "_job_id": r.job_id,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    # Bar chart
    chart_df = (
        df.sort_values("Run At")
        .drop_duplicates(subset=["Model", "Benchmark", "Exp ID"], keep="last")
        .copy()
    )
    fig = px.bar(
        chart_df,
        x="Model",
        y="Accuracy",
        color="Benchmark",
        barmode="group",
        hover_data={"Exp ID": True, "_job_id": True},
        title="Accuracy by Model (latest run per model × benchmark × Exp ID)",
        range_y=[0, 1],
        text_auto=".3f",
        height=400,
    )
    fig.update_layout(legend_title_text="Benchmark")
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table — row selection enabled to drill into per-sample details
    st.subheader("All Results")
    st.caption("Click any row to inspect per-sample questions and answers.")
    display_df = df.drop(columns=["_job_id"])
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="results_table",
    )

    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_job_id = rows[selected_idx]["_job_id"]
        st.session_state.selected_job_id = selected_job_id
        st.rerun()


def _render_run_detail(service: EvaluationService, result: EvaluationResult) -> None:
    """Full-page detail view for a single completed run."""
    # Header + back navigation
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.selected_job_id = None
            st.rerun()
    with col_title:
        st.subheader(f"Run Detail — {result.job_id}")

    # Run metadata
    st.markdown(
        f"**Model:** {result.model_name} &nbsp;·&nbsp; "
        f"**Benchmark:** {result.benchmark_name} &nbsp;·&nbsp; "
        f"**Exp ID:** {result.exp_id or '—'} &nbsp;·&nbsp; "
        f"**Samples:** {result.max_samples} &nbsp;·&nbsp; "
        f"**Run at:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; "
        f"**Exec time:** {result.execution_time_seconds:.1f}s"
    )

    # Aggregate metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.metrics.get('accuracy', 0.0):.4f}")
    m2.metric("Balanced Acc", f"{result.metrics.get('balanced_accuracy', 0.0):.4f}")
    m3.metric("F1 Macro", f"{result.metrics.get('f1_macro', 0.0):.4f}")
    m4.metric("F1 Weighted", f"{result.metrics.get('f1_weighted', 0.0):.4f}")

    st.divider()

    # Per-sample predictions
    predictions = service.get_run_detail(result.job_id)
    if not predictions:
        st.info(
            "Sample-level data is not available for this run. "
            "Only runs completed after this feature was deployed store per-sample details."
        )
        return

    st.subheader(f"Sample Predictions ({len(predictions)} samples)")

    # Correct/incorrect filter
    filter_mode = st.radio(
        "Show",
        ["All", "Correct only", "Incorrect only"],
        horizontal=True,
        key="detail_filter",
    )
    if filter_mode == "Correct only":
        predictions = [p for p in predictions if p.is_correct]
    elif filter_mode == "Incorrect only":
        predictions = [p for p in predictions if not p.is_correct]

    if not predictions:
        st.info("No samples match the selected filter.")
        return

    # Detect dynamic metadata keys from the first sample
    meta_keys: list[str] = []
    if predictions[0].metadata:
        meta_keys = list(predictions[0].metadata.keys())

    sample_rows = []
    for p in predictions:
        row: dict[str, object] = {
            "#": p.sample_idx,
            "Question": (p.input_text[:120] + "…") if len(p.input_text) > 120 else p.input_text,
            "Correct": p.correct_letter,
            "Predicted": p.predicted_letter if p.predicted_letter is not None else "—",
            "Result": "✓" if p.is_correct else "✗",
        }
        for key in meta_keys:
            row[key] = p.metadata.get(key, "")
        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)

    def _highlight_result(col: pd.Series) -> list[str]:
        if col.name != "Result":
            return [""] * len(col)
        return [
            "background-color: #d4edda; color: #155724" if v == "✓"
            else "background-color: #f8d7da; color: #721c24"
            for v in col
        ]

    styled = sample_df.style.apply(_highlight_result, axis=0)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Expandable raw prediction viewer
    with st.expander("Show full question text and raw model outputs"):
        for p in predictions:
            result_icon = "✓" if p.is_correct else "✗"
            st.markdown(f"**Sample #{p.sample_idx}** &nbsp; {result_icon}")
            st.text(p.input_text)
            col_pred, col_target = st.columns(2)
            with col_pred:
                st.markdown("**Model output:**")
                st.code(p.raw_prediction, language=None)
            with col_target:
                st.markdown("**Ground truth:**")
                st.code(p.raw_target, language=None)
            st.divider()


def _render_grouped(service: EvaluationService, results: list) -> None:
    """Chart and table for grouped (exp_id × model × benchmark) results."""
    groups = service.group_results(results)

    if not groups:
        st.info("No groups to display.")
        return

    # Build numeric DataFrame for styling + chart
    numeric_rows = []
    for g in groups:
        row: dict[str, object] = {
            "Exp ID": g.exp_id,
            "Model": g.model_name,
            "Benchmark": g.benchmark_name,
            "Runs": g.n_runs,
        }
        for key in _HEADLINE_METRICS:
            label = _METRIC_LABELS[key]
            row[label] = round(g.mean_metrics.get(key, 0.0), 4)
            row[f"{label} Std"] = round(g.std_metrics.get(key, 0.0), 4)
        row["Total Samples"] = int(g.mean_metrics.get("n_samples", 0))
        row["Unparseable"] = int(g.mean_metrics.get("n_unparseable", 0))
        numeric_rows.append(row)

    numeric_df = pd.DataFrame(numeric_rows)

    # Bar chart with error bars on Accuracy
    fig = px.bar(
        numeric_df,
        x="Model",
        y="Accuracy",
        error_y="Accuracy Std",
        color="Benchmark",
        barmode="group",
        facet_col="Exp ID" if numeric_df["Exp ID"].nunique() > 1 else None,
        title="Accuracy by Model (mean ± std across runs in each Exp ID)",
        range_y=[0, 1],
        text_auto=".3f",
        height=420,
    )
    fig.update_layout(legend_title_text="Benchmark")
    st.plotly_chart(fig, use_container_width=True)

    # Grouped comparison table — show "mean ± std" strings, highlight best means
    st.subheader("Grouped Results (mean ± std)")
    display_rows = []
    for g in groups:
        row = {
            "Exp ID": g.exp_id,
            "Model": g.model_name,
            "Benchmark": g.benchmark_name,
            "Runs": g.n_runs,
        }
        for key in _HEADLINE_METRICS:
            label = _METRIC_LABELS[key]
            mean = g.mean_metrics.get(key, 0.0)
            std = g.std_metrics.get(key, 0.0)
            row[label] = f"{mean:.4f} ± {std:.4f}"
        row["Total Samples"] = int(g.mean_metrics.get("n_samples", 0))
        row["Unparseable"] = int(g.mean_metrics.get("n_unparseable", 0))
        display_rows.append(row)

    display_df = pd.DataFrame(display_rows)

    highlight_cols = list(_METRIC_LABELS.values())
    numeric_mean_cols = {_METRIC_LABELS[k]: numeric_df[_METRIC_LABELS[k]] for k in _HEADLINE_METRICS}

    # Apply green highlight to the rows with the highest mean per metric column
    def _highlight_best(col: pd.Series) -> list[str]:
        numeric_col = numeric_mean_cols.get(col.name)
        if numeric_col is None or len(numeric_col) == 0:
            return [""] * len(col)
        max_val = numeric_col.max()
        return [
            "background-color: #d4edda" if numeric_col.iloc[i] == max_val else ""
            for i in range(len(col))
        ]

    styled = display_df.style.apply(
        _highlight_best, subset=highlight_cols, axis=0
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
