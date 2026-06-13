from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from fmeval.app.ui_components import (
    STATUS_COLORS,
    render_diff_card,
    render_qa_card,
)
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

_CARDS_PER_PAGE = 25


def _metric_label(key: str) -> str:
    return _METRIC_LABELS.get(key, key.replace("_", " ").title())


def _init_session_state() -> None:
    defaults: dict[str, object] = {
        "dash_filter_exp_ids": [],
        "dash_filter_models": [],
        "dash_filter_benchmarks": [],
        "dash_date_from": None,
        "dash_date_to": None,
        "dash_view_mode": "Individual runs",
        "selected_job_id": None,
        # Rotated after every delete: st.dataframe selections are index-based,
        # so a stale selection would point at the wrong rows otherwise.
        "results_table_nonce": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_dashboard_page(service: EvaluationService) -> None:
    """Dashboard tab: Runs / Compare / Trends / Jobs sub-tabs."""
    _init_session_state()

    tab_runs, tab_compare, tab_trends, tab_jobs = st.tabs(
        ["📊 Runs", "⚖️ Compare", "📈 Trends", "⚙️ Jobs"]
    )
    with tab_runs:
        _render_runs_tab(service)
    with tab_compare:
        _render_compare_tab(service)
    with tab_trends:
        _render_trends_tab(service)
    with tab_jobs:
        _render_jobs_tab(service)


# ---------------------------------------------------------------------------
# Runs tab — filters, metric chart, results table, drill-in, edit/delete
# ---------------------------------------------------------------------------

def _render_runs_tab(service: EvaluationService) -> None:
    filtered = _render_filters_and_query(service)
    if filtered is None:
        return

    # Drill-in: show single-run detail instead of chart + table.
    if st.session_state.selected_job_id:
        result_map = {r.job_id: r for r in filtered}
        selected_result = result_map.get(st.session_state.selected_job_id)
        if selected_result:
            _render_run_detail(service, selected_result)
            return
        # Run no longer in filtered set (filter changed or deleted) — clear.
        st.session_state.selected_job_id = None

    col_view, col_metric, col_export = st.columns([2, 2, 1])
    with col_view:
        view_mode = st.radio(
            "View",
            ["Individual runs", "Grouped by Exp ID"],
            horizontal=True,
            key="dash_view_mode",
        )
    with col_metric:
        metric_keys = service.list_metric_keys(filtered)
        metric = st.selectbox(
            "Chart metric",
            metric_keys,
            format_func=_metric_label,
            key="dash_chart_metric",
        )
    with col_export:
        st.write("")
        st.download_button(
            label="Export CSV",
            data=service.export_csv(),
            file_name="evaluation_results.csv",
            mime="text/csv",
            width="stretch",
        )

    if view_mode == "Individual runs":
        _render_individual(service, filtered, metric)
    else:
        _render_grouped(service, filtered, metric)


def _render_filters_and_query(
    service: EvaluationService,
) -> list[EvaluationResult] | None:
    """Filter panel + query. Returns None (after rendering a notice) if empty."""
    with st.expander("Filters", expanded=False):
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
            if st.button("Clear Filters", width="stretch"):
                for key in (
                    "dash_filter_exp_ids",
                    "dash_filter_models",
                    "dash_filter_benchmarks",
                    "dash_date_from",
                    "dash_date_to",
                ):
                    st.session_state[key] = [] if "filter" in key else None
                st.rerun()

    filters = ResultsFilter(
        exp_ids=st.session_state.dash_filter_exp_ids or None,
        model_names=st.session_state.dash_filter_models or None,
        benchmark_names=st.session_state.dash_filter_benchmarks or None,
        date_from=(
            datetime.combine(date_from, datetime.min.time()) if date_from else None
        ),
        date_to=(
            datetime.combine(date_to, datetime.max.time().replace(microsecond=0))
            if date_to
            else None
        ),
    )
    filtered = service.query_results(filters)

    if not filtered:
        any_filter_active = any(
            [
                st.session_state.dash_filter_exp_ids,
                st.session_state.dash_filter_models,
                st.session_state.dash_filter_benchmarks,
                date_from,
                date_to,
            ]
        )
        if any_filter_active:
            st.info("No results match the current filters. Adjust the filter panel above.")
        else:
            st.info("No completed results yet. Run an evaluation to see results here.")
        return None
    return filtered


def _apply_rate_range(values: list[float]) -> list[float] | None:
    """y-axis [0, 1] only when the metric looks like a rate, else autoscale."""
    return [0, 1] if values and all(0.0 <= v <= 1.0 for v in values) else None


@st.dialog("Confirm deletion")
def _confirm_delete_dialog(service: EvaluationService, job_ids: list[str], labels: list[str]) -> None:
    st.warning(
        f"Delete {len(job_ids)} run(s)? Metrics and per-sample predictions "
        "are removed permanently."
    )
    for label in labels:
        st.markdown(f"- {label}")
    col_ok, col_cancel = st.columns(2)
    if col_ok.button("Delete", type="primary", width="stretch"):
        n = service.delete_runs(job_ids)
        st.session_state.selected_job_id = None
        st.session_state.results_table_nonce += 1
        st.toast(f"Deleted {n} run(s).")
        st.rerun()
    if col_cancel.button("Cancel", width="stretch"):
        st.rerun()


def _render_individual(
    service: EvaluationService, results: list[EvaluationResult], metric: str
) -> None:
    """Metric chart + selectable results table with view/delete actions."""
    metric_name = _metric_label(metric)
    rows = [
        {
            "Exp ID": r.exp_id,
            "Model": r.model_name,
            "Benchmark": r.benchmark_name,
            metric_name: round(r.metrics.get(metric, 0.0), 4),
            "Accuracy": round(r.metrics.get("accuracy", 0.0), 4),
            "Balanced Acc": round(r.metrics.get("balanced_accuracy", 0.0), 4),
            "F1 Macro": round(r.metrics.get("f1_macro", 0.0), 4),
            "F1 Weighted": round(r.metrics.get("f1_weighted", 0.0), 4),
            "Samples": int(r.metrics.get("n_samples", 0)),
            "Unparseable": int(r.metrics.get("n_unparseable", 0)),
            "Run At": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Exec (s)": round(r.execution_time_seconds, 1),
            "Notes": (r.notes[:60] + "…") if len(r.notes) > 60 else r.notes,
            "_job_id": r.job_id,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)

    chart_df = (
        df.sort_values("Run At")
        .drop_duplicates(subset=["Model", "Benchmark", "Exp ID"], keep="last")
        .copy()
    )
    fig = px.bar(
        chart_df,
        x="Model",
        y=metric_name,
        color="Benchmark",
        barmode="group",
        hover_data={"Exp ID": True, "_job_id": True},
        title=f"{metric_name} by Model (latest run per model × benchmark × Exp ID)",
        range_y=_apply_rate_range(chart_df[metric_name].tolist()),
        text_auto=".3f",
        height=400,
    )
    fig.update_layout(legend_title_text="Benchmark")
    st.plotly_chart(fig, width="stretch")

    st.subheader("All Results")
    st.caption(
        "Select one row to view per-sample details or edit it; "
        "select multiple rows to bulk-delete."
    )
    display_df = df.drop(columns=["_job_id"])
    event = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"results_table_{st.session_state.results_table_nonce}",
    )

    selected_rows = list(event.selection.rows)
    if not selected_rows:
        return
    selected_ids = [rows[i]["_job_id"] for i in selected_rows]
    labels = [
        f"`{rows[i]['_job_id']}` — {rows[i]['Model']} on {rows[i]['Benchmark']} "
        f"({rows[i]['Exp ID']})"
        for i in selected_rows
    ]
    col_view, col_delete, _ = st.columns([1, 1, 3])
    with col_view:
        if len(selected_ids) == 1 and st.button(
            "View details", type="primary", width="stretch"
        ):
            st.session_state.selected_job_id = selected_ids[0]
            st.rerun()
    with col_delete:
        if st.button(f"Delete ({len(selected_ids)})", width="stretch"):
            _confirm_delete_dialog(service, selected_ids, labels)


def _render_grouped(
    service: EvaluationService, results: list[EvaluationResult], metric: str
) -> None:
    """Chart and table for grouped (exp_id × model × benchmark) results."""
    groups = service.group_results(results)
    if not groups:
        st.info("No groups to display.")
        return

    metric_name = _metric_label(metric)
    numeric_rows = []
    for g in groups:
        row: dict[str, object] = {
            "Exp ID": g.exp_id,
            "Model": g.model_name,
            "Benchmark": g.benchmark_name,
            "Runs": g.n_runs,
            metric_name: round(g.mean_metrics.get(metric, 0.0), 4),
            f"{metric_name} Std": round(g.std_metrics.get(metric, 0.0), 4),
        }
        for key in _HEADLINE_METRICS:
            label = _METRIC_LABELS[key]
            row.setdefault(label, round(g.mean_metrics.get(key, 0.0), 4))
            row.setdefault(f"{label} Std", round(g.std_metrics.get(key, 0.0), 4))
        row["Total Samples"] = int(g.mean_metrics.get("n_samples", 0))
        row["Unparseable"] = int(g.mean_metrics.get("n_unparseable", 0))
        numeric_rows.append(row)

    numeric_df = pd.DataFrame(numeric_rows)

    fig = px.bar(
        numeric_df,
        x="Model",
        y=metric_name,
        error_y=f"{metric_name} Std",
        color="Benchmark",
        barmode="group",
        facet_col="Exp ID" if numeric_df["Exp ID"].nunique() > 1 else None,
        title=f"{metric_name} by Model (mean ± std across runs in each Exp ID)",
        range_y=_apply_rate_range(numeric_df[metric_name].tolist()),
        text_auto=".3f",
        height=420,
    )
    fig.update_layout(legend_title_text="Benchmark")
    st.plotly_chart(fig, width="stretch")

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
    numeric_mean_cols = {
        _METRIC_LABELS[k]: numeric_df[_METRIC_LABELS[k]] for k in _HEADLINE_METRICS
    }

    def _highlight_best(col: pd.Series) -> list[str]:
        numeric_col = numeric_mean_cols.get(col.name)
        if numeric_col is None or len(numeric_col) == 0:
            return [""] * len(col)
        max_val = numeric_col.max()
        return [
            "background-color: #d4edda" if numeric_col.iloc[i] == max_val else ""
            for i in range(len(col))
        ]

    styled = display_df.style.apply(_highlight_best, subset=highlight_cols, axis=0)
    st.dataframe(styled, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Run detail — metrics, edit panel, category breakdown, Q/A cards
# ---------------------------------------------------------------------------

def _render_run_detail(service: EvaluationService, result: EvaluationResult) -> None:
    """Full-page detail view for a single completed run."""
    col_back, col_title, col_delete = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", width="stretch"):
            st.session_state.selected_job_id = None
            st.rerun()
    with col_title:
        st.subheader(f"Run Detail — {result.job_id}")
    with col_delete:
        if st.button("Delete run", width="stretch"):
            _confirm_delete_dialog(
                service,
                [result.job_id],
                [f"`{result.job_id}` — {result.model_name} on {result.benchmark_name}"],
            )

    st.markdown(
        f"**Model:** {result.model_name} &nbsp;·&nbsp; "
        f"**Benchmark:** {result.benchmark_name} &nbsp;·&nbsp; "
        f"**Exp ID:** {result.exp_id or '—'} &nbsp;·&nbsp; "
        f"**Samples:** {result.max_samples} &nbsp;·&nbsp; "
        f"**Run at:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; "
        f"**Exec time:** {result.execution_time_seconds:.1f}s"
    )
    if result.notes:
        st.info(f"📝 {result.notes}")

    with st.expander("✏️ Edit run (experiment label and notes)"):
        new_exp_id = st.text_input(
            "Experiment ID", value=result.exp_id, max_chars=64, key="edit_exp_id"
        )
        new_notes = st.text_area(
            "Notes",
            value=result.notes,
            placeholder="e.g. baseline before prompt tweak; ran on rtx_3090",
            key="edit_notes",
        )
        if st.button("Save changes", type="primary"):
            updated = service.update_run(
                result.job_id, exp_id=new_exp_id.strip(), notes=new_notes.strip()
            )
            if updated:
                st.toast("Run updated.")
                st.rerun()
            else:
                st.error("Update failed — run not found.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.metrics.get('accuracy', 0.0):.4f}")
    m2.metric("Balanced Acc", f"{result.metrics.get('balanced_accuracy', 0.0):.4f}")
    m3.metric("F1 Macro", f"{result.metrics.get('f1_macro', 0.0):.4f}")
    m4.metric("F1 Weighted", f"{result.metrics.get('f1_weighted', 0.0):.4f}")

    st.divider()

    predictions = service.get_run_detail(result.job_id)
    if not predictions:
        st.info(
            "Sample-level data is not available for this run. "
            "Only runs completed after this feature was deployed store per-sample details."
        )
        return

    _render_category_breakdown(service, result.job_id)

    st.subheader(f"Sample Predictions ({len(predictions)} samples)")

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

    meta_keys = list(predictions[0].metadata.keys()) if predictions[0].metadata else []
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
    st.dataframe(styled, width="stretch", hide_index=True)

    with st.expander("Show full question text and raw model outputs", expanded=False):
        page = _paginate(len(predictions), key="detail_cards_page")
        for p in predictions[page * _CARDS_PER_PAGE:(page + 1) * _CARDS_PER_PAGE]:
            render_qa_card(p)


def _render_category_breakdown(service: EvaluationService, job_id: str) -> None:
    """Accuracy sliced by a sample-metadata key chosen by the user."""
    meta_keys = service.list_metadata_keys(job_id)
    if not meta_keys:
        return
    st.subheader("Breakdown by Category")
    default_idx = next(
        (i for i, k in enumerate(meta_keys) if k in ("category", "difficulty")), 0
    )
    key = st.selectbox(
        "Slice accuracy by", meta_keys, index=default_idx, key="breakdown_key"
    )
    breakdown = service.get_category_breakdown(job_id, key)
    if not breakdown.rows:
        st.info(f"No samples carry the '{key}' metadata key.")
        return

    bd_df = pd.DataFrame(
        [
            {
                key: row.value,
                "Accuracy": row.accuracy,
                "Correct": row.n_correct,
                "Samples": row.n_samples,
            }
            for row in breakdown.rows
        ]
    )
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig = px.bar(
            bd_df,
            x=key,
            y="Accuracy",
            range_y=[0, 1],
            text_auto=".3f",
            hover_data={"Correct": True, "Samples": True},
            height=320,
        )
        st.plotly_chart(fig, width="stretch")
    with col_table:
        st.dataframe(
            bd_df.assign(Accuracy=bd_df["Accuracy"].round(4)),
            width="stretch",
            hide_index=True,
        )
    st.divider()


def _paginate(n_items: int, key: str) -> int:
    """Page selector for long card lists; returns the zero-based page index."""
    n_pages = (n_items + _CARDS_PER_PAGE - 1) // _CARDS_PER_PAGE
    if n_pages <= 1:
        return 0
    page = st.selectbox(
        f"Page ({_CARDS_PER_PAGE} samples per page)",
        list(range(1, n_pages + 1)),
        format_func=lambda p: f"{p} / {n_pages}",
        key=key,
    )
    return page - 1


# ---------------------------------------------------------------------------
# Compare tab — run-vs-run agreement and side-by-side answers
# ---------------------------------------------------------------------------

def _run_option_label(r: EvaluationResult) -> str:
    return (
        f"{r.exp_id or '—'} · {r.model_name} · {r.benchmark_name} · "
        f"{r.timestamp.strftime('%Y-%m-%d %H:%M')} · {r.job_id}"
    )


def _render_compare_tab(service: EvaluationService) -> None:
    st.caption(
        "Pick two runs on the same benchmark to see question-by-question "
        "agreement and side-by-side answers."
    )
    all_results = service.query_results(ResultsFilter())
    if len(all_results) < 2:
        st.info("Need at least two completed runs to compare.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        run_a = st.selectbox(
            "Run A",
            all_results,
            format_func=_run_option_label,
            key="compare_run_a",
        )
    candidates_b = [
        r
        for r in all_results
        if r.benchmark_name == run_a.benchmark_name and r.job_id != run_a.job_id
    ]
    with col_b:
        if not candidates_b:
            st.selectbox("Run B (same benchmark)", ["No other run on this benchmark"], disabled=True)
            return
        run_b = st.selectbox(
            "Run B (same benchmark)",
            candidates_b,
            format_func=_run_option_label,
            key="compare_run_b",
        )

    try:
        comparison = service.compare_runs(run_a.job_id, run_b.job_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    if comparison.n_common == 0:
        st.warning(
            "These runs share no per-sample data (older runs may predate "
            "sample-level storage)."
        )
        return

    label_a = f"A · {run_a.model_name}"
    label_b = f"B · {run_b.model_name}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Both correct", comparison.both_correct)
    c2.metric(f"Only {label_a} ✓", comparison.only_a_correct)
    c3.metric(f"Only {label_b} ✓", comparison.only_b_correct)
    c4.metric("Both wrong", comparison.both_wrong)
    st.caption(
        f"{comparison.n_common} common samples · "
        f"A accuracy {comparison.result_a.metrics.get('accuracy', 0.0):.4f} · "
        f"B accuracy {comparison.result_b.metrics.get('accuracy', 0.0):.4f}"
    )

    agreement_df = pd.DataFrame(
        [
            {
                "#": d.sample_idx,
                "Question": (d.input_text[:100] + "…") if len(d.input_text) > 100 else d.input_text,
                "Correct": d.correct_letter,
                label_a: f"{d.predicted_a or '—'} {'✓' if d.a_correct else '✗'}",
                label_b: f"{d.predicted_b or '—'} {'✓' if d.b_correct else '✗'}",
                "Agreement": d.agreement.replace("_", " "),
            }
            for d in comparison.diffs
        ]
    )

    _AGREEMENT_STYLE = {
        "both correct": "background-color: #d4edda",
        "both wrong": "background-color: #f8d7da",
        "only a correct": "background-color: #fff3cd",
        "only b correct": "background-color: #cfe2ff",
    }

    def _highlight_agreement(col: pd.Series) -> list[str]:
        if col.name != "Agreement":
            return [""] * len(col)
        return [_AGREEMENT_STYLE.get(v, "") for v in col]

    st.dataframe(
        agreement_df.style.apply(_highlight_agreement, axis=0),
        width="stretch",
        hide_index=True,
    )

    disagreements_only = st.toggle("Disagreements only", value=True, key="compare_diff_only")
    diffs = [
        d
        for d in comparison.diffs
        if not disagreements_only or d.a_correct != d.b_correct
    ]
    if not diffs:
        st.info("No disagreements — both runs answered every common sample the same way.")
        return
    with st.expander(f"Side-by-side answers ({len(diffs)} samples)", expanded=True):
        page = _paginate(len(diffs), key="compare_cards_page")
        for d in diffs[page * _CARDS_PER_PAGE:(page + 1) * _CARDS_PER_PAGE]:
            render_diff_card(d, label_a, label_b)


# ---------------------------------------------------------------------------
# Trends tab — metric over time per model × benchmark
# ---------------------------------------------------------------------------

def _render_trends_tab(service: EvaluationService) -> None:
    st.caption("Track a metric across repeated runs over time.")
    all_results = service.query_results(ResultsFilter())
    if not all_results:
        st.info("No completed results yet.")
        return

    col_metric, col_benchmark = st.columns(2)
    with col_metric:
        metric = st.selectbox(
            "Metric",
            service.list_metric_keys(all_results),
            format_func=_metric_label,
            key="trend_metric",
        )
    benchmarks = sorted({r.benchmark_name for r in all_results})
    with col_benchmark:
        benchmark = st.selectbox(
            "Benchmark", ["All"] + benchmarks, key="trend_benchmark"
        )

    results = [
        r
        for r in all_results
        if benchmark == "All" or r.benchmark_name == benchmark
    ]
    metric_name = _metric_label(metric)
    trend_df = pd.DataFrame(
        [
            {
                "Run At": r.timestamp,
                metric_name: r.metrics.get(metric, 0.0),
                "Model": r.model_name,
                "Benchmark": r.benchmark_name,
                "Exp ID": r.exp_id,
                "Job ID": r.job_id,
            }
            for r in sorted(results, key=lambda r: r.timestamp)
        ]
    )
    fig = px.line(
        trend_df,
        x="Run At",
        y=metric_name,
        color="Model",
        line_dash="Benchmark",
        markers=True,
        hover_data={"Exp ID": True, "Job ID": True},
        title=f"{metric_name} over time",
        range_y=_apply_rate_range(trend_df[metric_name].tolist()),
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Jobs tab — persisted job tracking with status badges
# ---------------------------------------------------------------------------

def _render_jobs_tab(service: EvaluationService) -> None:
    col_refresh, col_caption = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()
    with col_caption:
        st.caption(
            "Jobs persist across app restarts. Running Slurm jobs are "
            "reattached on startup; interrupted local (mock) jobs are marked failed."
        )

    jobs = service.poll_jobs()
    if not jobs:
        st.info("No jobs yet. Go to **Run Evaluation** to start one.")
        return

    counts = {status: 0 for status in STATUS_COLORS}
    for j in jobs:
        counts[j.status.value] = counts.get(j.status.value, 0) + 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queued", counts.get("queued", 0))
    c2.metric("Running", counts.get("running", 0))
    c3.metric("Completed", counts.get("completed", 0))
    c4.metric("Failed", counts.get("failed", 0))

    job_rows = [
        {
            "Job ID": j.job_id,
            "Exp ID": j.exp_id,
            "Model": j.model_name,
            "Benchmark": j.benchmark_name,
            "Samples": j.max_samples,
            "Status": j.status.value,
            "Started": j.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Error": (j.error_message or "")[:120],
        }
        for j in sorted(jobs, key=lambda x: x.created_at, reverse=True)
    ]
    jobs_df = pd.DataFrame(job_rows)

    def _style_status(col: pd.Series) -> list[str]:
        if col.name != "Status":
            return [""] * len(col)
        return [
            f"background-color: {STATUS_COLORS.get(v, '#6c757d')}; "
            "color: white; font-weight: 600; border-radius: 6px; text-align: center"
            for v in col
        ]

    st.dataframe(
        jobs_df.style.apply(_style_status, axis=0),
        width="stretch",
        hide_index=True,
    )
