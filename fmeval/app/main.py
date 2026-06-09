"""Streamlit entry point for the Foundation Model Evaluation Platform.

Run with:
    streamlit run fmeval/app/main.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fmeval.config import build_default_benchmark_registry, build_default_model_registry
from fmeval.execution.mock_runner import MockRunner
from fmeval.services.evaluation_service import EvaluationService
from fmeval.storage.repository import ResultsRepository


@st.cache_resource
def get_service() -> EvaluationService:
    """Create a single EvaluationService shared across all Streamlit rerenders."""
    db_path = Path(__file__).parent.parent.parent / "data" / "results.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return EvaluationService(
        model_registry=build_default_model_registry(),
        benchmark_registry=build_default_benchmark_registry(),
        runner=MockRunner(),
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
