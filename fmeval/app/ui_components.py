"""Reusable presentation components: page CSS, status badges, Q/A cards.

Pure rendering — no business logic and no service calls. Lives in the app
layer so the chat-bubble sample viewer and badges can be shared between the
run-detail and compare views without duplicating markup.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from fmeval.evaluation.result import SamplePrediction
    from fmeval.services.types import SampleDiff

# Status palette shared by badges and the jobs-table styler.
STATUS_COLORS = {
    "queued": "#6c757d",
    "running": "#0d6efd",
    "completed": "#198754",
    "failed": "#dc3545",
}

_CSS = """
<style>
.fm-badge {
    display: inline-block;
    padding: 2px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.02em;
}
.fm-card {
    border: 1px solid #e3e6ea;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
}
.fm-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #475467;
}
.fm-sample-id { font-weight: 700; color: #344054; }
.fm-question {
    background: #f2f6fc;
    border: 1px solid #dbe7fb;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
    white-space: pre-wrap;
    font-size: 0.88rem;
    color: #1d2939;
}
.fm-answer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 6px;
}
.fm-answer {
    border-radius: 10px;
    padding: 10px 12px;
    white-space: pre-wrap;
    font-size: 0.85rem;
    border: 1px solid #e3e6ea;
    background: #fafbfc;
    max-height: 280px;
    overflow-y: auto;
}
.fm-answer-correct { border-left: 4px solid #198754; }
.fm-answer-wrong   { border-left: 4px solid #dc3545; }
.fm-truth          { border-left: 4px solid #6c757d; background: #f8f9fa; }
.fm-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #667085;
    margin-bottom: 4px;
}
.fm-chip {
    display: inline-block;
    background: #eef2f6;
    color: #475467;
    border-radius: 999px;
    padding: 1px 9px;
    font-size: 0.72rem;
    margin: 6px 6px 0 0;
}
</style>
"""


def inject_css() -> None:
    """Inject the shared stylesheet once per page render (call from main())."""
    st.markdown(_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return a pill-shaped colored badge for a job status, as inline HTML."""
    color = STATUS_COLORS.get(status, "#6c757d")
    return (
        f'<span class="fm-badge" style="background:{color}">'
        f"{html.escape(status)}</span>"
    )


def _esc(text: str) -> str:
    """Escape user/model text for HTML embedding (placeholders like <TS_0> included)."""
    return html.escape(text or "")


def _chips(metadata: dict) -> str:
    return "".join(
        f'<span class="fm-chip">{_esc(str(k))}: {_esc(str(v))}</span>'
        for k, v in metadata.items()
    )


def _answer_block(label: str, text: str, correct: bool | None) -> str:
    """One answer bubble. correct=None renders the neutral ground-truth style."""
    if correct is None:
        cls = "fm-truth"
    else:
        cls = "fm-answer-correct" if correct else "fm-answer-wrong"
    return (
        f'<div><div class="fm-label">{_esc(label)}</div>'
        f'<div class="fm-answer {cls}">{_esc(text)}</div></div>'
    )


def render_qa_card(pred: "SamplePrediction") -> None:
    """Chat-bubble style card for one sample: question, model output, ground truth."""
    icon = "✓" if pred.is_correct else "✗"
    icon_color = "#198754" if pred.is_correct else "#dc3545"
    predicted = pred.predicted_letter or "—"
    card = (
        '<div class="fm-card">'
        '<div class="fm-card-header">'
        f'<span class="fm-sample-id">Sample #{pred.sample_idx} '
        f'<span style="color:{icon_color}">{icon}</span></span>'
        f"<span>predicted <b>{_esc(predicted)}</b> · correct <b>{_esc(pred.correct_letter)}</b></span>"
        "</div>"
        f'<div class="fm-question">{_esc(pred.input_text)}</div>'
        '<div class="fm-answer-grid">'
        + _answer_block("Model output", pred.raw_prediction, pred.is_correct)
        + _answer_block("Ground truth", pred.raw_target, None)
        + "</div>"
        f"<div>{_chips(pred.metadata)}</div>"
        "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)


def render_diff_card(diff: "SampleDiff", label_a: str, label_b: str) -> None:
    """Side-by-side card comparing two runs' answers on the same question."""
    mark_a = "✓" if diff.a_correct else "✗"
    mark_b = "✓" if diff.b_correct else "✗"
    card = (
        '<div class="fm-card">'
        '<div class="fm-card-header">'
        f'<span class="fm-sample-id">Sample #{diff.sample_idx}</span>'
        f"<span>correct answer: <b>{_esc(diff.correct_letter)}</b></span>"
        "</div>"
        f'<div class="fm-question">{_esc(diff.input_text)}</div>'
        '<div class="fm-answer-grid">'
        + _answer_block(
            f"{label_a} — {diff.predicted_a or '—'} {mark_a}",
            diff.raw_prediction_a,
            diff.a_correct,
        )
        + _answer_block(
            f"{label_b} — {diff.predicted_b or '—'} {mark_b}",
            diff.raw_prediction_b,
            diff.b_correct,
        )
        + "</div>"
        + _answer_block("Ground truth", diff.raw_target, None)
        + f"<div>{_chips(diff.metadata)}</div>"
        "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)
