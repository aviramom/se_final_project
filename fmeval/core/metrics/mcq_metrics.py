"""MCQ evaluation metrics: accuracy, balanced accuracy, F1, precision, recall."""

from __future__ import annotations

import re
from typing import Literal

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from fmeval.core.metrics.base import Metric

# Matches "A)" / "B)" / "C)" / "D)" (case-insensitive via .upper() caller)
_ANSWER_PAREN_RE = re.compile(r"([A-D])\)")
# Fallback: standalone letter with no adjacent word chars
_ANSWER_WORD_RE = re.compile(r"(?<!\w)([A-D])(?!\w)")


def extract_letter(text: str) -> str | None:
    """Extract the first A–D answer letter from a model output or target string.

    Tries "A)" style first (most specific), then a standalone-letter fallback.
    Returns None if no letter is found — the pipeline treats these as wrong.
    """
    upper = text.upper()
    m = _ANSWER_PAREN_RE.search(upper)
    if m:
        return m.group(1)
    m = _ANSWER_WORD_RE.search(upper)
    return m.group(1) if m else None


class MCQMetrics(Metric):
    """All standard metrics for multiple-choice question evaluation.

    Computed in one pass so callers get everything from a single compute() call.

    Aggregate metrics returned:
        accuracy             — fraction of exactly correct letters
        balanced_accuracy    — average per-class recall (handles class imbalance)
        f1_macro             — unweighted mean F1 across answer-letter classes
        f1_weighted          — support-weighted mean F1
        precision_macro      — unweighted mean precision
        precision_weighted   — support-weighted mean precision
        recall_macro         — unweighted mean recall
        recall_weighted      — support-weighted mean recall
        n_samples            — total predictions
        n_unparseable        — predictions where no A–D letter was extracted

    Per-class metrics (keyed by letter present in the targets):
        f1_<L>               — F1 for class L
        precision_<L>        — precision for class L
        recall_<L>           — recall for class L
        support_<L>          — number of true examples in class L
    """

    @property
    def name(self) -> str:
        return "mcq_metrics"

    @property
    def applicable_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["text", "multimodal"]

    def compute(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> dict[str, float]:
        if len(predictions) != len(targets):
            raise ValueError(
                f"MCQMetrics.compute: predictions ({len(predictions)}) and "
                f"targets ({len(targets)}) must have the same length."
            )

        true_letters = [extract_letter(t) for t in targets]
        pred_letters_raw = [extract_letter(p) for p in predictions]

        n_unparseable = sum(1 for p in pred_letters_raw if p is None)

        # Replace None with a sentinel that is never a valid class so it counts
        # as wrong but does not create a spurious new class in true_letters.
        pred_letters = [p if p is not None else "?" for p in pred_letters_raw]

        # Classes derived from targets only — avoids polluting labels with "?"
        labels = sorted(set(true_letters))

        results: dict[str, float] = {
            "accuracy": float(accuracy_score(true_letters, pred_letters)),
            "balanced_accuracy": float(
                balanced_accuracy_score(true_letters, pred_letters, adjusted=False)
            ),
            "f1_macro": float(
                f1_score(true_letters, pred_letters, average="macro",
                         labels=labels, zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(true_letters, pred_letters, average="weighted",
                         labels=labels, zero_division=0)
            ),
            "precision_macro": float(
                precision_score(true_letters, pred_letters, average="macro",
                                labels=labels, zero_division=0)
            ),
            "precision_weighted": float(
                precision_score(true_letters, pred_letters, average="weighted",
                                labels=labels, zero_division=0)
            ),
            "recall_macro": float(
                recall_score(true_letters, pred_letters, average="macro",
                             labels=labels, zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(true_letters, pred_letters, average="weighted",
                             labels=labels, zero_division=0)
            ),
            "n_samples": float(len(predictions)),
            "n_unparseable": float(n_unparseable),
        }

        # Per-class breakdown
        f1_pc = f1_score(true_letters, pred_letters, average=None,
                         labels=labels, zero_division=0)
        prec_pc = precision_score(true_letters, pred_letters, average=None,
                                  labels=labels, zero_division=0)
        rec_pc = recall_score(true_letters, pred_letters, average=None,
                              labels=labels, zero_division=0)

        for i, lbl in enumerate(labels):
            results[f"f1_{lbl}"] = float(f1_pc[i])
            results[f"precision_{lbl}"] = float(prec_pc[i])
            results[f"recall_{lbl}"] = float(rec_pc[i])
            results[f"support_{lbl}"] = float(true_letters.count(lbl))

        return results
