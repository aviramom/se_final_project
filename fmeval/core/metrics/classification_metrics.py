"""Classification metrics for free-label tasks (e.g. UCR ICL).

Unlike MCQMetrics, which assumes the answer is one of a fixed A–D letter set,
this metric works with an arbitrary set of class labels derived from the targets
themselves.  It is the evaluation method for in-context-learning classification
benchmarks where the model is asked to reply with one class label drawn from the
support set (which may be integers, words, or any string).

The predicted label is recovered from the raw model output by matching against
the known label set using the same priority rules as the reference UCR ICL
evaluator: exact match first, then a handful of "The class is X" / "Predicted
Label: X" / "label: X" style phrasings.  Anything that matches no label is
counted as wrong (an unparseable prediction) without inventing a phantom class.
"""

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

# Sentinel for predictions that match no known label.  Never a real class, so it
# counts as wrong for every metric without polluting the set of true labels.
_INVALID = "INVALID_PREDICTION"


def extract_label(response: str, label_set: list[str]) -> str | None:
    """Return the first label in label_set that the response selects, or None.

    Candidate labels are tried longest-first so a short label cannot shadow a
    longer one that contains it (e.g. "1" must not match inside "10").  The
    `(?!\\d)` guard stops a numeric label matching a longer number's prefix.
    """
    # Longest-first, then lexical, for deterministic tie-breaking.
    for label in sorted(label_set, key=lambda x: (-len(x), x)):
        esc = re.escape(label)
        if response == label:
            return label
        if f"The class is {label}" in response or f"The class is <{label}>" in response:
            return label
        if re.search(
            rf"Predicted\s*Label\s*:\s*[\"'<\[]?\s*{esc}(?!\d)", response, re.IGNORECASE
        ):
            return label
        if re.search(
            rf"Predicted\s*:\s*[\"'<\[]?\s*{esc}(?!\d)", response, re.IGNORECASE
        ):
            return label
        if re.search(
            rf"(?<!\w)label\s*:\s*[\"'<\[]?\s*{esc}(?!\d)", response, re.IGNORECASE
        ):
            return label
        if re.search(
            rf"(?:correct\s+)?label\s+is\s+[\"'<\[]?\s*{esc}(?!\d)",
            response,
            re.IGNORECASE,
        ):
            return label
    return None


class ClassificationMetrics(Metric):
    """Balanced-accuracy-centred metrics over an arbitrary class-label set.

    The label set is inferred from the targets, so the metric needs no prior
    knowledge of the benchmark.  Mirrors the aggregate-key shape of MCQMetrics
    (accuracy, balanced_accuracy, f1/precision/recall macro+weighted, n_samples,
    n_unparseable, and per-class breakdowns) so the dashboard's metric selector,
    grouping, and CSV export work without special-casing.

    balanced_accuracy is the primary metric: it averages per-class recall, which
    matters because UCR datasets are frequently class-imbalanced.
    """

    @property
    def name(self) -> str:
        return "classification_metrics"

    @property
    def applicable_modalities(
        self,
    ) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["text", "multimodal"]

    def label_predictions(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> tuple[list[str | None], list[str | None]]:
        """Per-sample predicted/true labels.

        True labels are the targets verbatim (the dataset stores a clean label).
        Predicted labels are recovered from the raw output by matching against
        the label set derived from the targets; None if nothing matches.
        """
        label_set = sorted(set(targets))
        pred_labels = [extract_label(p, label_set) for p in predictions]
        return pred_labels, list(targets)

    def compute(
        self,
        predictions: list[str],
        targets: list[str],
    ) -> dict[str, float]:
        if len(predictions) != len(targets):
            raise ValueError(
                f"ClassificationMetrics.compute: predictions ({len(predictions)}) "
                f"and targets ({len(targets)}) must have the same length."
            )

        pred_labels_raw, true_labels = self.label_predictions(predictions, targets)
        n_unparseable = sum(1 for p in pred_labels_raw if p is None)
        # Unparseable → sentinel so it counts wrong but never becomes a true class.
        pred_labels = [p if p is not None else _INVALID for p in pred_labels_raw]

        # Classes come from the targets only — keeps _INVALID out of the labels.
        labels = sorted(set(targets))

        results: dict[str, float] = {
            "accuracy": float(accuracy_score(true_labels, pred_labels)),
            "balanced_accuracy": float(
                balanced_accuracy_score(true_labels, pred_labels, adjusted=False)
            ),
            "f1_macro": float(
                f1_score(
                    true_labels,
                    pred_labels,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    true_labels,
                    pred_labels,
                    average="weighted",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "precision_macro": float(
                precision_score(
                    true_labels,
                    pred_labels,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "precision_weighted": float(
                precision_score(
                    true_labels,
                    pred_labels,
                    average="weighted",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    true_labels,
                    pred_labels,
                    average="macro",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "recall_weighted": float(
                recall_score(
                    true_labels,
                    pred_labels,
                    average="weighted",
                    labels=labels,
                    zero_division=0,
                )
            ),
            "n_samples": float(len(predictions)),
            "n_unparseable": float(n_unparseable),
            "num_of_classes": float(len(labels)),
        }

        f1_pc = f1_score(
            true_labels, pred_labels, average=None, labels=labels, zero_division=0
        )
        prec_pc = precision_score(
            true_labels, pred_labels, average=None, labels=labels, zero_division=0
        )
        rec_pc = recall_score(
            true_labels, pred_labels, average=None, labels=labels, zero_division=0
        )

        for i, lbl in enumerate(labels):
            results[f"f1_{lbl}"] = float(f1_pc[i])
            results[f"precision_{lbl}"] = float(prec_pc[i])
            results[f"recall_{lbl}"] = float(rec_pc[i])
            results[f"support_{lbl}"] = float(targets.count(lbl))

        return results
