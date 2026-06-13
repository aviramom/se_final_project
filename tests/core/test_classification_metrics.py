"""Tests for ClassificationMetrics (free-label classification scoring)."""

from __future__ import annotations

from fmeval.core.metrics.classification_metrics import (
    ClassificationMetrics,
    extract_label,
)


class TestExtractLabel:
    def test_exact_match(self):
        assert extract_label("2", ["1", "2", "3"]) == "2"

    def test_class_is_phrasing(self):
        assert extract_label("The class is 3", ["1", "2", "3"]) == "3"

    def test_predicted_label_phrasing(self):
        assert extract_label("Predicted Label: 1", ["1", "2"]) == "1"

    def test_label_colon_phrasing(self):
        assert extract_label("label: 2 (confident)", ["1", "2"]) == "2"

    def test_longest_first_avoids_substring_shadowing(self):
        # "1" must not match inside "10" — longer labels are tried first and the
        # (?!\d) guard prevents a numeric prefix match.
        assert extract_label("10", ["1", "10"]) == "10"

    def test_no_match_returns_none(self):
        assert extract_label("I am not sure", ["1", "2"]) is None


class TestCompute:
    def test_perfect_predictions(self):
        m = ClassificationMetrics()
        res = m.compute(["1", "2", "1", "2"], ["1", "2", "1", "2"])
        assert res["accuracy"] == 1.0
        assert res["balanced_accuracy"] == 1.0
        assert res["n_unparseable"] == 0.0
        assert res["num_of_classes"] == 2.0

    def test_unparseable_counts_wrong(self):
        m = ClassificationMetrics()
        res = m.compute(["1", "garbage"], ["1", "2"])
        assert res["n_unparseable"] == 1.0
        assert res["accuracy"] == 0.5
        # Per-class keys exist for every true label.
        assert "support_1" in res and "support_2" in res

    def test_balanced_accuracy_handles_imbalance(self):
        m = ClassificationMetrics()
        # Class "1" dominates; predicting all "1" gives high raw accuracy but
        # balanced accuracy near 0.5.
        preds = ["1", "1", "1", "1"]
        targets = ["1", "1", "1", "2"]
        res = m.compute(preds, targets)
        assert res["accuracy"] == 0.75
        assert res["balanced_accuracy"] == 0.5

    def test_label_predictions_returns_aligned_tokens(self):
        m = ClassificationMetrics()
        preds, trues = m.label_predictions(["2", "nope"], ["2", "1"])
        assert preds == ["2", None]
        assert trues == ["2", "1"]
