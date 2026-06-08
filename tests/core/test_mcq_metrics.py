"""Tests for fmeval.core.metrics.mcq_metrics."""

import warnings

import pytest

from fmeval.core.metrics.mcq_metrics import MCQMetrics, extract_letter

# sklearn emits UndefinedMetricWarning when predictions contain classes not
# seen in targets (e.g. mock model predicts A on a B-only subset).  These
# are expected in test scenarios; suppress to keep output readable.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning",
                                        "ignore::UserWarning")


# ── extract_letter ─────────────────────────────────────────────────────────────

class TestExtractLetter:
    def test_letter_with_paren(self):
        assert extract_letter("A) some text") == "A"

    def test_letter_lowercase_paren(self):
        assert extract_letter("b) text") == "B"

    def test_letter_in_middle_of_sentence(self):
        assert extract_letter("The answer is B) Decrease") == "B"

    def test_standalone_letter_fallback(self):
        assert extract_letter("The answer is C") == "C"

    def test_none_when_no_letter(self):
        assert extract_letter("I don't know") is None

    def test_none_for_empty_string(self):
        assert extract_letter("") is None

    def test_letter_not_extracted_from_inside_word(self):
        # "BALANCE" should not yield "A" or "B" as a standalone match
        # The "A)" pattern won't match; standalone letters inside words won't either
        result = extract_letter("BALANCE of power")
        # Only "B" could match at word boundary in "BALANCE of power" if it
        # is at word start. Let's verify the actual behaviour:
        # "BALANCE" uppercased is "BALANCE"; \b matches before B (start of word).
        # After B comes A which is \w, so \bB\b does NOT match. Result: None.
        assert result is None

    def test_target_format_c_with_text(self):
        assert extract_letter("C) Decrease") == "C"

    def test_target_format_d_with_long_text(self):
        assert extract_letter("D) No pattern at all was found") == "D"


# ── MCQMetrics.compute ─────────────────────────────────────────────────────────

class TestMCQMetrics:
    def setup_method(self):
        self.metric = MCQMetrics()

    def test_name(self):
        assert self.metric.name == "mcq_metrics"

    def test_applicable_modalities(self):
        assert "multimodal" in self.metric.applicable_modalities
        assert "text" in self.metric.applicable_modalities

    def test_perfect_accuracy(self):
        preds = ["A) opt", "B) opt", "C) opt"]
        targets = ["A) x", "B) y", "C) z"]
        m = self.metric.compute(preds, targets)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_unparseable"] == 0.0

    def test_zero_accuracy(self):
        preds = ["B)", "C)", "A)"]
        targets = ["A) x", "B) y", "C) z"]
        m = self.metric.compute(preds, targets)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_partial_accuracy(self):
        preds = ["A)", "B)", "D)"]
        targets = ["A) x", "B) y", "C) z"]
        m = self.metric.compute(preds, targets)
        assert m["accuracy"] == pytest.approx(2 / 3)

    def test_unparseable_counted(self):
        # "unknown" has no standalone A-D letter — truly unparseable
        preds = ["A)", "unknown", "B)"]
        targets = ["A) x", "A) y", "B) z"]
        m = self.metric.compute(preds, targets)
        assert m["n_unparseable"] == 1.0

    def test_all_required_keys_present(self):
        preds = ["A)"] * 10
        targets = ["A) opt"] * 5 + ["B) opt"] * 5
        m = self.metric.compute(preds, targets)
        for key in (
            "accuracy", "balanced_accuracy",
            "f1_macro", "f1_weighted",
            "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted",
            "n_samples", "n_unparseable",
        ):
            assert key in m, f"Missing key: {key}"

    def test_per_class_keys_present(self):
        preds = ["A)", "B)"]
        targets = ["A) x", "B) y"]
        m = self.metric.compute(preds, targets)
        for lbl in ("A", "B"):
            assert f"f1_{lbl}" in m
            assert f"precision_{lbl}" in m
            assert f"recall_{lbl}" in m
            assert f"support_{lbl}" in m

    def test_support_counts_are_correct(self):
        preds = ["A)", "A)", "B)", "B)", "B)"]
        targets = ["A) x", "A) y", "B) z", "B) w", "B) v"]
        m = self.metric.compute(preds, targets)
        assert m["support_A"] == 2.0
        assert m["support_B"] == 3.0

    def test_n_samples(self):
        preds = ["A)"] * 7
        targets = ["A) x"] * 7
        m = self.metric.compute(preds, targets)
        assert m["n_samples"] == 7.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            self.metric.compute(["A)"], ["A) x", "B) y"])

    def test_balanced_accuracy_for_imbalanced_classes(self):
        # 9 class-A, 1 class-B; always predict A → balanced_accuracy < accuracy
        preds = ["A)"] * 10
        targets = ["A) x"] * 9 + ["B) y"]
        m = self.metric.compute(preds, targets)
        assert m["accuracy"] == pytest.approx(0.9)
        # balanced_accuracy = (recall_A + recall_B) / 2 = (1.0 + 0.0) / 2 = 0.5
        assert m["balanced_accuracy"] == pytest.approx(0.5)

    def test_mock_always_a_metrics(self):
        """Mirrors what the mock model will produce on a balanced 2-class dataset."""
        preds = ["A)"] * 6
        targets = ["A) x"] * 3 + ["B) y"] * 3
        m = self.metric.compute(preds, targets)
        assert m["accuracy"] == pytest.approx(0.5)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["recall_A"] == pytest.approx(1.0)
        assert m["recall_B"] == pytest.approx(0.0)
