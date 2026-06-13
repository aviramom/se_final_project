# fmeval/core/metrics/ — Metric abstractions

`Metric` is the ABC for computing evaluation scores from raw predictions and targets.
Each subclass is selected at runtime by modality — there is no hard-coded mapping
between a benchmark name and a metric.

**Status: implemented.** `base.py` and `mcq_metrics.py` are written and tested
(`tests/core/test_mcq_metrics.py`).

---

## Files

```
metrics/
  CLAUDE.md
  __init__.py                 ← exports: Metric, MCQMetrics, extract_letter, ClassificationMetrics, extract_label
  base.py                     ← ✅ Metric ABC (compute + label_predictions)
  mcq_metrics.py              ← ✅ MCQMetrics (A–D MCQ metrics) + extract_letter()
  classification_metrics.py   ← ✅ ClassificationMetrics (free class labels) + extract_label()
```

---

## Metric ABC (`base.py`)

```python
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def applicable_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]: ...

    @abstractmethod
    def compute(self, predictions: list[str], targets: list[str]) -> dict[str, float]:
        """Returns a flat dict of metric_name → score (not a single float).
        A single subclass can report multiple related scores in one pass."""
        ...

    @abstractmethod
    def label_predictions(
        self, predictions: list[str], targets: list[str]
    ) -> tuple[list[str | None], list[str | None]]:
        """Per-sample (predicted_label, true_label) tokens — the pipeline's
        counterpart to compute(). Lets the pipeline record is_correct without
        knowing how this metric parses an answer (MCQ letter, class label, …).
        None = unparseable."""
        ...
```

**Which metric runs is decided by the dataset, not the modality.** Each `Dataset`
exposes a `metric` property (default `MCQMetrics` on the ABC; `UCRICLDataset`
returns `ClassificationMetrics`). `MockRunner` and `cluster_worker.py` build the
pipeline with `dataset.metric`; the pipeline calls `metric.label_predictions` for
per-sample scoring. `applicable_modalities` is still declared for documentation
and future modality-based filtering.

---

## MCQMetrics (`mcq_metrics.py`) — **implemented**

Handles all multiple-choice question evaluation. One call to `compute()` returns:

| Key | Description |
|---|---|
| `accuracy` | fraction of correctly predicted letters |
| `balanced_accuracy` | average per-class recall (handles class imbalance) |
| `f1_macro` / `f1_weighted` | unweighted / support-weighted mean F1 |
| `precision_macro` / `precision_weighted` | mean precision |
| `recall_macro` / `recall_weighted` | mean recall |
| `n_samples` | total predictions |
| `n_unparseable` | predictions where no A–D letter was extracted |
| `f1_A`, `precision_A`, `recall_A`, `support_A` … | per-class breakdown for each letter present in targets |

**`extract_letter(text: str) -> str | None`** — utility exported from this module.
Tries `([A-D])\)` first, then `\b([A-D])\b` as fallback. Returns `None` if no
letter is found.

---

## Concrete subclass pattern

Each metric file must:

1. Subclass `Metric`.
2. Implement `name`, `applicable_modalities`, and `compute`.
3. Keep `compute` stateless — receives everything as arguments; returns `dict[str, float]`.

`EvaluationService` selects metrics by filtering on
`applicable_modalities ⊇ {dataset.modality}`. No metric should know the name of
the model or benchmark.

---

## ClassificationMetrics (`classification_metrics.py`) — **implemented**

Free-label classification (the UCR ICL benchmark). Same aggregate-key shape as
`MCQMetrics` (so the dashboard, grouping, and CSV export need no special-casing)
but over an arbitrary class-label set inferred from the targets — not A–D letters.
Adds `num_of_classes`. `balanced_accuracy` is the primary metric (UCR is often
class-imbalanced).

**`extract_label(response, label_set) -> str | None`** — recovers the predicted
label by matching `response` against the known labels using the reference UCR
evaluator's priority rules (exact, `The class is X`, `Predicted Label: X`,
`Predicted: X`, `label: X`, `label is X`). Candidates are tried **longest-first**
with a `(?!\d)` guard so a short label can't shadow a longer one (`"1"` vs `"10"`).
Unmatched predictions become an `INVALID_PREDICTION` sentinel (counted wrong,
never a phantom class).

## Metric assignments

The dataset picks its metric (`dataset.metric`); modality is no longer the
selector. Current assignments:

| Dataset family        | Metric                |
|-----------------------|-----------------------|
| `tsexam1` (MCQ)       | MCQMetrics            |
| `icl_ucr_*` (UCR ICL) | ClassificationMetrics |
