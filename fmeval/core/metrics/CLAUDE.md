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
  __init__.py       ← exports: Metric, MCQMetrics, extract_letter
  base.py           ← ✅ Metric ABC
  mcq_metrics.py    ← ✅ MCQMetrics (all MCQ evaluation metrics) + extract_letter()
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
```

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

## Metric assignments by modality

| Modality      | Metrics applied          |
|---------------|--------------------------|
| `multimodal`  | MCQMetrics               |
| `text`        | MCQMetrics               |
| `time_series` | *(reserved — not current focus)* |
