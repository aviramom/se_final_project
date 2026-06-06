# fmeval/core/metrics/ — Metric abstractions

`Metric` is the ABC for computing evaluation scores from raw predictions and targets.
Each subclass is selected at runtime by modality — there is no hard-coded mapping
between a benchmark name and a metric.

---

## Files

```
metrics/
  CLAUDE.md
  base.py          ← Metric ABC
  mse.py           ← MSE  (time-series)
  mae.py           ← MAE  (time-series)
  exact_match.py   ← ExactMatch (text)
  f1.py            ← F1   (text)
```

---

## Metric ABC (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Literal

class Metric(ABC):
    """
    Computes a scalar score from predictions and targets.
    Must not know which model produced the predictions or which benchmark
    supplied the targets.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier stored in EvaluationResult, e.g. 'mse'."""
        ...

    @property
    @abstractmethod
    def applicable_modalities(self) -> list[Literal["text", "time_series"]]: ...

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> float:
        """
        predictions and targets arrive pre-parsed by ResultParser.
        Text:        list[str], list[str]
        Time-series: np.ndarray [batch, horizon], np.ndarray [batch, horizon]
        Returns a single float score.
        """
        ...
```

---

## Concrete subclass pattern

Each metric file must:

1. Subclass `Metric`.
2. Implement `name`, `applicable_modalities`, and `compute`.
3. Keep `compute` stateless — it receives everything it needs as arguments.

`EvaluationService` selects metrics by filtering on `applicable_modalities ==
dataset.modality`. No metric should know the name of the model or benchmark.

---

## Metric assignments by modality

| Modality      | Metrics applied          |
|---------------|--------------------------|
| `time_series` | MSE, MAE                 |
| `text`        | ExactMatch, F1           |
