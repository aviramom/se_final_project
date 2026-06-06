# fmeval/core/ — Layer 3: Domain

The center of the architecture. Three abstract base classes and one shared dataclass.
Everything else in the system exists to serve or invoke these abstractions. They depend
on nothing — no UI, no database, no cluster.

**Dependency rule:** `core/` must not import from `app/`, `services/`, `execution/`,
`storage/`, or `config/`. Pure Python only.

---

## Sub-packages

```
core/
  CLAUDE.md
  datasets/   ← Dataset ABC + concrete benchmark subclasses
  models/     ← ModelWrapper ABC + concrete HuggingFace wrappers
  metrics/    ← Metric ABC + MSE, MAE, ExactMatch, F1
  sample.py   ← Sample dataclass (shared across sub-packages)
```

---

## Sample (`sample.py`)

The standardized unit that flows between `Dataset` and `ModelWrapper`.

```python
@dataclass
class Sample:
    input: Any          # list[str] for text; np.ndarray for time-series
    target: Any         # ground-truth label or future values
    modality: Literal["text", "time_series"]
    metadata: dict      # benchmark-specific extras (id, split, etc.)
```

Every `Dataset` yields `Sample` objects. Every `ModelWrapper` receives them via
`format_input(sample)`. No other representation crosses the Dataset ↔ ModelWrapper
boundary.

---

## Modality as the routing key

A `Dataset` declares its `modality`; that tag drives two downstream decisions:

1. **Compatibility check** — `EvaluationService` rejects a run if
   `dataset.modality not in model.supported_modalities` before anything executes.
2. **Metric selection** — the service picks `Metric` subclasses whose
   `applicable_modalities` includes `dataset.modality`. No `if benchmark == X`
   branching anywhere.

This is the mechanism that eliminates the repetitive adapter glue the project exists
to remove. Keep it clean.

---

## Invariant

If any file in `core/` imports from outside `core/`, the layering has leaked.
Flag it in review rather than papering over it.
