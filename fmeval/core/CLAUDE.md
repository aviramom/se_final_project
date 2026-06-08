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

## Sample (`sample.py`) — **implemented**

The standardized unit that flows between `Dataset` and `ModelWrapper`.

```python
@dataclass
class Sample:
    input_text: str               # NL prompt; <TS_N> tokens mark TS positions
    input_ts:   list[np.ndarray]  # raw TS arrays, index-matched to <TS_N> tokens
    output:     str               # ground-truth text answer / label
    metadata:   dict              # benchmark-specific extras (id, split, etc.)
```

Every `Dataset` yields `Sample` objects in **canonical (separate) form** — placeholders
in the text, raw arrays alongside. Every `ModelWrapper` receives them via
`format_input(sample)`, which calls `SampleFormatter` to convert to the form the model
expects. No other representation crosses the Dataset ↔ ModelWrapper boundary.

### combined vs. separate

`SampleFormatter` (in `core/datasets/formatters.py`) converts between the two views:

- **separate** (canonical) — `<TS_N>` tokens in `input_text`, raw arrays in `input_ts`.
  Passed as-is to models with a dedicated TS encoder.
- **combined** — tokens replaced by serialized float lists; `input_ts` is cleared.
  Used by LLMs that consume a single text string.

`ModelWrapper` declares `input_mode: Literal["combined", "separate"]` and calls the
formatter inside `format_input`. The dataset never knows which mode will be used.

---

## Modality as the routing key

All current datasets are `"multimodal"` (text + time series in, text out). The
`modality` property is kept broad (`Literal["text", "time_series", "multimodal"]`) so
pure-text or pure-TS datasets can slot in later without interface changes.

`modality` drives two downstream decisions:

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
