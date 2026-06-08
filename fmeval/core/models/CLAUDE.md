# fmeval/core/models/ — ModelWrapper abstractions

`ModelWrapper` is the ABC that hides every HuggingFace model's unique loading and
inference API behind a single interface. Adding a new model = one new file here + one
registry entry. Nothing else changes.

**All models are HuggingFace models.** No external REST APIs, no API keys.

**All models produce text output** — the research focus is multimodal TS+text tasks
where both LLMs and specialized multimodal models generate a natural-language answer.

---

**Status: partially implemented.** `base.py` and `mock_model.py` are written and
tested (`tests/core/test_mock_model.py`). `registry.py` and real model wrappers are
not yet implemented.

---

## Files

```
models/
  CLAUDE.md
  __init__.py      ← ✅ exports: ModelWrapper, MockModel
  base.py          ← ✅ ModelWrapper ABC
  mock_model.py    ← ✅ MockModel (always answers a fixed letter — for pipeline testing)
  registry.py      ← MODEL_REGISTRY dict + get_model() factory  (not yet written)
  <model>.py       ← one file per model family  (not yet written)
```

---

## ModelWrapper ABC (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Literal
from fmeval.core.sample import Sample

class ModelWrapper(ABC):
    """
    Wraps one HuggingFace model family. Handles model-specific loading,
    input formatting, and inference. Never knows about metrics or benchmarks.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier, e.g. 'llama3-8b'."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        """All current models declare ["multimodal"]."""
        ...

    @property
    @abstractmethod
    def input_mode(self) -> Literal["combined", "separate"]:
        """
        How this model consumes a Sample's time series data.

        "combined"  — SampleFormatter.to_combined() is called before format_input;
                      the model receives a single text string with TS values inlined.
                      Use for standard LLMs (LLaMA, Mistral, etc.).

        "separate"  — Sample is passed as-is; the model receives both input_text
                      (with <TS_N> tokens) and input_ts (raw arrays) and handles
                      the fusion itself.
                      Use for models with a dedicated TS encoder.
        """
        ...

    @abstractmethod
    def format_input(self, sample: Sample) -> Any:
        """
        Convert a Sample into the model's expected input format.
        Must call SampleFormatter based on self.input_mode before further formatting.
        Returns whatever structure predict() expects (string, tensor, dict, etc.).
        """
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> list[str]:
        """
        Run inference on a batch of formatted inputs (output of format_input).
        Always returns list[str] — all models output text.
        This is the code that runs on the cluster; in mock mode it runs locally.
        """
        ...
```

---

## Registry (`registry.py`)

```python
MODEL_REGISTRY: dict[str, type[ModelWrapper]] = {}

def get_model(name: str, **kwargs) -> ModelWrapper:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
```

The registry is the **only** place that maps string names to classes. `EvaluationService`
calls `get_model(name)` — it never imports concrete wrapper classes directly.

---

## Concrete wrapper pattern

Each model file must:

1. Subclass `ModelWrapper`.
2. Accept `model_id: str` in `__init__` (the HuggingFace repo id). This lets the
   caller choose a variant without touching code.
3. Declare `input_mode` — `"combined"` for LLMs, `"separate"` for models with a TS
   encoder.
4. In `format_input`: call `SampleFormatter.to_combined(sample)` or
   `SampleFormatter.to_separate(sample)` based on `self.input_mode`, then apply any
   model-specific prompt template.
5. Implement `predict` to return `list[str]`.
6. Register itself: `MODEL_REGISTRY["my_model"] = MyModelWrapper` at module level.

---

## MockModel (`mock_model.py`) — **implemented**

```python
model = MockModel(answer="A")   # always returns "A)" for every input
model = MockModel(answer="B")   # always returns "B)"
```

`input_mode = "combined"` — calls `SampleFormatter.to_combined()` in `format_input`.
Used to verify the full pipeline end-to-end without loading any real weights.

---

## Public exports (`__init__.py`)

```python
from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.mock_model import MockModel
```

When `registry.py` is implemented, add:
```python
from fmeval.core.models.registry import get_model, MODEL_REGISTRY
```
