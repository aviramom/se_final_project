# fmeval/core/models/ — ModelWrapper abstractions

`ModelWrapper` is the ABC that hides every HuggingFace model's unique loading and
inference API behind a single interface. Adding a new model = one new file here + one
registry entry. Nothing else changes.

**All models are HuggingFace models.** No external REST APIs, no API keys.

---

## Files

```
models/
  CLAUDE.md
  base.py          ← ModelWrapper ABC
  registry.py      ← MODEL_REGISTRY dict + get_model() factory
  __init__.py      ← exports: ModelWrapper, get_model
  chronos.py       ← Amazon Chronos (time-series)
  timesfm.py       ← Google TimesFM (time-series)
  llama3.py        ← Meta LLaMA 3 (text)
  ...              ← one file per model family
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
    input formatting, and inference. Never knows about metrics or benchmark format.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier, e.g. 'chronos-t5-small'."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[Literal["text", "time_series"]]: ...

    @abstractmethod
    def format_input(self, sample: Sample) -> Any:
        """
        Convert a standardized Sample into the model's expected input format.
        Text: returns a prompt string.
        Time-series: returns a tensor or array of the context window.
        """
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Run inference on a batch of formatted inputs (output of format_input).
        Text:        returns list[str]
        Time-series: returns np.ndarray shape [batch, horizon]
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
2. Accept `model_id: str` in `__init__` (the HuggingFace repo id, e.g.
   `"amazon/chronos-t5-small"`). This lets the caller choose a variant without
   touching code.
3. Load the HuggingFace model in `__init__` (via `transformers.pipeline`,
   `AutoModel`, or the model's own HF client class such as `ChronosPipeline`).
4. Implement `format_input`, `predict`, `model_name`, and `supported_modalities`.
5. Register itself: `MODEL_REGISTRY["chronos"] = ChronosWrapper` at module level.

---

## Public exports (`__init__.py`)

```python
from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.registry import get_model, MODEL_REGISTRY
```

Callers use:
```python
from fmeval.core.models import get_model
model = get_model("chronos", model_id="amazon/chronos-t5-small", device="cuda")
```
