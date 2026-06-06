# fmeval/config/ — Layer 5 (part 2): Registries

Loaded once at startup. Answers two questions for the system: "what models and
benchmarks are available?" (for the UI dropdowns) and "give me the object for name X"
(for `EvaluationService`).

**Dependency rule:** `config/` may import from `core/` to instantiate `ModelWrapper`
and `Dataset` subclasses. It must not import from `app/`, `services/`, `execution/`,
or `storage/`.

---

## Files

```
config/
  CLAUDE.md
  model_registry.py       ← ModelRegistry
  benchmark_registry.py   ← BenchmarkRegistry
  settings.py             ← app-level config (runner mode, db path, data dirs)
```

---

## ModelRegistry (`model_registry.py`)

```python
@dataclass
class ModelInfo:
    name: str                        # registry key, e.g. "chronos"
    display_name: str                # shown in UI dropdown
    modalities: list[str]
    default_model_id: str            # HuggingFace repo id

class ModelRegistry:
    def list(self) -> list[ModelInfo]: ...
    def get(self, name: str, **kwargs) -> ModelWrapper: ...
```

`get` delegates to `get_model()` from `fmeval.core.models.registry`. The registry
here is the configuration layer on top of the core factory.

---

## BenchmarkRegistry (`benchmark_registry.py`)

```python
@dataclass
class BenchmarkInfo:
    name: str
    display_name: str
    modality: str
    data_path: Path

class BenchmarkRegistry:
    def list(self) -> list[BenchmarkInfo]: ...
    def get(self, name: str) -> Dataset: ...
```

`get` instantiates the correct `Dataset` subclass with the configured data path.

---

## Settings (`settings.py`)

Centralizes the few knobs the app needs at runtime:

- `RUNNER_MODE`: `"mock"` | `"slurm"` | `"precomputed"` (read from env or config file)
- `DB_PATH`: path to the SQLite database file
- `DATA_DIR`: root for `data/dummy/` and `data/precomputed/`
- `SLURM_*`: cluster-specific vars (partition, account, etc.) — only relevant when
  `RUNNER_MODE = "slurm"`

The active `Runner` implementation is selected here based on `RUNNER_MODE` and
injected into `EvaluationService`.
