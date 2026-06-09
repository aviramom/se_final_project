# fmeval/config/ — Layer 5 (part 2): Registries

Loaded once at startup. Answers two questions: "what is available?" (for UI dropdowns)
and "give me the object for name X" (for `EvaluationService`).

**Dependency rule:** `config/` may import from `core/` to instantiate `ModelWrapper`
and `Dataset` subclasses. It must not import from `app/`, `services/`, `execution/`,
or `storage/`.

---

## Files

```
config/
  __init__.py             ✅ exports all four public names
  CLAUDE.md
  model_registry.py       ✅ ModelInfo, ModelRegistry, build_default_model_registry()
  benchmark_registry.py   ✅ BenchmarkInfo, BenchmarkRegistry, build_default_benchmark_registry()
```

`settings.py` is not needed for the POC — runner mode and db path are wired in
`fmeval/app/main.py` directly. Add it when the Slurm runner requires env-level config.

---

## ModelRegistry (`model_registry.py`)

```python
@dataclass
class ModelInfo:
    name: str           # registry key used in EvaluationConfig
    display_name: str   # shown in UI dropdowns
    modalities: list[str]

class ModelRegistry:
    def register(self, info: ModelInfo, factory: Callable[[], ModelWrapper]) -> None: ...
    def list(self) -> list[ModelInfo]: ...
    def get(self, name: str) -> ModelWrapper: ...  # calls factory() — fresh instance each time
```

**Factory pattern:** `get()` calls `factory()` on every invocation so each submitted
job gets a fresh `ModelWrapper` with no shared state.

### Currently registered models

| name | display_name | factory |
|---|---|---|
| `mock_always_a` | Mock Model (always A) | `MockModel("A")` |
| `mock_always_b` | Mock Model (always B) | `MockModel("B")` |
| `mock_always_c` | Mock Model (always C) | `MockModel("C")` |

To add a real LLM: `registry.register(ModelInfo(...), lambda: MyLLMWrapper(...))`.

---

## BenchmarkRegistry (`benchmark_registry.py`)

```python
@dataclass
class BenchmarkInfo:
    name: str
    display_name: str
    modality: str

class BenchmarkRegistry:
    def register(self, info: BenchmarkInfo, factory: Callable[[int | None], Dataset]) -> None: ...
    def list(self) -> list[BenchmarkInfo]: ...
    def get(self, name: str, max_samples: int | None = None) -> Dataset: ...
```

The factory receives `max_samples` from `EvaluationConfig` so the dataset is
constructed with the correct size limit without storing config state on the registry.

### Currently registered benchmarks

| name | display_name | factory |
|---|---|---|
| `tsexam1` | TimeSeriesExam1 (HuggingFace) | `TimeSeriesExam1Dataset(max_samples=…)` |

---

## Startup wiring (in `app/main.py`)

```python
@st.cache_resource
def get_service():
    return EvaluationService(
        model_registry=build_default_model_registry(),
        benchmark_registry=build_default_benchmark_registry(),
        runner=MockRunner(),
        repository=ResultsRepository(db_path),
    )
```
