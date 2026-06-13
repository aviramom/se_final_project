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

| name | display_name | requires_gpu | weight env var | factory |
|---|---|---|---|---|
| `mock_always_a` | Mock Model (always A) | No | — | `MockModel("A")` |
| `mock_always_b` | Mock Model (always B) | No | — | `MockModel("B")` |
| `mock_always_c` | Mock Model (always C) | No | — | `MockModel("C")` |
| `random_label` | Random Label (chance baseline) | No | — | `RandomLabelModel()` |
| `chatts-8b` | ChatTS-8B (ByteDance Research) | Yes | `CHATTS_MODEL_PATH` | `ChatTSModel(checkpoint_path=…)` |
| `qwen3-vl-8b` | Qwen3-VL-8B-Instruct (vision) | Yes | `QWEN_VL_MODEL_PATH` | `QwenVLModel(checkpoint_path=…)` |

GPU model weight paths are read from env vars at factory time and forwarded into the
sbatch script by `_build_runner()` in `app/main.py`.

To add a new model: one new file in `fmeval/core/models/` + one `registry.register(…)` call here.

---

## BenchmarkRegistry (`benchmark_registry.py`)

```python
@dataclass
class BenchmarkInfo:
    name: str
    display_name: str
    modality: str
    group: str = ""              # picker group (UCR datasets share one per category)
    short_name: str = ""         # label within a group (e.g. the dataset name)
    supports_few_shot: bool = False  # True → UI shows k / strategy / seed controls
    # __post_init__ defaults group/short_name to display_name when left blank.

class BenchmarkRegistry:
    def register(self, info: BenchmarkInfo, factory: DatasetFactory) -> None: ...
    def list(self) -> list[BenchmarkInfo]: ...
    def get(self, name, max_samples=None, dataset_params=None) -> Dataset: ...

# DatasetFactory = Callable[[int | None, dict], Dataset]
```

The factory receives `max_samples` plus a `dataset_params` dict (benchmark-specific
construction hints from `EvaluationConfig`, e.g. `num_shots`/`picking_strategy`/
`random_seed` for ICL). Factories ignore keys they don't use — the `tsexam1`
factory is `lambda max_s, _params: …`; `_ucr_factory` reads the few-shot keys.

### Currently registered benchmarks

| name | display_name | factory |
|---|---|---|
| `tsexam1` | TimeSeriesExam1 (HuggingFace) | `TimeSeriesExam1Dataset(max_samples=…)` |
| `icl_ucr_<Name>` (95) | UCR ICL: <Name> (<category>) | `UCRICLDataset(<Name>, data_path=_ucr_root(), max_samples=…)` |

All 95 feasible UCR datasets are registered programmatically from `_UCR_DATASETS`
(one entry per dataset, since `EvaluationConfig` selects by name only). `_ucr_root()`
reads the `UCR_DATA_PATH` env var (default: the cluster `Univariate_arff` path),
resolved lazily at factory-call time so a missing path locally doesn't break the
registry or the modality compat-check. `_ucr_factory(name)` binds each dataset name
in a closure to avoid loop late-binding. A friendlier UI picker for the 95 entries
is a planned follow-up.

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
