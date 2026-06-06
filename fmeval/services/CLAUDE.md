# fmeval/services/ — Layer 2: Orchestration

`EvaluationService` is the single class the UI talks to. It owns the evaluation
workflow end-to-end and delegates every piece of actual work to the layers below.
Nothing below this layer ever calls back up into it.

**Dependency rule:** `services/` may import from `core`, `execution`, `storage`,
and `config`. It must never import from `app/`.

---

## Files

```
services/
  __init__.py             ← package init (empty)
  CLAUDE.md
  types.py                ← EvaluationConfig, DashboardData
  evaluation_service.py   ← EvaluationService
```

---

## Types (`types.py`)

```python
@dataclass
class EvaluationConfig:
    """Input to EvaluationService.run_evaluation — selects model and benchmark."""
    model_name: str
    benchmark_name: str

@dataclass
class DashboardData:
    """Read-path payload shaped for UI rendering of job statuses and result comparisons."""
    jobs: list[EvaluationJob]      # TYPE_CHECKING import
    results: list[EvaluationResult]  # TYPE_CHECKING import
```

---

## EvaluationService (`evaluation_service.py`)

All cross-layer imports (`ModelRegistry`, `BenchmarkRegistry`, `Runner`,
`EvaluationJob`, `ResultsRepository`) are guarded under `TYPE_CHECKING` so the
module is importable before those layers have Python files.

### Public surface

```python
class EvaluationService:
    def __init__(
        self,
        model_registry: ModelRegistry,
        benchmark_registry: BenchmarkRegistry,
        runner: Runner,
        repository: ResultsRepository,
    ) -> None: ...

    def list_models(self) -> list[ModelInfo]: ...
    def list_benchmarks(self) -> list[BenchmarkInfo]: ...
    def run_evaluation(self, config: EvaluationConfig) -> str: ...  # returns job_id
    def poll_jobs(self) -> list[EvaluationJob]: ...
    def get_dashboard_data(self) -> DashboardData: ...
    def export_csv(self, job_id: str) -> str: ...  # returns file path
```

### run_evaluation flow (write path)

1. Resolve `config.model_name` → `ModelWrapper` via `ModelRegistry`.
2. Resolve `config.benchmark_name` → `Dataset` via `BenchmarkRegistry`.
3. Validate `model.supports(dataset.modality)` — raise early with a clear message if not.
4. Ask `ScriptGenerator` for an `sbatch` script.
5. Hand the script to the configured `Runner`; receive a `JobHandle`.
6. Persist an `EvaluationJob` with status `queued` via `ResultsRepository`.
7. Return the `job_id` immediately. Never block waiting for the job to finish.

### poll_jobs flow

For each in-progress job: ask the `Runner` for its current `JobStatus`. On
`completed`, call `ResultParser` → select the right `Metric` subclass(es) by
`dataset.modality` → `compute(predictions, targets)` → `ResultsRepository.save`.
On `failed`, update status and store an error message.

### Dashboard / read path

`get_dashboard_data()` queries `ResultsRepository` and shapes the result into a
`DashboardData` object the UI renders directly. No computation happens here.

---

## Constraints

- `EvaluationService.__init__` receives its dependencies (registries, runner, repository)
  by injection so tests can swap in fakes without touching config files.
- No Streamlit/Gradio imports here. The service must be importable from a plain Python
  script (e.g. a batch job or a test).
- All failures surface as structured return values or typed exceptions, never as
  unhandled crashes.
