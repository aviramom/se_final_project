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
  types.py                ← EvaluationConfig, DashboardData, ResultsFilter, GroupedResult
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
    max_samples: int = 50
    exp_id: str = ""   # free-text label; "" → auto-slug "run-YYYYMMDD-HHMMSS" in service

@dataclass
class DashboardData:
    """Read-path payload shaped for UI rendering of job statuses and result comparisons."""
    jobs: list[EvaluationJob]        # TYPE_CHECKING import
    results: list[EvaluationResult]  # TYPE_CHECKING import

@dataclass
class ResultsFilter:
    """Multi-value filter bag passed to EvaluationService.query_results().
    None on any field = no restriction. Empty list treated as None."""
    model_names: list[str] | None = None
    benchmark_names: list[str] | None = None
    exp_ids: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None

@dataclass
class GroupedResult:
    """Aggregated view of runs sharing the same exp_id × model × benchmark.
    std_metrics values are 0.0 for groups with only one run.
    n_samples and n_unparseable are summed; all other metrics are averaged."""
    exp_id: str
    model_name: str
    benchmark_name: str
    n_runs: int
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    run_timestamps: list[datetime]
```

---

## EvaluationService (`evaluation_service.py`)

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
    def list_exp_ids(self) -> list[str]: ...          # distinct exp_ids from DB
    def query_results(self, filters: ResultsFilter) -> list[EvaluationResult]: ...
    def group_results(self, results: list[EvaluationResult]) -> list[GroupedResult]: ...
    def export_csv(self, job_id: str | None = None) -> str: ...
    def get_run_detail(self, job_id: str) -> list[SamplePrediction]: ...
    # Returns per-sample predictions ordered by sample_idx; [] for unknown/pre-feature jobs.
```

### run_evaluation flow (write path)

1. Resolve `config.model_name` → `ModelWrapper` via `ModelRegistry`.
2. Resolve `config.benchmark_name` → `Dataset` via `BenchmarkRegistry`.
3. Validate `model.supports(dataset.modality)` — raise `ValueError` early if not.
4. Derive `exp_id`: `config.exp_id.strip() or datetime.now().strftime("run-%Y%m%d-%H%M%S")`.
5. Hand job to the configured `Runner`; receive a handle.
6. Track `EvaluationJob` in `self._jobs` (in-memory, session-scoped).
7. Return the `job_id` immediately — never blocks.

### poll_jobs flow

For each non-terminal job: ask the `Runner` for current `JobStatus`. On `completed`,
call `_finalize_job` → retrieve `RunResult` → build `EvaluationResult` (with `exp_id`
and `max_samples` from the job) → `ResultsRepository.save` + `ResultsRepository.save_sample_predictions`.
On `failed`, record error.

### query_results / group_results

`query_results` pushes date filters to the DB via `repository.query()`, then
post-filters multi-value lists in Python (model_names, benchmark_names, exp_ids).

`group_results` aggregates by `(exp_id, model_name, benchmark_name)` using
`statistics` stdlib. `n_samples` and `n_unparseable` are summed; all other metric
keys are mean-averaged with stdev computed alongside. Lives here (not in the
repository and not in the UI) because it is business logic with a defined contract.

---

## Constraints

- `EvaluationService.__init__` receives its dependencies by injection so tests can
  swap in fakes without touching config files.
- No Streamlit/Gradio imports here. The service must be importable from a plain
  Python script (e.g. a batch job or a test).
- All failures surface as structured return values or typed exceptions, never as
  unhandled crashes.
