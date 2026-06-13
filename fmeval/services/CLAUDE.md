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
    num_shots: int = 1                 # few-shot ICL: support examples per class
    picking_strategy: str = "random"   # "first" | "random" | "reversed"
    random_seed: int = 0
    # .dataset_params() bundles the three few-shot fields into the dict the
    # benchmark factory expects; benchmarks that don't use them ignore the keys.

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

@dataclass
class SampleDiff:
    """One sample present in both runs of a comparison; .agreement property
    returns both_correct / only_a_correct / only_b_correct / both_wrong."""
    sample_idx: int; input_text: str; correct_letter: str; raw_target: str
    predicted_a: str | None; predicted_b: str | None
    raw_prediction_a: str; raw_prediction_b: str
    a_correct: bool; b_correct: bool; metadata: dict

@dataclass
class RunComparison:
    """compare_runs output: 4-way agreement counts over the common samples
    (joined on sample_idx) + all SampleDiffs."""
    result_a: EvaluationResult; result_b: EvaluationResult
    n_common: int; both_correct: int; only_a_correct: int
    only_b_correct: int; both_wrong: int; diffs: list[SampleDiff]

@dataclass
class CategoryBreakdownRow:
    value: str; n_samples: int; n_correct: int; accuracy: float

@dataclass
class CategoryBreakdown:
    """Per-category accuracy slices of one run from sample metadata."""
    job_id: str; exp_id: str; model_name: str; key: str
    rows: list[CategoryBreakdownRow]
```

`COUNT_METRIC_KEYS` (module constant) = `{"n_samples", "n_unparseable"}` — metric
keys that are counts, excluded from the dashboard metric selector.

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

    # Run editing & deletion (metrics immutable; only exp_id/notes editable):
    def update_run(self, job_id, *, exp_id=None, notes=None) -> bool: ...
    def delete_run(self, job_id: str) -> bool: ...    # result + predictions + job record
    def delete_runs(self, job_ids: list[str]) -> int: ...

    # Analytics:
    def compare_runs(self, job_id_a, job_id_b) -> RunComparison: ...
    # ValueError if a run is missing or benchmarks differ; joins on sample_idx (intersection).
    def list_metadata_keys(self, job_id: str) -> list[str]: ...
    def get_category_breakdown(self, job_id: str, key: str) -> CategoryBreakdown: ...
    def get_category_breakdowns(self, job_ids, key) -> list[CategoryBreakdown]: ...
    def list_metric_keys(self, results) -> list[str]: ...
    # Headline metrics first, then rest alphabetically; count keys excluded.
```

### Job persistence

Jobs are persisted to the repository's `jobs` table (`save_job` on submit,
`update_job_status` on every status transition in `poll_jobs`). `__init__`
calls `_restore_jobs()`: terminal jobs are restored handle-less; non-terminal
jobs reattach via `runner.deserialize_handle()` when `runner_type` matches the
active runner (Slurm), otherwise they're marked failed with
"Orphaned by app restart." Exec time for a job finalized after a restart
includes the downtime (documented POC inaccuracy).

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
