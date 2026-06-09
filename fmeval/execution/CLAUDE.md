# fmeval/execution/ — Layer 4: Execution

Handles everything between "a run has been requested" and "a RunResult is available."
The `Runner` ABC is the key seam — swap implementations without touching anything else.

**Dependency rule:** `execution/` may import from `core/` and `evaluation/`. It must
not import from `app/`, `services/`, or `storage/`.

---

## Files

```
execution/
  __init__.py       ✅ exports EvaluationJob, JobStatus, Runner, MockRunner
  CLAUDE.md
  job.py            ✅ EvaluationJob dataclass + JobStatus enum
  runner.py         ✅ Runner ABC
  mock_runner.py    ✅ MockRunner (ThreadPoolExecutor, local CPU)
```

Slurm execution is deferred. When ready, add `slurm_runner.py` that implements
`Runner` without changing any other layer.

---

## EvaluationJob & JobStatus (`job.py`)

```python
class JobStatus(str, Enum):   # inherits str — serializes without .value
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class EvaluationJob:
    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    status: JobStatus
    created_at: datetime
    max_samples: int
    exp_id: str = ""           # experiment label; propagated to EvaluationResult on completion
    handle: Any = None         # Future for MockRunner; Slurm job ID for SlurmRunner
    error_message: str | None = None
```

`handle` is typed `Any` — the service never inspects it directly.

---

## Runner ABC (`runner.py`)

```python
class Runner(ABC):
    @abstractmethod
    def submit(self, job: EvaluationJob, dataset: Dataset, model: ModelWrapper) -> Any:
        """Start the job; return a runner-specific handle stored on job.handle."""
        ...

    @abstractmethod
    def get_status(self, job: EvaluationJob) -> JobStatus:
        """Return current status by inspecting job.handle."""
        ...

    @abstractmethod
    def get_result(self, job: EvaluationJob) -> RunResult:
        """Return the completed RunResult. Raises RuntimeError if not yet done."""
        ...
```

The interface passes `dataset` and `model` directly (no `ScriptGenerator` for the POC).
A future `SlurmRunner` would generate the sbatch script internally.

---

## MockRunner (`mock_runner.py`)

Runs `LocalEvaluationPipeline` in a `ThreadPoolExecutor` (max 2 workers). Uses
`concurrent.futures.Future` for status polling — `.done()`, `.running()`, `.exception()`
and `.result()` are all available without a lock on the result itself.

`MCQMetrics` is hardcoded as the metric — all current datasets are multimodal MCQ.
When a non-MCQ dataset is added, pass a metric factory into `MockRunner.__init__`.

```python
class MockRunner(Runner):
    def __init__(self, max_workers: int = 2) -> None: ...
    def submit(self, job, dataset, model) -> Future: ...
    def get_status(self, job) -> JobStatus: ...
    def get_result(self, job) -> RunResult: ...
```

---

## Constraints

- `MockRunner` and any future `PrecomputedRunner` must be usable with no cluster
  credentials and no network access beyond the initial dataset download.
- The same `ModelWrapper.predict` code path runs regardless of runner.
- Never add Slurm-specific logic to `Runner` ABC or `EvaluationJob` — keep them generic.
