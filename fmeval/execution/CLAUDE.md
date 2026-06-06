# fmeval/execution/ — Layer 4: Execution

Handles everything between "a run has been requested" and "raw output logs exist on
disk." The key design point: the `Runner` ABC is the POC seam — swap implementations
without touching anything else.

**Dependency rule:** `execution/` may import from `core/` (to type-hint `Dataset`,
`ModelWrapper`, `Sample`). It must not import from `app/`, `services/`, or `storage/`.

---

## Files

```
execution/
  CLAUDE.md
  script_generator.py   ← ScriptGenerator
  runner.py             ← Runner ABC + JobHandle
  slurm_runner.py       ← SlurmRunner
  mock_runner.py        ← MockRunner
  precomputed_runner.py ← PrecomputedRunner
  result_parser.py      ← ResultParser
  job.py                ← EvaluationJob dataclass + JobStatus enum
```

---

## EvaluationJob & JobStatus (`job.py`)

```python
class JobStatus(Enum):
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
    handle: Any          # runner-specific handle (e.g. slurm job id, file path)
    created_at: datetime
    error_message: str | None = None
```

---

## ScriptGenerator (`script_generator.py`)

Builds a self-contained `sbatch` script from an `EvaluationConfig`. The script
activates the environment, calls the evaluation entry point with the right model and
benchmark arguments, and writes predictions + targets to a log file. Returns the
script as a string (caller writes it to disk or passes it to the runner).

---

## Runner ABC (`runner.py`)

```python
class Runner(ABC):
    @abstractmethod
    def submit(self, script: str, job: EvaluationJob) -> Any:
        """Submit the script; return a runner-specific handle."""
        ...

    @abstractmethod
    def get_status(self, handle: Any) -> JobStatus:
        """Poll the runner for current job status."""
        ...

    @abstractmethod
    def get_output_path(self, handle: Any) -> Path:
        """Return the path where the job wrote its output log."""
        ...
```

### Implementations

`SlurmRunner` — shells out to `sbatch` to submit and `squeue` to poll status.
For the POC this may be left as a stub that raises `NotImplementedError` if the
cluster is unreachable.

`MockRunner` — runs the evaluation inline (no subprocess), using a tiny dummy dataset
from `data/dummy/`. Completes synchronously; returns `COMPLETED` on the first poll.
This is how the demo runs without a cluster.

`PrecomputedRunner` — ignores the script entirely and returns a handle that points to
a pre-existing log file in `data/precomputed/`. Allows instant dashboard demos from
canned results.

---

## ResultParser (`result_parser.py`)

Reads the raw output log produced by a completed job and returns two parallel
sequences: `predictions` and `targets`, typed correctly for the modality (list of
strings for text, numpy arrays for time-series). The parser must not compute metrics —
it only extracts and structures the raw data.

---

## Constraints

- The same `ModelWrapper.predict` code path runs in all three runner modes. The runner
  controls *where* it executes; the model wrapper controls *how*.
- `MockRunner` and `PrecomputedRunner` must be fully usable with no network access and
  no cluster credentials.
