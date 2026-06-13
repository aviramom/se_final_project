# fmeval/execution/ — Layer 4: Execution

Handles everything between "a run has been requested" and "a RunResult is available."
The `Runner` ABC is the key seam — swap implementations without touching anything else.

**Dependency rule:** `execution/` may import from `core/` and `evaluation/`. It must
not import from `app/`, `services/`, or `storage/`.

---

## Files

```
execution/
  __init__.py         ✅ exports EvaluationJob, JobStatus, Runner, MockRunner, SlurmRunner, SlurmConfig
  CLAUDE.md
  job.py              ✅ EvaluationJob dataclass + JobStatus enum
  runner.py           ✅ Runner ABC
  mock_runner.py      ✅ MockRunner (ThreadPoolExecutor, local CPU)
  slurm_config.py     ✅ SlurmConfig dataclass (SSH + Slurm resource parameters)
  slurm_runner.py     ✅ SlurmRunner (submits sbatch jobs over SSH, polls squeue)
  cluster_worker.py   ✅ Worker script uploaded to cluster per job; runs LocalEvaluationPipeline
```

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
    dataset_params: dict = {}  # benchmark-specific dataset hints (few-shot k/strategy/seed)
```

`dataset_params` carries the few-shot ICL options from `EvaluationConfig`.
`SlurmRunner` emits them as worker CLI flags (`--num-shots/--picking-strategy/
--random-seed`) so the cluster reconstructs the dataset exactly as configured.
It is **not** persisted to `JobRecord` — restored jobs are only polled, by which
point submission (which baked the params into the remote `job.sh`) already happened.

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

    def get_error_log(self, job: EvaluationJob) -> str | None:
        """Return a brief error log snippet for a failed job, or None.
        Default returns None. SlurmRunner overrides to fetch the .err file tail."""
        return None

    runner_type: str = "unknown"   # class attr; "mock" / "slurm" — persisted with each job

    def serialize_handle(self, handle: Any) -> str | None:
        """JSON for persistence, or None (default). MockRunner keeps the default —
        Futures can't survive a restart. SlurmRunner serializes SlurmHandle
        (three plain strings) so a restarted app can reattach."""

    def deserialize_handle(self, handle_json: str) -> Any:
        """Rebuild a handle from serialize_handle() output (default None)."""
```

The interface passes `dataset` and `model` directly. `SlurmRunner` generates the
sbatch script and uploads the worker internally — callers never see those details.

`get_error_log` is a non-abstract optional method. `EvaluationService.poll_jobs()`
calls it when a job transitions to FAILED and stores the result as `job.error_message`,
which is shown in the UI. `MockRunner` inherits the default `None` return.

---

## MockRunner (`mock_runner.py`)

Runs `LocalEvaluationPipeline` in a `ThreadPoolExecutor` (max 2 workers). Uses
`concurrent.futures.Future` for status polling — `.done()`, `.running()`, `.exception()`
and `.result()` are all available without a lock on the result itself.

The metric comes from `dataset.metric` (each `Dataset` declares its evaluation
method — `MCQMetrics` for MCQ benchmarks, `ClassificationMetrics` for UCR ICL).
`cluster_worker.py` does the same. Neither hardcodes a metric.

```python
class MockRunner(Runner):
    def __init__(self, max_workers: int = 2) -> None: ...
    def submit(self, job, dataset, model) -> Future: ...
    def get_status(self, job) -> JobStatus: ...
    def get_result(self, job) -> RunResult: ...
```

---

## SlurmRunner (`slurm_runner.py`) + SlurmConfig (`slurm_config.py`)

Submits jobs to `slurm.bgu.ac.il` over SSH. On `submit()`:
1. SSH `mkdir` the job directory on the cluster (`~/fmeval_jobs/<job_id>/`).
2. SCP `cluster_worker.py` → `worker.py` in that directory.
3. Build and SCP an `sbatch` script (`job.sh`) with the resource spec from `SlurmConfig`.
4. SSH `sbatch job.sh` and parse the returned Slurm job ID into a `SlurmHandle`.

On `get_status()`: SSH `squeue` for that job ID; map Slurm state strings → `JobStatus`.
On `get_result()`: SCP `result.json` written by the worker; deserialise into `RunResult`.
On `get_error_log()`: SSH `cat {remote_job_dir}/slurm_*.err | tail -40`; returns the
Python traceback from a failed job so `EvaluationService` can surface it in the UI.

`SlurmConfig` fields: `host`, `user`, `remote_work_dir`, `ssh_key_path`, `partition`,
`time_limit`, `gpus_per_node`, `gpu_type`, `cpus_per_task`, `mem_gb`, `python_bin`,
`fmeval_dir`, `env_setup_commands` (list of shell lines prepended to the sbatch script,
e.g. `module load`).

`gpu_type` (optional, e.g. `"rtx_3090"`) refines the `--gres` directive to
`--gres=gpu:rtx_3090:N`, targeting a specific GPU family. Without it, Slurm picks any
available GPU. **Always set this for real model runs** — the cluster has Pascal (GTX
1080 Ti, sm_61) nodes that are incompatible with PyTorch 2.12 (min sm_75). Use
`rtx_2080`, `rtx_3090`, `rtx_4090`, or `rtx_6000`. Set via `SLURM_GPU_TYPE` env var.

**Cluster layout** (per job):
```
~/fmeval_jobs/<job_id>/
  job.sh            sbatch script
  worker.py         cluster_worker.py uploaded from local repo
  result.json       written by worker on success
  slurm_<N>.out     stdout log
  slurm_<N>.err     stderr log
```

## cluster_worker.py

Standalone script that runs inside the Slurm job. Accepts `--dataset`, `--model`,
`--max-samples`, `--output`, and optional few-shot flags `--num-shots`,
`--picking-strategy`, `--random-seed` (forwarded to `registry.get(dataset_params=…)`). Imports `fmeval` from `~/fmeval_project` on the
cluster (installed as editable via `pip install -e .`), runs `LocalEvaluationPipeline`,
and writes `result.json`. Prints `FMEVAL_RESULT_WRITTEN:<path>` as a sentinel line that
`SlurmRunner.get_result()` looks for before fetching the file.

---

## Constraints

- `MockRunner` must be usable with no cluster credentials and no network access beyond
  the initial dataset download.
- The same `ModelWrapper.predict` code path runs regardless of runner.
- Never add Slurm-specific logic to `Runner` ABC or `EvaluationJob` — keep them generic.
- `model_name` returned by any `ModelWrapper` must be **lowercase** — it is used as a
  CLI argument passed to `cluster_worker.py` and looked up in the registry on the cluster.
