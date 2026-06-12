# Foundation Model Evaluation Platform

A centralized pipeline for evaluating and comparing foundation models across modalities.
Pick a model + benchmark in the UI, the system runs inference on a Slurm cluster, and
displays a comparison dashboard with accuracy, F1, and per-class breakdowns.

---

## Quick start (local / mock mode)

No cluster needed — jobs run on your Mac using `MockRunner`.

```bash
# 1. Create the virtual environment (Python 3.11+)
uv venv .venv --python 3.11   # or: python3 -m venv .venv

# 2. Install dependencies
uv pip install -r requirements.txt --python .venv/bin/python

# 3. Install the package in editable mode (required for imports)
uv pip install -e . --python .venv/bin/python

# 4. Launch the app
.venv/bin/streamlit run fmeval/app/main.py
```

Open http://localhost:8501 in your browser.

---

## Run on the Slurm cluster

The app connects to `slurm.bgu.ac.il` over SSH and submits `sbatch` jobs.
The cluster already has the code and a Python venv set up at `~/fmeval_project`.

### Prerequisites (one-time)

- SSH key copied to the cluster (`ssh-copy-id -i ~/.ssh/id_ed25519.pub aviramom@slurm.bgu.ac.il`)
- Verify it works: `ssh aviramom@slurm.bgu.ac.il "echo SSH OK"`

### Sync code changes to the cluster

Run this from your Mac any time you change the code locally:

```bash
rsync -av --progress \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.git' --exclude='data/' --exclude='fmeval.egg-info' \
  /Users/omeraviram/Projects/final_project/ \
  aviramom@slurm.bgu.ac.il:~/fmeval_project/
```

### Launch the app in Slurm mode

```bash
FMEVAL_RUNNER=slurm .venv/bin/streamlit run fmeval/app/main.py
```

Jobs are submitted to the `cpu` partition by default. Override any setting with env vars:

| Env var | Default | What it controls |
|---|---|---|
| `FMEVAL_RUNNER` | `mock` | `mock` = local, `slurm` = cluster |
| `SLURM_PARTITION` | `cpu` | Slurm partition name |
| `SLURM_GPUS` | `0` | GPUs per node (0 = CPU-only) |
| `SLURM_CPUS` | `2` | CPUs per task |
| `SLURM_MEM_GB` | `16` | Memory in GB |
| `SLURM_TIME_LIMIT` | `01:00:00` | Wall-clock time limit |

Example — run on a GPU node:

```bash
FMEVAL_RUNNER=slurm SLURM_PARTITION=rtx3090 SLURM_GPUS=1 \
  .venv/bin/streamlit run fmeval/app/main.py
```

---

## Run the tests

```bash
.venv/bin/pytest
```

---

## Other useful commands

```bash
# Lint + format
.venv/bin/ruff check . && .venv/bin/ruff format .

# Type check
.venv/bin/mypy fmeval

# Check running/pending cluster jobs
ssh aviramom@slurm.bgu.ac.il "squeue -u aviramom"

# Cancel a cluster job
ssh aviramom@slurm.bgu.ac.il "scancel <job_id>"

# Read stdout log for a job
ssh aviramom@slurm.bgu.ac.il "cat ~/fmeval_jobs/<job-id>/slurm_*.out"

# Read stderr log (tracebacks) for a job
ssh aviramom@slurm.bgu.ac.il "cat ~/fmeval_jobs/<job-id>/slurm_*.err"
```

---

## Project layout

```
fmeval/
  app/               Streamlit UI (config page + dashboard)
  core/
    datasets/        Dataset ABCs + TimeSeriesExam1 benchmark
    models/          ModelWrapper ABC + MockModel
    metrics/         MCQMetrics (accuracy, F1, per-class)
  evaluation/        LocalEvaluationPipeline (runs on Mac or cluster)
  execution/         Runner ABC, MockRunner, SlurmRunner, cluster_worker.py
  services/          EvaluationService (orchestration layer)
  storage/           SQLite ResultsRepository
  config/            ModelRegistry + BenchmarkRegistry
data/
  results.db         SQLite results (auto-created on first run, Mac only)
tests/               Unit + integration tests
```

---

## Cluster layout (on `slurm.bgu.ac.il`)

```
~/fmeval_project/    Full project source + .venv (synced from Mac via rsync)
~/fmeval_jobs/       One subdirectory per submitted job:
  <job-id>/
    job.sh           sbatch script
    worker.py        evaluation worker (uploaded per job)
    result.json      metrics + predictions (written on completion)
    slurm_*.out      stdout log
    slurm_*.err      stderr log
```
