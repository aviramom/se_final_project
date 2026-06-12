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

In mock mode only the **Mock models** are usable — GPU models (ChatTS-8B, Qwen3-VL-8B)
are blocked locally because they require cluster GPUs.

---

## Available models

| Registry name | UI display name | Requires GPU | Weights env var |
|---|---|---|---|
| `mock_always_a` | Mock Model (always A) | No | — |
| `mock_always_b` | Mock Model (always B) | No | — |
| `mock_always_c` | Mock Model (always C) | No | — |
| `chatts-8b` | ChatTS-8B (ByteDance Research) | Yes | `CHATTS_MODEL_PATH` |
| `qwen3-vl-8b` | Qwen3-VL-8B-Instruct (vision) | Yes | `QWEN_VL_MODEL_PATH` |

**ChatTS-8B** encodes raw time-series arrays natively via a patch-based MLP encoder
on top of a Qwen3-8B backbone.

**Qwen3-VL-8B-Instruct** is a general vision-language model. Before inference each
time series is plotted as a matplotlib chart (line + mean + ±1σ band) and passed as
an image, letting the model reason about patterns visually.

---

## Run on the Slurm cluster

The app connects to `slurm.bgu.ac.il` over SSH and submits `sbatch` jobs.
The cluster already has the code and a Python venv set up at `~/fmeval_project`.

### Prerequisites (one-time)

- SSH key copied to the cluster: `ssh-copy-id -i ~/.ssh/id_ed25519.pub aviramom@slurm.bgu.ac.il`
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

---

### Run with Mock models (no GPU, Slurm still used for dispatch)

```bash
FMEVAL_RUNNER=slurm \
  .venv/bin/streamlit run fmeval/app/main.py
```

---

### Run with ChatTS-8B

ChatTS-8B requires a GPU node. `SLURM_GPU_TYPE=rtx_3090` (or newer) is **required** —
the cluster also has GTX 1080 Ti nodes (sm_61 architecture) that are incompatible with
PyTorch 2.12+.

```bash
FMEVAL_RUNNER=slurm \
SLURM_PARTITION=main \
SLURM_GPUS=1 \
SLURM_GPU_TYPE=rtx_3090 \
SLURM_CPUS=4 \
SLURM_MEM_GB=24 \
SLURM_TIME_LIMIT=02:00:00 \
CHATTS_MODEL_PATH=/home/aviramom/models/chatts-8b \
  .venv/bin/streamlit run fmeval/app/main.py
```

If `CHATTS_MODEL_PATH` is omitted the worker downloads the model from HuggingFace Hub
on every job (~16 GB).

---

### Run with Qwen3-VL-8B-Instruct

Same GPU constraints as ChatTS-8B apply. The VL model also needs enough VRAM for
bfloat16 weights (~16 GB), so request a 24 GB card.

```bash
FMEVAL_RUNNER=slurm \
SLURM_PARTITION=main \
SLURM_GPUS=1 \
SLURM_GPU_TYPE=rtx_3090 \
SLURM_CPUS=4 \
SLURM_MEM_GB=24 \
SLURM_TIME_LIMIT=02:00:00 \
QWEN_VL_MODEL_PATH=/home/aviramom/models/qwen3-vl-8b \
  .venv/bin/streamlit run fmeval/app/main.py
```

---

### Run both GPU models in the same session

Set both weight paths — the app uses whichever model the user selects in the UI:

```bash
FMEVAL_RUNNER=slurm \
SLURM_PARTITION=main \
SLURM_GPUS=1 \
SLURM_GPU_TYPE=rtx_3090 \
SLURM_CPUS=4 \
SLURM_MEM_GB=24 \
SLURM_TIME_LIMIT=02:00:00 \
CHATTS_MODEL_PATH=/home/aviramom/models/chatts-8b \
QWEN_VL_MODEL_PATH=/home/aviramom/models/qwen3-vl-8b \
  .venv/bin/streamlit run fmeval/app/main.py
```

---

## Environment variable reference

| Env var | Default | What it controls |
|---|---|---|
| `FMEVAL_RUNNER` | `mock` | `mock` = local threads, `slurm` = cluster |
| `SLURM_PARTITION` | `cpu` | Slurm partition name |
| `SLURM_GPUS` | `0` | GPUs per node (0 = CPU-only) |
| `SLURM_GPU_TYPE` | *(any)* | GPU family constraint, e.g. `rtx_3090`. Always set for GPU jobs to avoid sm_61 nodes |
| `SLURM_CPUS` | `2` | CPUs per task |
| `SLURM_MEM_GB` | `16` | Memory in GB |
| `SLURM_TIME_LIMIT` | `01:00:00` | Wall-clock time limit (HH:MM:SS) |
| `CHATTS_MODEL_PATH` | *(HF Hub)* | Local path to ChatTS-8B weights on the cluster |
| `QWEN_VL_MODEL_PATH` | *(HF Hub)* | Local path to Qwen3-VL-8B-Instruct weights on the cluster |
| `SLURM_HOST` | `slurm.bgu.ac.il` | Cluster login node hostname |
| `SLURM_USER` | `aviramom` | SSH username |
| `SLURM_SSH_KEY` | `~/.ssh/id_ed25519` | Path to SSH private key |
| `SLURM_WORK_DIR` | `/home/aviramom/fmeval_jobs` | Job directory root on the cluster |
| `SLURM_PYTHON_BIN` | `/home/aviramom/fmeval_project/.venv/bin/python` | Python interpreter on the cluster |
| `SLURM_FMEVAL_DIR` | `/home/aviramom/fmeval_project` | fmeval repo root on the cluster (added to PYTHONPATH) |

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
    models/          ModelWrapper ABC + MockModel + ChatTSModel + QwenVLModel
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
~/models/
  chatts-8b/         Pre-downloaded ChatTS-8B weights (point CHATTS_MODEL_PATH here)
  qwen3-vl-8b/       Pre-downloaded Qwen3-VL-8B-Instruct weights (point QWEN_VL_MODEL_PATH here)
```
