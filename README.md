# Foundation Model Evaluation Platform

**Pick a model and a benchmark in a web UI. The platform runs inference asynchronously
(locally or on a Slurm GPU cluster) and shows the results in a comparison dashboard.
No per-benchmark adapter code needed.**

<!-- Screenshot: no image is committed yet. Add one (e.g. docs/dashboard.png) and reference it here. -->
> _Screenshot coming soon: the app has a **Run Evaluation** tab and a **Dashboard** tab._

---

## Features

- **No glue code per benchmark.** Every benchmark is a `Dataset` subclass that emits a
  standard `Sample` (prompt with `<TS_N>` placeholders plus raw numpy arrays). Every model is a
  `ModelWrapper` that formats `Sample`s the way it needs: arrays serialized as text,
  passed raw to a time-series encoder, or plotted as an image for a vision model.
- **Asynchronous execution.** Setting up a run and viewing results are separate steps.
  Set `FMEVAL_RUNNER` to choose where jobs run: `MockRunner` (local thread, the default)
  or `SlurmRunner` (sends `sbatch` jobs over SSH, checks `squeue`, then downloads `result.json`).
- **Jobs survive restarts.** Job state is saved to SQLite. When the app restarts,
  running Slurm jobs are picked up again.
- **Metrics chosen per dataset.** `MCQMetrics` scores multiple-choice (A–D) benchmarks.
  `ClassificationMetrics` scores free-label classification. Both report accuracy,
  balanced accuracy, F1, precision and recall (macro and weighted), plus a per-class
  breakdown and a count of unparseable answers.
- **Dashboard** with four tabs:
  - **Runs**: filter runs, chart any metric as bars, open a run to see its details,
    edit its notes or experiment ID, delete it, view the per-category breakdown, or export to CSV.
  - **Compare**: see how often two runs agree on each question, with their answers side by side.
  - **Trends**: plot a metric over time.
  - **Jobs**: see each job's status.
- **Built-in models:** fixed-answer mock models, a random-label chance baseline,
  **ChatTS-8B** (a time-series-native LLM) and **Qwen3-VL-8B-Instruct** (a vision LLM that
  receives each time series as a plotted chart).
- **Built-in benchmarks:** **TimeSeriesExam1** (746 multiple-choice questions, from
  HuggingFace) and few-shot classification on 94 datasets from the **UCR** time-series archive.

## Architecture

The design has five layers in a ports-and-adapters style. Each layer depends only on the
layers below it, so the whole pipeline can be tested without a cluster.

```
 Presentation    fmeval/app/         Streamlit: Run Evaluation + Dashboard (no business logic)
       │
 Orchestration   fmeval/services/    EvaluationService: the only class the UI calls
       │
 Domain          fmeval/core/        Dataset · ModelWrapper · Metric (ABCs) · Sample · SampleFormatter
       │
 Execution       fmeval/execution/   Runner ABC → MockRunner | SlurmRunner (+ cluster_worker.py)
                 fmeval/evaluation/  LocalEvaluationPipeline (runs on your machine or on a cluster node)
       │
 Data & config   fmeval/storage/     ResultsRepository (SQLite: results, per-sample predictions, jobs)
                 fmeval/config/      ModelRegistry · BenchmarkRegistry
```

**Run flow:**
1. The UI calls `EvaluationService.run_evaluation(config)`.
2. The service looks up the dataset and model in the registries.
3. It checks that the model supports the dataset's modality.
4. It sends the job to the active runner and returns a `job_id` right away.
5. `poll_jobs()` collects finished jobs and saves them to SQLite.
6. The dashboard reads from SQLite only.

**Extending:** a new benchmark needs one `Dataset` subclass (copy
`fmeval/core/datasets/template.py`) plus a registry entry. A new model needs one
`ModelWrapper` subclass plus a registry entry. Nothing else changes. You can find more detail
in `CLAUDE.md` and the `CLAUDE.md` file inside each package.

## Quick start (local mock mode)

You need Python 3.11 or newer. You don't need a cluster or a GPU.

```bash
uv venv .venv --python 3.11                        # or: python3.11 -m venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e . --python .venv/bin/python      # required so `fmeval` is importable

.venv/bin/streamlit run fmeval/app/main.py         # or: ./run.sh mock
```

Then open http://localhost:8501.

- Results are saved to `data/results.db`, which is created on the first run.
- Mock mode can only run the CPU models: the mock models and `random_label`. GPU models
  are blocked from running locally.
- TimeSeriesExam1 is downloaded from HuggingFace the first time you use it.
- The UCR benchmarks need a local copy of the archive. Set `UCR_DATA_PATH` to point to it
  (see below).

## Tests

```bash
.venv/bin/pytest
```

There are 264 tests covering datasets, formatters, metrics, model wrappers, the
pipeline, storage, the service layer and the Slurm runner. They need no GPU,
cluster or network, because torch, transformers and SSH are mocked. Last run: **264 passed**.

Lint and type checking:

```bash
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/mypy fmeval
```

## Project context

This is a university **Software Engineering final project**, built by Omer Aviram as a
proof of concept. The customer requirements are in [`ARD_Project.pdf`](ARD_Project.pdf).
The project was scoped deliberately: one user, no authentication, and a mock runner so
that demos don't depend on a live cluster queue.

---

## Reference

### Available models

| Registry name | UI display name | GPU | Weights env var |
|---|---|---|---|
| `mock_always_a` / `_b` / `_c` | Mock Model (always A/B/C) | No | — |
| `random_label` | Random Label (chance baseline) | No | — |
| `chatts-8b` | ChatTS-8B (ByteDance Research) | Yes | `CHATTS_MODEL_PATH` |
| `qwen3-vl-8b` | Qwen3-VL-8B-Instruct (vision) | Yes | `QWEN_VL_MODEL_PATH` |

**ChatTS-8B** reads raw time-series arrays directly. It uses a patch-based MLP encoder
on top of a Qwen3-8B backbone.

**Qwen3-VL-8B-Instruct** is a general vision-language model. Before inference, each time
series is drawn as a matplotlib chart (the line, its mean and a ±1σ band) and passed to
the model as an image.

### Running on a Slurm cluster

[`CLUSTER_SETUP.md`](CLUSTER_SETUP.md) covers one-time setup: an SSH key, a project copy and a
venv on the cluster, and a smoke test. After that, `run.sh` starts the app with sensible
GPU defaults:

```bash
./run.sh chatts    # ChatTS-8B on a Slurm GPU node
./run.sh qwen      # Qwen3-VL-8B-Instruct on a Slurm GPU node
./run.sh all       # both GPU models available in one session
```

You can also set the variables yourself:

```bash
FMEVAL_RUNNER=slurm SLURM_PARTITION=main SLURM_GPUS=1 SLURM_GPU_TYPE=rtx_3090 \
SLURM_CPUS=4 SLURM_MEM_GB=24 SLURM_TIME_LIMIT=02:00:00 \
CHATTS_MODEL_PATH=/home/<user>/models/chatts-8b \
QWEN_VL_MODEL_PATH=/home/<user>/models/qwen3-vl-8b \
  .venv/bin/streamlit run fmeval/app/main.py
```

Notes:
- **Always set `SLURM_GPU_TYPE`** (for example `rtx_3090`) for GPU jobs. The cluster also has
  GTX 1080 Ti nodes (sm_61), and recent PyTorch builds don't support them.
- If a weights path isn't set, the worker downloads the model from the HuggingFace Hub
  (about 16 GB) on every job. Qwen3-VL needs about 16 GB of VRAM for its bfloat16
  weights, so request a 24 GB card.
- To use mock models with Slurm handling job dispatch, set only `FMEVAL_RUNNER=slurm`.

To sync local code changes to the cluster, run this from the repository root:

```bash
rsync -av --progress \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.git' --exclude='data/' --exclude='fmeval.egg-info' \
  ./ <user>@<slurm-host>:~/fmeval_project/
```

### Environment variables

| Env var | Default | What it controls |
|---|---|---|
| `FMEVAL_RUNNER` | `mock` | `mock` = local threads, `slurm` = cluster |
| `SLURM_PARTITION` | `cpu` | Slurm partition name |
| `SLURM_GPUS` | `0` | GPUs per node (0 = CPU-only) |
| `SLURM_GPU_TYPE` | *(any)* | GPU family constraint, e.g. `rtx_3090`. Always set it for GPU jobs to avoid sm_61 nodes |
| `SLURM_CPUS` | `2` | CPUs per task |
| `SLURM_MEM_GB` | `16` | Memory in GB |
| `SLURM_TIME_LIMIT` | `01:00:00` | Wall-clock time limit (HH:MM:SS) |
| `CHATTS_MODEL_PATH` | *(HF Hub)* | Path to the ChatTS-8B weights on the cluster |
| `QWEN_VL_MODEL_PATH` | *(HF Hub)* | Path to the Qwen3-VL-8B-Instruct weights on the cluster |
| `UCR_DATA_PATH` | *(cluster default in `benchmark_registry.py`)* | Root of the UCR `Univariate_arff/` archive |
| `SLURM_HOST` | `slurm.bgu.ac.il` | Cluster login node hostname |
| `SLURM_USER` | `aviramom` | SSH username |
| `SLURM_SSH_KEY` | `~/.ssh/id_ed25519` | Path to the SSH private key |
| `SLURM_WORK_DIR` | `/home/aviramom/fmeval_jobs` | Root directory for jobs on the cluster |
| `SLURM_PYTHON_BIN` | `/home/aviramom/fmeval_project/.venv/bin/python` | Python interpreter on the cluster |
| `SLURM_FMEVAL_DIR` | `/home/aviramom/fmeval_project` | fmeval repo root on the cluster (added to `PYTHONPATH`) |

### Cluster commands

```bash
ssh <user>@<slurm-host> "squeue -u <user>"                            # running/pending jobs
ssh <user>@<slurm-host> "scancel <job_id>"                            # cancel a job
ssh <user>@<slurm-host> "cat ~/fmeval_jobs/<job-id>/slurm_*.out"      # stdout log
ssh <user>@<slurm-host> "cat ~/fmeval_jobs/<job-id>/slurm_*.err"      # stderr (tracebacks)
```

### Project layout

```
fmeval/
  app/               Streamlit UI (config page + dashboard)
  core/
    datasets/        Dataset ABCs, SampleFormatter, TimeSeriesExam1, UCR few-shot, JSONL template
    models/          ModelWrapper ABC, MockModel, RandomLabelModel, ChatTSModel, QwenVLModel
    metrics/         Metric ABC, MCQMetrics, ClassificationMetrics
  evaluation/        LocalEvaluationPipeline (runs locally or on a cluster node)
  execution/         Runner ABC, MockRunner, SlurmRunner, SlurmConfig, cluster_worker.py
  services/          EvaluationService (orchestration layer)
  storage/           SQLite ResultsRepository
  config/            ModelRegistry + BenchmarkRegistry
notebooks/           TimeSeriesExam1 demo notebook (+ script that regenerates it)
tests/               Unit + integration tests
data/results.db      SQLite results (auto-created, gitignored)
run.sh               Launcher: ./run.sh [mock|chatts|qwen|all]
```

### Cluster layout

```
~/fmeval_project/    Project source + .venv (synced with rsync)
~/fmeval_jobs/<job-id>/
    job.sh           sbatch script
    worker.py        evaluation worker (uploaded per job)
    result.json      metrics + predictions (written on completion)
    slurm_*.out/.err stdout / stderr logs
~/models/            Pre-downloaded weights (chatts-8b/, qwen3-vl-8b/)
```
