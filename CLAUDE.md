# CLAUDE.md

This file gives Claude Code the context it needs to work on this project. Keep it short, high-signal, and up to date — when a convention or command changes, edit this file.

---

## 1. What this project is

**Foundation Model Evaluation Platform** — a centralized, lightweight pipeline for evaluating and comparing foundation models across modalities (currently **text** and **time-series**).

The problem it solves: today, running benchmark X on model Y requires hand-written adapter code, because every benchmark has a different structure and every model expects inputs formatted differently. This platform removes that glue work. A user picks pre-loaded models + benchmarks in a web UI, the system standardizes the data, runs inference (on a Slurm cluster, asynchronously), parses results, and shows a comparison dashboard.

It is a university **Software Engineering final project**. The graded deliverable is a working **proof of concept (POC)**, not a production system. Scope discipline matters more than feature breadth.

---

## 2. Architecture

Strict layered design (lightweight ports-and-adapters). Five layers; dependencies point **inward** toward the domain. The two core abstractions sit at the center and depend on nothing. Each layer knows only the layer directly below it — this is what makes the whole pipeline testable without a live cluster.

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 · Presentation       Streamlit / Gradio    │
│           ConfigPage  ·  DashboardPage               │
├─────────────────────────────────────────────────────┤
│  Layer 2 · Orchestration      EvaluationService     │
├─────────────────────────────────────────────────────┤
│  Layer 3 · Domain             Dataset (ABC)         │
│                               ModelWrapper (ABC)    │
│                               Metric (ABC)          │
│                               Sample (dataclass)    │
├─────────────────────────────────────────────────────┤
│  Layer 4 · Execution          ScriptGenerator       │
│                               Runner (ABC)          │
│                               ResultParser          │
│                               EvaluationJob         │
├─────────────────────────────────────────────────────┤
│  Layer 5 · Data & Config      ResultsRepository     │
│                               ModelRegistry         │
│                               BenchmarkRegistry     │
└─────────────────────────────────────────────────────┘
```

### Layer 1 — Presentation

Two pages, both in Streamlit/Gradio. `ConfigPage` populates dropdowns from the registries, collects evaluation parameters, and on "Run" calls one method on `EvaluationService`. `DashboardPage` is split into four sub-tabs: **Runs** (filterable results, any-metric bar chart, drill-in run detail with edit/delete and category breakdown), **Compare** (run-vs-run per-question agreement + side-by-side answers), **Trends** (metric over time), **Jobs** (persisted job statuses with badges). Shared styling/cards live in `ui_components.py`; the theme is in `.streamlit/config.toml`.

Hard rule: no business logic in the UI. It never reads a raw benchmark file or computes a metric — it only calls `EvaluationService` and renders the result.

### Layer 2 — Orchestration

`EvaluationService` is the single class the UI talks to. It owns the workflow and does none of the actual work. Public surface: `list_models()`, `list_benchmarks()` (delegates to registries), `run_evaluation(config) -> job_id`, `poll_jobs()`, `get_dashboard_data()`, `list_exp_ids()`, `query_results(filters)`, `group_results(results)`, `export_csv()`, plus run management (`update_run` — exp_id/notes, `delete_run`/`delete_runs`) and analytics (`compare_runs`, `get_category_breakdown(s)`, `list_metadata_keys`, `list_metric_keys`). Jobs are persisted to the DB and restored on startup (`_restore_jobs`): running Slurm jobs reattach via serialized handles; orphaned mock jobs are marked failed. Everything below is invoked by the service; nothing below ever calls back up.

### Layer 3 — Domain (core abstractions)

These three ABCs and one dataclass are the heart of the system. Get their interfaces right before generating code around them. A change here that forces edits in any other layer means the abstraction is leaking — flag it rather than papering over it.

`Dataset` (ABC) wraps a benchmark in its **original file format** and exposes a `modality` property plus an iterator yielding `Sample` objects. Adding a benchmark = one new subclass, zero changes elsewhere. The Data Immutability constraint lives here — standardization happens in code, never by editing raw files. All current benchmarks subclass `MultimodalDataset` which fixes `modality="multimodal"` and enforces placeholder validation.

`Sample` (dataclass) — **implemented** — the standardized unit between `Dataset` and `ModelWrapper`: `input_text` (NL prompt with `<TS_N>` placeholders), `input_ts` (list of raw numpy arrays), `output` (ground-truth text string), `metadata` (dict).

`SampleFormatter` (in `core/datasets/formatters.py`) — **implemented** — converts between the canonical **separate** form (placeholders + raw arrays) that all datasets emit, and the **combined** form (TS serialized inline in text) that LLMs consume. `ModelWrapper` declares `input_mode` and calls the formatter in `format_input`.

`ModelWrapper` (ABC) declares `supported_modalities`, `input_mode` (`"combined"` or `"separate"`), `format_input(sample)`, and `predict(...) -> list[str]`. All models output text. `predict` is the code that runs on the cluster; in mock mode the same code runs locally.

`Metric` (ABC) — **implemented** — exposes `applicable_modalities`, `compute(predictions, targets) -> dict[str, float]`, and `label_predictions(predictions, targets)` (per-sample answer-token extraction the pipeline uses for is_correct). Two concrete metrics: `MCQMetrics` (A–D letters) and `ClassificationMetrics` (free class labels, e.g. UCR ICL — balanced accuracy over an arbitrary label set). Both return accuracy, balanced\_accuracy, F1, precision, recall (macro + weighted), n\_unparseable, and per-class breakdowns in one call.

**Per-dataset evaluation method.** A `Dataset` declares its `metric` (defaults to `MCQMetrics` on the ABC; `UCRICLDataset` overrides to `ClassificationMetrics`). The runners read `dataset.metric` and the pipeline scores each sample via `metric.label_predictions`, so nothing hardcodes a metric — adding a benchmark with a new answer format = one dataset subclass plus (if needed) one metric.

**Modality as the compatibility key.** A `Dataset` declares its modality (`"multimodal"` for all current datasets); that single tag determines which `ModelWrapper`s are compatible. No `if benchmark == X` branching anywhere — modality drives model routing, the dataset's `metric` drives scoring.

### Layer 4 — Execution

`Runner` (ABC) provides `submit(job, dataset, model) -> handle`, `get_status(job)`, and `get_result(job)` — the most important POC seam. Two implementations are active: `MockRunner` (runs `LocalEvaluationPipeline` in a local thread) and `SlurmRunner` (uploads a worker script + sbatch job over SSH, polls `squeue`, fetches `result.json` on completion). Swap the runner via the `FMEVAL_RUNNER` env var; nothing else changes. `SlurmConfig` holds all SSH and resource parameters. `cluster_worker.py` is the script uploaded per job. `JobStatus` (enum): `queued / running / completed / failed`. `EvaluationJob` (dataclass) carries job state across the async boundary.

### Layer 5 — Data & Configuration

`ResultsRepository` is a thin SQLite wrapper with three tables: `evaluation_results` (model name, benchmark name, modality, metrics dict, timestamp, execution time, `exp_id`, `max_samples`, `notes`), `sample_predictions` (per-question records), and `jobs` (persisted `JobRecord` snapshots so jobs survive app restarts). Schema migrations run automatically in `__init__` so existing DBs upgrade on first open. `ModelRegistry` and `BenchmarkRegistry` are loaded at startup and answer two questions: "what's available?" for the UI, and "give me object X" for the service.

### Key interactions

**Run flow (write path).** `ConfigPage → EvaluationService.run_evaluation(config)`. The service resolves names through the registries into a `Dataset` and a `ModelWrapper`, checks modality compatibility, submits to the active runner (`MockRunner` or `SlurmRunner`, chosen by `FMEVAL_RUNNER` env var), and returns a `job_id` immediately — the UI never blocks. Later, `poll_jobs()` asks the runner for status; on completion it saves an `EvaluationResult` to `ResultsRepository`.

**Dashboard (read path).** `DashboardPage → get_dashboard_data() → ResultsRepository.query → charts`. The read and write paths are fully independent, which is what satisfies the "configuration phase vs. asynchronous results-viewing phase" requirement.

### Layout

```
fmeval/
  __init__.py     # package root
  app/                              # ✅ Streamlit UI
    main.py                         # ✅ entry point + @st.cache_resource singleton
    config_page.py                  # ✅ render_config_page(service)
    dashboard_page.py               # ✅ render_dashboard_page(service) — Runs/Compare/Trends/Jobs sub-tabs
    ui_components.py                # ✅ shared CSS, status badges, Q/A + diff cards (pure rendering)
  core/
    sample.py                       # ✅ Sample dataclass (input_text, input_ts, output)
    datasets/
      base.py                       # ✅ Dataset ABC
      base_multimodal.py            # ✅ MultimodalDataset (modality fixed, validation)
      formatters.py                 # ✅ SampleFormatter (combined ↔ separate)
      template.py                   # ✅ JSONLMultimodalDataset (copy-paste template)
      tsexam1.py                    # ✅ TimeSeriesExam1Dataset (HuggingFace, 746 MCQ)
      ucr_icl.py                    # ✅ UCRICLDataset (UCR ARFF few-shot ICL classification; lazy, UCR_DATA_PATH)
    models/
      base.py                       # ✅ ModelWrapper ABC
      mock_model.py                 # ✅ MockModel (fixed-answer baseline)
      random_label_model.py         # ✅ RandomLabelModel (CPU chance baseline; parses options from prompt)
      chatts_model.py               # ✅ ChatTSModel (bytedance-research/ChatTS-8B, separate mode, lazy loading, cluster-verified)
      __init__.py                   # ✅
    metrics/
      base.py                       # ✅ Metric ABC (compute + label_predictions)
      mcq_metrics.py                # ✅ MCQMetrics + extract_letter()
      classification_metrics.py     # ✅ ClassificationMetrics + extract_label() (free class labels)
      __init__.py                   # ✅
  evaluation/                       # ✅ Local synchronous evaluation pipeline
    pipeline.py                     # ✅ LocalEvaluationPipeline
    result.py                       # ✅ RunResult, SamplePrediction
    __init__.py                     # ✅
  execution/                        # ✅ Async execution layer
    job.py                          # ✅ EvaluationJob, JobStatus
    runner.py                       # ✅ Runner ABC
    mock_runner.py                  # ✅ MockRunner (ThreadPoolExecutor, local CPU)
    slurm_config.py                 # ✅ SlurmConfig dataclass (SSH + resource params, incl. gpu_type)
    slurm_runner.py                 # ✅ SlurmRunner (sbatch over SSH, polls squeue)
    cluster_worker.py               # ✅ Worker uploaded per job; runs LocalEvaluationPipeline
    __init__.py                     # ✅
  storage/                          # ✅ SQLite persistence
    models.py                       # ✅ EvaluationResult (incl. notes) + JobRecord dataclasses
    repository.py                   # ✅ ResultsRepository (results, sample_predictions, jobs tables; update/delete)
    __init__.py                     # ✅
  config/                           # ✅ Registries
    model_registry.py               # ✅ ModelInfo, ModelRegistry, build_default_model_registry() — registers mock_always_{a,b,c} + chatts-8b
    benchmark_registry.py           # ✅ BenchmarkInfo, BenchmarkRegistry, build_default_benchmark_registry()
    __init__.py                     # ✅
  services/
    __init__.py                     # ✅
    types.py                        # ✅ EvaluationConfig, DashboardData, ResultsFilter, GroupedResult, RunComparison, SampleDiff, CategoryBreakdown
    evaluation_service.py           # ✅ EvaluationService (run/poll/restore jobs, edit/delete runs, compare, breakdowns)
data/
  results.db                        # SQLite results DB (auto-created on first run)
notebooks/
  tsexam1_demo.ipynb                # ✅ Executed demo: sample inspection + evaluation walkthrough
  build_notebook.py                 # script that regenerates the notebook
tests/
  core/
    test_sample.py                  # ✅
    test_formatters.py              # ✅
    test_jsonl_dataset.py           # ✅
    test_tsexam1.py                 # ✅
    test_mcq_metrics.py             # ✅
    test_classification_metrics.py  # ✅ ClassificationMetrics + extract_label() (label set, longest-first)
    test_ucr_icl.py                 # ✅ UCRICLDataset (synthetic ARFF, end-to-end via RandomLabelModel)
    test_mock_model.py              # ✅
    test_chatts_model.py            # ✅ ChatTSModel: format_input + predict (mocked, no GPU)
  evaluation/
    test_pipeline.py                # ✅ LocalEvaluationPipeline + RunResult (30 tests)
  storage/
    test_repository.py              # ✅ ResultsRepository: save/query/list_exp_ids/migration
  services/
    test_evaluation_service.py      # ✅ query_results, group_results, exp_id auto-slug
  execution/
    test_slurm_runner.py            # ✅ SlurmRunner integration tests
pyproject.toml    # package install config (pip install -e .)
ARD_Project.pdf   # ground-truth requirements — consult when unsure
CLAUDE.md
README.md         # setup + launch commands (local and Slurm)
```

---

## 4. Tech stack & hard constraints

- **Language:** Python. Everything stays Python-native so it plugs into the existing research codebase.
- **UI:** Python-native only — **Streamlit or Gradio** (pick one and commit to it). No JS frameworks.
- **Compute:** real execution targets a **Slurm**-managed GPU cluster via generated `sbatch` scripts.
- **Storage:** local DB / structured files — **SQLite** is the natural fit for the POC.
- **Async by nature:** evaluations are NOT real-time. Jobs go into a queue; the UI must clearly separate the *configuration* phase from the *results-viewing* phase. The app must stay responsive (screen transitions / standard queries < 2s) while jobs run.
- **Metrics:** multimodal MCQ tasks → `MCQMetrics` (accuracy, balanced\_accuracy, F1, precision, recall — all macro + weighted + per-class).
- **Stored per run:** model name, benchmark name, modality, timestamp, execution time, computed metrics, `exp_id` (experiment label), `max_samples`, `notes` (editable free text). `exp_id` and `notes` are the only editable fields; metrics are immutable. Runs can be deleted (result + sample predictions + job record).

### POC / demo constraint (important)
Real runs can take hours and depend on cluster availability, so the system **supports an offline/simulated mode** via `MockRunner` (local thread) and a **live cluster mode** via `SlurmRunner` — switchable with `FMEVAL_RUNNER=mock|slurm`. The demo never depends on a live queue; set `FMEVAL_RUNNER=mock` (the default) to run everything locally.

---

## 5. Conventions

- Follow PEP 8. Use type hints on all public functions and class interfaces.
- Define the dataset and model interfaces as abstract base classes (`abc.ABC`); concrete benchmarks/models subclass them.
- Keep functions small and single-purpose. No business logic inside UI callbacks — UI calls into `core`/`execution`, never the reverse.
- Handle external failures gracefully: cluster timeouts, node crashes, missing logs, malformed benchmark files. Catch, log, and surface a clear failure **status** in the UI — never crash the app.
- Docstrings on every public class/function explaining the *why*, not just the *what*.

---

## 6. Commands

```bash
# environment: .venv at repo root (Python 3.11, managed with uv)
# install deps:   uv pip install -r requirements.txt --python .venv/bin/python
# install pkg:    uv pip install -e . --python .venv/bin/python   ← required once for imports to work

# run the app (local mock mode):
#   .venv/bin/streamlit run fmeval/app/main.py

# run the app (Slurm cluster mode — GPU jobs with ChatTS-8B):
#   FMEVAL_RUNNER=slurm \
#   SLURM_PARTITION=main \
#   SLURM_GPUS=1 \
#   SLURM_GPU_TYPE=rtx_3090 \       ← required: avoids sm_61 nodes (GTX 1080 Ti) incompatible with PyTorch 2.12
#   SLURM_CPUS=4 \
#   SLURM_MEM_GB=24 \
#   SLURM_TIME_LIMIT=02:00:00 \
#   CHATTS_MODEL_PATH=/home/aviramom/models/chatts-8b \
#   UCR_DATA_PATH=/home/aviramom/ucr_data/Univariate_arff \  ← required for icl_ucr_*; home-staged so it's mounted on every node (/cs/azencot_fsas is CS-lab nodes only)
#   .venv/bin/streamlit run fmeval/app/main.py

# sync code changes to the cluster:
#   rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
#     --exclude='.git' --exclude='data/' --exclude='fmeval.egg-info' \
#     /Users/omeraviram/Projects/final_project/ \
#     aviramom@slurm.bgu.ac.il:~/fmeval_project/

# run tests:      .venv/bin/pytest
# lint/format:    .venv/bin/ruff check . && .venv/bin/ruff format .
# type check:     .venv/bin/mypy fmeval
```

---

## 7. Testing

- Unit-test the abstractions hardest: a new dataset subclass and a new model wrapper should be testable in isolation with no cluster.
- Test metric calculators against known inputs/expected outputs.
- The `LocalEvaluationPipeline` + `MockModel` already provide full end-to-end tests in `tests/evaluation/test_pipeline.py` — keep these passing when adding new models or metrics.
- When the Slurm execution layer is added, add tests that run config → mock execution → parse → metric → store using `MockRunner`.

---

## 8. Guardrails for Claude

- **Respect POC scope.** Don't add auth, multi-user sessions, real cluster orchestration, or extra modalities unless asked. The ARD notes auth is a *future* concern; the POC is single-user local.
- **Never require editing raw benchmark files.** Standardization happens in code (uniform dataset classes), not by mutating data.
- **When the requirements are ambiguous, check `ARD_Project.pdf` and ask** rather than inventing behavior.
- **Plan before large changes.** For anything touching the core abstractions or crossing layers, propose the interface/plan first, then implement.
- Prefer extending the existing layer structure over introducing new top-level modules.
