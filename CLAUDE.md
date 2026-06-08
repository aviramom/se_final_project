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

Two pages, both in Streamlit/Gradio. `ConfigPage` populates dropdowns from the registries, collects evaluation parameters, and on "Run" calls one method on `EvaluationService`. `DashboardPage` shows job statuses, renders comparison charts/tables, and triggers CSV export.

Hard rule: no business logic in the UI. It never reads a raw benchmark file or computes a metric — it only calls `EvaluationService` and renders the result.

### Layer 2 — Orchestration

`EvaluationService` is the single class the UI talks to. It owns the workflow and does none of the actual work. Public surface: `list_models()`, `list_benchmarks()` (delegates to registries), `run_evaluation(config) -> job_id`, `poll_jobs()`, `get_dashboard_data()`, `export_csv()`. Everything below is invoked by the service; nothing below ever calls back up.

### Layer 3 — Domain (core abstractions)

These three ABCs and one dataclass are the heart of the system. Get their interfaces right before generating code around them. A change here that forces edits in any other layer means the abstraction is leaking — flag it rather than papering over it.

`Dataset` (ABC) wraps a benchmark in its **original file format** and exposes a `modality` property plus an iterator yielding `Sample` objects. Adding a benchmark = one new subclass, zero changes elsewhere. The Data Immutability constraint lives here — standardization happens in code, never by editing raw files. All current benchmarks subclass `MultimodalDataset` which fixes `modality="multimodal"` and enforces placeholder validation.

`Sample` (dataclass) — **implemented** — the standardized unit between `Dataset` and `ModelWrapper`: `input_text` (NL prompt with `<TS_N>` placeholders), `input_ts` (list of raw numpy arrays), `output` (ground-truth text string), `metadata` (dict).

`SampleFormatter` (in `core/datasets/formatters.py`) — **implemented** — converts between the canonical **separate** form (placeholders + raw arrays) that all datasets emit, and the **combined** form (TS serialized inline in text) that LLMs consume. `ModelWrapper` declares `input_mode` and calls the formatter in `format_input`.

`ModelWrapper` (ABC) declares `supported_modalities`, `input_mode` (`"combined"` or `"separate"`), `format_input(sample)`, and `predict(...) -> list[str]`. All models output text. `predict` is the code that runs on the cluster; in mock mode the same code runs locally.

`Metric` (ABC) — **implemented** — exposes `applicable_modalities` and `compute(predictions, targets) -> dict[str, float]`. `MCQMetrics` is the concrete implementation: returns accuracy, balanced\_accuracy, F1, precision, recall (macro + weighted), n\_unparseable, and per-class breakdowns in a single call.

**Modality as the matching key.** A `Dataset` declares its modality (`"multimodal"` for all current datasets); that single tag determines which `ModelWrapper`s are compatible and which `Metric`s apply. No `if benchmark == X` branching anywhere — modality drives the routing.

### Layer 4 — Execution

`ScriptGenerator` builds the `sbatch` script (env vars, paths, run command). `Runner` (ABC) provides `submit(job) -> handle` and `get_status(handle)` — the most important POC seam. Three implementations: `SlurmRunner` (shells out to `sbatch`/`squeue`), `MockRunner` (runs a truncated dummy eval locally), `PrecomputedRunner` (returns a handle pointing at a canned log). Swap the runner; nothing else changes. `ResultParser` reads raw output logs into predictions + targets ready for a `Metric`. `JobStatus` (enum): `queued / running / completed / failed`. `EvaluationJob` (dataclass) carries job state across the async boundary.

### Layer 5 — Data & Configuration

`ResultsRepository` is a thin SQLite wrapper storing `EvaluationResult` records (model name, benchmark name, modality, metrics dict, timestamp, execution time). `ModelRegistry` and `BenchmarkRegistry` are loaded at startup and answer two questions: "what's available?" for the UI, and "give me object X" for the service.

### Key interactions

**Run flow (write path).** `ConfigPage → EvaluationService.run_evaluation(config)`. The service resolves names through the registries into a `Dataset` and a `ModelWrapper`, checks `model.supports(dataset.modality)`, asks `ScriptGenerator` for a script, hands it to the configured `Runner`, persists an `EvaluationJob` as `queued`, and returns immediately — the UI never blocks. Later, `poll_jobs()` asks the `Runner` for status; on completion it calls `ResultParser` → selects the right `Metric` by modality → `compute` → `ResultsRepository.save`.

**Dashboard (read path).** `DashboardPage → get_dashboard_data() → ResultsRepository.query → charts`. The read and write paths are fully independent, which is what satisfies the "configuration phase vs. asynchronous results-viewing phase" requirement.

### Suggested layout

```
fmeval/
  __init__.py     # package root
  app/            # Streamlit/Gradio UI: ConfigPage + DashboardPage
  core/
    sample.py              # ✅ Sample dataclass (input_text, input_ts, output)
    datasets/
      base.py              # ✅ Dataset ABC
      base_multimodal.py   # ✅ MultimodalDataset (modality fixed, validation)
      formatters.py        # ✅ SampleFormatter (combined ↔ separate)
      template.py          # ✅ JSONLMultimodalDataset (copy-paste template)
      <benchmark>.py       # one file per concrete benchmark
    models/
      base.py              # ✅ ModelWrapper ABC
      mock_model.py        # ✅ MockModel (fixed-answer baseline)
      __init__.py          # ✅
    metrics/
      base.py              # ✅ Metric ABC
      mcq_metrics.py       # ✅ MCQMetrics + extract_letter()
      __init__.py          # ✅
  evaluation/              # ✅ Local synchronous evaluation pipeline
    pipeline.py            # ✅ LocalEvaluationPipeline
    result.py              # ✅ RunResult, SamplePrediction
    __init__.py            # ✅
  execution/      # ScriptGenerator, Runner ABC, SlurmRunner, MockRunner,
                  # PrecomputedRunner, ResultParser, EvaluationJob, JobStatus
  storage/        # ResultsRepository (SQLite), EvaluationResult
  config/         # ModelRegistry, BenchmarkRegistry
  services/
    __init__.py
    types.py              # ✅ EvaluationConfig, DashboardData
    evaluation_service.py # EvaluationService (stub)
data/
  dummy/          # tiny JSONL datasets for local / dummy runs
  precomputed/    # canned cluster output logs for the offline demo
notebooks/
  tsexam1_demo.ipynb       # ✅ Executed demo: sample inspection + evaluation walkthrough
  build_notebook.py        # script that regenerates the notebook
tests/
  core/
    test_sample.py         # ✅
    test_formatters.py     # ✅
    test_jsonl_dataset.py  # ✅
    test_tsexam1.py        # ✅
    test_mcq_metrics.py    # ✅
    test_mock_model.py     # ✅
  evaluation/
    test_pipeline.py       # ✅ LocalEvaluationPipeline + RunResult (30 tests)
ARD_Project.pdf   # ground-truth requirements — consult when unsure
CLAUDE.md
```

---

## 4. Tech stack & hard constraints

- **Language:** Python. Everything stays Python-native so it plugs into the existing research codebase.
- **UI:** Python-native only — **Streamlit or Gradio** (pick one and commit to it). No JS frameworks.
- **Compute:** real execution targets a **Slurm**-managed GPU cluster via generated `sbatch` scripts.
- **Storage:** local DB / structured files — **SQLite** is the natural fit for the POC.
- **Async by nature:** evaluations are NOT real-time. Jobs go into a queue; the UI must clearly separate the *configuration* phase from the *results-viewing* phase. The app must stay responsive (screen transitions / standard queries < 2s) while jobs run.
- **Metrics:** multimodal MCQ tasks → `MCQMetrics` (accuracy, balanced\_accuracy, F1, precision, recall — all macro + weighted + per-class).
- **Stored per run:** model name, benchmark name, timestamp, execution time, computed metrics.

### POC / demo constraint (important)
Real runs can take hours and depend on cluster availability, so the system **must support an offline/simulated mode**: either a drastically reduced "dummy" dataset run locally, OR parsing pre-computed logs to visualize instantly. Build the execution layer behind an interface with a **MockRunner** and a **SlurmRunner**, switchable by config, so the demo never depends on a live queue.

The first thing to build is the **vertical slice** from the ARD's POC plan: one dropdown model + one multimodal benchmark → "Run" generates a script → mock/dummy execution → parse → compute ExactMatch / F1 → show in a table/chart.

---

## 5. Conventions

- Follow PEP 8. Use type hints on all public functions and class interfaces.
- Define the dataset and model interfaces as abstract base classes (`abc.ABC`); concrete benchmarks/models subclass them.
- Keep functions small and single-purpose. No business logic inside UI callbacks — UI calls into `core`/`execution`, never the reverse.
- Handle external failures gracefully: cluster timeouts, node crashes, missing logs, malformed benchmark files. Catch, log, and surface a clear failure **status** in the UI — never crash the app.
- Docstrings on every public class/function explaining the *why*, not just the *what*.

---

## 6. Commands

> Fill these in as the project takes shape; keep them accurate so Claude can verify its own work.

```bash
# environment: .venv at repo root (Python 3.14)
# install:        pip install -r requirements.txt
# run the app:    streamlit run fmeval/app/main.py
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
