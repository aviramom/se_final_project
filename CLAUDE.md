# CLAUDE.md

This file gives Claude Code the context it needs to work on this project. Keep it short, high-signal, and up to date — when a convention or command changes, edit this file.

---

## 1. What this project is

**Foundation Model Evaluation Platform** — a centralized, lightweight pipeline for evaluating and comparing foundation models across modalities (currently **text** and **time-series**).

The problem it solves: today, running benchmark X on model Y requires hand-written adapter code, because every benchmark has a different structure and every model expects inputs formatted differently. This platform removes that glue work. A user picks pre-loaded models + benchmarks in a web UI, the system standardizes the data, runs inference (on a Slurm cluster, asynchronously), parses results, and shows a comparison dashboard.

It is a university **Software Engineering final project**. The graded deliverable is a working **proof of concept (POC)**, not a production system. Scope discipline matters more than feature breadth.

---

## 2. The two abstractions everything depends on

These are the heart of the architecture. Get the interfaces right before generating lots of code around them.

- **Uniform Dataset class** — wraps a benchmark *in its original file format* and exposes one standardized interface (e.g. `get_inputs()`, `get_targets()`, `modality`). It must NOT require the raw benchmark files to be manually altered or pre-processed (see Data Immutability constraint). Adding a new benchmark = writing one subclass, no changes elsewhere.
- **Model wrapper** — wraps a foundation model and handles modality-specific input formatting (prompt construction for text, windowing/formatting for time-series). Adding a new model = writing one wrapper, no changes elsewhere.

If a change to either of these forces edits in the UI, execution, or scoring layers, the abstraction is leaking — flag it rather than papering over it.

---

## 3. Architecture (layers, decoupled)

```
UI (Streamlit/Gradio)
   → Config registry (which models/benchmarks exist, their params)
      → Uniform Dataset classes  +  Model wrappers   (standardization)
         → Execution manager (generate sbatch script → submit → track status)
            → Compute cluster (Slurm)  /  Mock runner (POC)
               → Result parser → Metric calculator
                  → Results store (SQLite)
                     → Dashboard (charts, tables) + CSV export
```

Each layer talks to the next through a narrow interface. The UI must never reach into a dataset's raw format; the metric calculator must never know which model produced the predictions.

### Suggested layout (adjust as it grows)
```
fmeval/
  app/            # Streamlit/Gradio UI: config page + dashboard page
  core/
    datasets/     # uniform dataset base class + concrete benchmark subclasses
    models/       # model wrapper base class + concrete wrappers
    metrics/      # MSE/MAE (time-series), exact-match/F1 (text)
  execution/      # sbatch script generation, submission, status tracking, MOCK runner
  storage/        # SQLite schema + persistence helpers
  config/         # registries of available models & benchmarks
data/
  dummy/          # tiny datasets for local/dummy runs
  precomputed/    # canned cluster output logs for the offline demo
tests/
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
- **Metrics:** time-series → MSE, MAE. Text → exact-match, F1.
- **Stored per run:** model name, benchmark name, timestamp, execution time, computed metrics.

### POC / demo constraint (important)
Real runs can take hours and depend on cluster availability, so the system **must support an offline/simulated mode**: either a drastically reduced "dummy" dataset run locally, OR parsing pre-computed logs to visualize instantly. Build the execution layer behind an interface with a **MockRunner** and a **SlurmRunner**, switchable by config, so the demo never depends on a live queue.

The first thing to build is the **vertical slice** from the ARD's POC plan: one dropdown model + one time-series benchmark → "Run" generates a script → mock/dummy execution → parse → compute MSE → show in a table/chart.

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
# environment (suggest uv or venv + pip)
# install:        pip install -r requirements.txt
# run the app:    streamlit run fmeval/app/main.py
# run tests:      pytest
# lint/format:    ruff check . && ruff format .
# type check:     mypy fmeval
```

---

## 7. Testing

- Unit-test the abstractions hardest: a new dataset subclass and a new model wrapper should be testable in isolation with no cluster.
- Test metric calculators against known inputs/expected outputs.
- Test the parser against the pre-computed logs in `data/precomputed/`.
- The MockRunner makes the whole pipeline end-to-end testable without Slurm — there should be a test that runs config → mock execution → parse → metric → store.

---

## 8. Guardrails for Claude

- **Respect POC scope.** Don't add auth, multi-user sessions, real cluster orchestration, or extra modalities unless asked. The ARD notes auth is a *future* concern; the POC is single-user local.
- **Never require editing raw benchmark files.** Standardization happens in code (uniform dataset classes), not by mutating data.
- **When the requirements are ambiguous, check `ARD_Project.pdf` and ask** rather than inventing behavior.
- **Plan before large changes.** For anything touching the core abstractions or crossing layers, propose the interface/plan first, then implement.
- Prefer extending the existing layer structure over introducing new top-level modules.
