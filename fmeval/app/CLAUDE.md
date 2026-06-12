# fmeval/app/ — Layer 1: Presentation

Streamlit UI. Two tabs, nothing else. All business logic lives in the layers below;
this layer only calls `EvaluationService` and renders what comes back.

**Dependency rule:** `app/` may import from `fmeval.services` only. It must never
import from `core`, `execution`, `storage`, or `config` directly.

---

## Files

```
app/
  __init__.py         ✅ empty package marker
  CLAUDE.md
  main.py             ✅ entry point + @st.cache_resource service singleton
  config_page.py      ✅ render_config_page(service)
  dashboard_page.py   ✅ render_dashboard_page(service)
```

---

## Launch

```bash
# Local mode (MockRunner — no cluster needed, default)
.venv/bin/streamlit run fmeval/app/main.py

# Slurm mode (SlurmRunner — submits real jobs to slurm.bgu.ac.il)
FMEVAL_RUNNER=slurm .venv/bin/streamlit run fmeval/app/main.py

# Slurm mode with non-default resources
FMEVAL_RUNNER=slurm SLURM_PARTITION=rtx3090 SLURM_GPUS=1 \
  .venv/bin/streamlit run fmeval/app/main.py
```

## Runner selection (`main.py` — `_build_runner()`)

The runner is chosen at startup from the `FMEVAL_RUNNER` env var (default `"mock"`).
`"slurm"` instantiates `SlurmRunner` with `SlurmConfig` built from env vars:

| Env var | Default |
|---|---|
| `FMEVAL_RUNNER` | `mock` |
| `SLURM_HOST` | `slurm.bgu.ac.il` |
| `SLURM_USER` | `aviramom` |
| `SLURM_WORK_DIR` | `/home/aviramom/fmeval_jobs` |
| `SLURM_SSH_KEY` | `~/.ssh/id_ed25519` |
| `SLURM_PARTITION` | `cpu` |
| `SLURM_TIME_LIMIT` | `01:00:00` |
| `SLURM_GPUS` | `0` |
| `SLURM_CPUS` | `2` |
| `SLURM_MEM_GB` | `16` |
| `SLURM_PYTHON_BIN` | `/home/aviramom/fmeval_project/.venv/bin/python` |
| `SLURM_FMEVAL_DIR` | `/home/aviramom/fmeval_project` |
| `CHATTS_MODEL_PATH` | *(unset — falls back to HF Hub download)* |

`CHATTS_MODEL_PATH` is forwarded into the sbatch `env_setup_commands` automatically
by `_build_runner()`. Set it locally to a pre-downloaded path on the cluster so the
worker skips the 16 GB download on every job:

```bash
FMEVAL_RUNNER=slurm SLURM_PARTITION=<gpu-partition> SLURM_GPUS=1 \
SLURM_MEM_GB=32 SLURM_TIME_LIMIT=04:00:00 \
CHATTS_MODEL_PATH=/home/aviramom/models/chatts-8b \
.venv/bin/streamlit run fmeval/app/main.py
```

## Service singleton (`main.py`)

```python
@st.cache_resource
def get_service() -> EvaluationService:
    db_path = Path(__file__).parent.parent.parent / "data" / "results.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return EvaluationService(
        model_registry=build_default_model_registry(),
        benchmark_registry=build_default_benchmark_registry(),
        runner=_build_runner(),   # MockRunner or SlurmRunner per FMEVAL_RUNNER
        repository=ResultsRepository(db_path),
    )
```

`@st.cache_resource` creates one `EvaluationService` per server process. The
in-memory `_jobs` dict on the service persists across tab switches and rerenders
within a session. Results in SQLite persist across server restarts.

---

## ConfigPage (`config_page.py`) — `render_config_page(service)`

- **Model dropdown** populated from `service.list_models()` (display_name → name map)
- **Benchmark dropdown** populated from `service.list_benchmarks()`
- **Max samples slider** (10–200, default 50)
- **Experiment ID text input** — free-text label (max 64 chars); blank → service
  generates an auto-slug `"run-YYYYMMDD-HHMMSS"`. Resolved exp_id shown in success toast.
- **"Run Evaluation" button** → `service.run_evaluation(config)` in try/except;
  shows `st.success("Job {id} started · exp: {exp_id}")` or `st.error(msg)`
- **Recent Jobs table** at the bottom: calls `service.poll_jobs()` on each render
  (advances job states); includes `Exp ID` column.

---

## DashboardPage (`dashboard_page.py`) — `render_dashboard_page(service)`

- **Refresh button** → `service.poll_jobs()` + `st.rerun()`
- **Export CSV button** → `st.download_button(data=service.export_csv())`
- **Jobs table** — all jobs tracked this session (includes `Exp ID` column)
- **Filter panel** (`st.expander`) — multiselect for Experiment IDs, Models, Benchmarks;
  optional date-range inputs; "Clear Filters" button. Uses `st.session_state` keys so
  selections survive rerenders. Empty selection = no restriction on that dimension.
- **View toggle** — `"Individual runs"` or `"Grouped by Exp ID"`
  - *Individual*: grouped bar chart (x=Model, y=Accuracy, color=Benchmark, hover shows
    exp_id); results table with `on_select="rerun"` (Streamlit 1.35+) — clicking a row
    navigates to the **Run Detail** view
  - *Grouped*: bar chart with error bars (mean ± std on Accuracy); grouped comparison
    table showing `"0.8200 ± 0.0300"` strings for headline metrics with best-row
    highlight; n_samples and n_unparseable are summed across runs in each group
- **Run Detail view** — replaces the chart + table when a row is selected:
  - Header with job metadata (model, benchmark, exp_id, samples, timestamp, exec time)
  - Four metric cards: Accuracy, Balanced Acc, F1 Macro, F1 Weighted
  - Filter radio: All / Correct only / Incorrect only
  - Per-sample table: `#`, Question (truncated), Correct letter, Predicted letter,
    Result (✓/✗ with color), dynamic metadata columns (difficulty, category, etc.)
  - Expandable section with full question text and raw model output per sample
  - "← Back" button returns to the comparison table
  - If no sample data is stored (run predates this feature), shows a notice instead

Session state keys managed by the dashboard:
`dash_filter_exp_ids`, `dash_filter_models`, `dash_filter_benchmarks`,
`dash_date_from`, `dash_date_to`, `dash_view_mode`, `selected_job_id`

---

## Constraints

- No metric computation, no file I/O, no SQL.
- `ValueError` from `run_evaluation` (modality mismatch) → `st.error()`.
- Target responsiveness: page transitions and standard queries < 2 s.
