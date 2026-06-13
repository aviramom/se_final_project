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
  main.py             ✅ entry point + @st.cache_resource service singleton + inject_css()
  config_page.py      ✅ render_config_page(service)
  dashboard_page.py   ✅ render_dashboard_page(service) — Runs/Compare/Trends/Jobs sub-tabs
  ui_components.py    ✅ shared CSS, status_badge(), render_qa_card(), render_diff_card()
```

The Streamlit theme lives in `.streamlit/config.toml` at the repo root.

---

## Launch

```bash
# Local mode (MockRunner — mock models only, no cluster needed)
.venv/bin/streamlit run fmeval/app/main.py
# or: ./run.sh mock

# ChatTS-8B on Slurm GPU node
./run.sh chatts

# Qwen3-VL-8B-Instruct on Slurm GPU node
./run.sh qwen

# Both GPU models available in the same session
./run.sh all
```

`run.sh` at the repo root is a convenience wrapper that sets all required env vars.
See `README.md` for the full env var reference and manual launch commands.

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
| `QWEN_VL_MODEL_PATH` | *(unset — falls back to HF Hub download)* |

Both model weight path vars are forwarded into the sbatch `env_setup_commands`
automatically by `_build_runner()` so the cluster worker sees them at runtime.

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
- **Two-step benchmark picker** — benchmarks are grouped by `BenchmarkInfo.group`
  ("Benchmark group" selectbox), then a second "Dataset" selectbox lists the
  `short_name`s within that group (hidden when a group has a single benchmark).
  This keeps the 95 UCR datasets out of one giant flat dropdown.
- **Max samples slider** (10–200, default 50)
- **Few-shot controls** (k / support selection / random seed) — shown only when the
  selected benchmark has `supports_few_shot=True` (UCR ICL). Flow into
  `EvaluationConfig.num_shots/picking_strategy/random_seed`; other benchmarks use
  defaults. The service forwards them as `dataset_params` to the dataset factory
  (and `SlurmRunner` passes them as worker CLI flags so cluster runs match the UI).
- **Experiment ID text input** — free-text label (max 64 chars); blank → service
  generates an auto-slug `"run-YYYYMMDD-HHMMSS"`. Resolved exp_id shown in success toast.
- **"Run Evaluation" button** → `service.run_evaluation(config)` in try/except;
  shows `st.success("Job {id} started · exp: {exp_id}")` or `st.error(msg)`
- **Recent Jobs table** at the bottom: calls `service.poll_jobs()` on each render
  (advances job states); includes `Exp ID` column.

---

## DashboardPage (`dashboard_page.py`) — `render_dashboard_page(service)`

Four sub-tabs (`st.tabs`):

- **📊 Runs** — filter panel (exp_ids / models / benchmarks / date range);
  **chart-metric selectbox** fed by `service.list_metric_keys()` (y-range pinned
  to [0,1] only when all values look like rates); Individual / Grouped toggle;
  Export CSV. Results table uses `selection_mode="multi-row"`: one selected row
  → "View details" + "Delete (1)", several → bulk "Delete (N)". Deletes go
  through an `@st.dialog` confirmation; after deletion the table widget key is
  rotated via `results_table_nonce` (selections are index-based and would
  otherwise point at the wrong rows).
- **Run Detail** (drill-in from Runs) — metadata line + notes banner; **Edit
  expander** (exp_id text_input + notes text_area → `service.update_run`);
  Delete-run button; metric cards; **Breakdown by Category** (metadata-key
  selectbox → `get_category_breakdown` bar chart + table); per-sample table;
  Q/A card viewer (`render_qa_card`, paginated 25/page).
- **⚖️ Compare** — two run selectboxes (B limited to A's benchmark);
  `service.compare_runs` → 4 agreement metric cards, color-coded per-question
  table, "Disagreements only" toggle with side-by-side `render_diff_card`s.
- **📈 Trends** — metric + benchmark selectboxes; `px.line` of metric vs
  timestamp (color=model, line_dash=benchmark).
- **⚙️ Jobs** — persisted jobs from `service.poll_jobs()` (restored across
  restarts), status-count metric row, table with colored status cells.

Session state keys managed by the dashboard:
`dash_filter_exp_ids`, `dash_filter_models`, `dash_filter_benchmarks`,
`dash_date_from`, `dash_date_to`, `dash_view_mode`, `selected_job_id`,
`results_table_nonce`

---

## Constraints

- No metric computation, no file I/O, no SQL.
- `ValueError` from `run_evaluation` (modality mismatch) → `st.error()`.
- Target responsiveness: page transitions and standard queries < 2 s.
