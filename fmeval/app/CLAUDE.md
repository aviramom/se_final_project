# fmeval/app/ — Layer 1: Presentation

Streamlit/Gradio UI. Two pages, nothing else. All business logic lives in the layers
below; this layer only calls `EvaluationService` and renders what comes back.

**Dependency rule:** `app/` may import from `fmeval.services` only. It must never
import from `core`, `execution`, `storage`, or `config` directly.

---

## Files

```
app/
  CLAUDE.md
  main.py          ← entry point: `streamlit run fmeval/app/main.py`
  config_page.py   ← ConfigPage
  dashboard_page.py← DashboardPage
```

---

## ConfigPage (`config_page.py`)

Responsibilities:
- On load, call `EvaluationService.list_models()` and `list_benchmarks()` to populate
  the dropdowns. Never hard-code model/benchmark names here.
- Collect: selected model(s), selected benchmark, runner mode (Mock / Slurm /
  Precomputed), any optional params (forecast horizon, context length, etc.).
- On "Run": call `EvaluationService.run_evaluation(config)`, receive a `job_id`,
  and surface it to the user. Return immediately — do not poll or block.

## DashboardPage (`dashboard_page.py`)

Responsibilities:
- Call `EvaluationService.get_dashboard_data()` and render the result as charts
  and comparison tables.
- Call `EvaluationService.poll_jobs()` to show live job statuses (queued / running /
  completed / failed). A simple periodic refresh (e.g. `st.rerun`) is fine for the POC.
- Provide a "Export CSV" button that calls `EvaluationService.export_csv()`.

## main.py

Wires the two pages into a navigation structure (sidebar or tabs). Initialises
`EvaluationService` once and passes it to both pages.

---

## Constraints

- No metric computation, no file parsing, no SQL queries.
- All user-visible error messages come from the service layer as structured data,
  not from caught exceptions in UI callbacks.
- Target responsiveness: page transitions and standard queries < 2 s.
