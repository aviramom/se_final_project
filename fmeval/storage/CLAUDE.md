# fmeval/storage/ — Layer 5 (part 1): Persistence

Thin SQLite wrapper. Stores completed evaluation results and serves them to the
dashboard. No business logic here — just reads and writes.

**Dependency rule:** `storage/` must not import from `app/` or `services/`.
It imports `SamplePrediction` from `evaluation/result.py` (needed at runtime
to construct objects in `get_sample_predictions`) and may import `core/` types.

---

## Files

```
storage/
  __init__.py     ✅ exports EvaluationResult, ResultsRepository
  CLAUDE.md
  models.py       ✅ EvaluationResult dataclass
  repository.py   ✅ ResultsRepository (SQLite)
```

No separate `migrations.py` — schema creation (`CREATE TABLE IF NOT EXISTS`) plus
`ALTER TABLE ADD COLUMN` migrations are called inside `ResultsRepository.__init__`.

---

## EvaluationResult (`models.py`)

```python
@dataclass
class EvaluationResult:
    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    metrics: dict[str, float]   # e.g. {"accuracy": 0.85, "f1_macro": 0.82, …}
    timestamp: datetime
    execution_time_seconds: float
    exp_id: str = ""    # experiment label; "" means auto-slug was used
    max_samples: int = 0
```

`exp_id` and `max_samples` have defaults so existing call-sites don't break.

---

## ResultsRepository (`repository.py`)

```python
class ResultsRepository:
    def __init__(self, db_path: Path) -> None:
        # mkdir parents + CREATE TABLE IF NOT EXISTS + idempotent ALTER TABLE migrations

    def save(self, result: EvaluationResult) -> None:
        # INSERT OR REPLACE — upserts on job_id PK

    def query(
        self,
        model_name: str | None = None,
        benchmark_name: str | None = None,
        exp_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[EvaluationResult]: ...

    def get_by_job_id(self, job_id: str) -> EvaluationResult | None: ...

    def list_exp_ids(self) -> list[str]:
        # SELECT DISTINCT exp_id WHERE exp_id != '' ORDER BY exp_id

    def export_csv(self, results: list[EvaluationResult]) -> str:
        # Returns CSV string (no file written) — pass to st.download_button
        # Columns: job_id, exp_id, model_name, benchmark_name, modality,
        #          <metric keys>, timestamp, execution_time_seconds, max_samples

    def save_sample_predictions(self, job_id: str, predictions: list[SamplePrediction]) -> None:
        # Bulk-inserts per-sample records (executemany). Called by _finalize_job.

    def get_sample_predictions(self, job_id: str) -> list[SamplePrediction]:
        # Returns all SamplePrediction records for a job, ordered by sample_idx.
        # Returns [] if the job has no stored predictions (pre-feature runs).
```

`db_path.parent.mkdir(parents=True, exist_ok=True)` is called in `__init__`, so
the `data/` directory is created automatically on first run.

---

## Schema

```sql
CREATE TABLE IF NOT EXISTS evaluation_results (
    job_id                  TEXT PRIMARY KEY,
    model_name              TEXT NOT NULL,
    benchmark_name          TEXT NOT NULL,
    modality                TEXT NOT NULL,
    metrics                 TEXT NOT NULL,              -- JSON
    timestamp               TEXT NOT NULL,              -- ISO-8601
    execution_time_seconds  REAL NOT NULL,
    exp_id                  TEXT NOT NULL DEFAULT '',
    max_samples             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sample_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL,
    sample_idx       INTEGER NOT NULL,
    input_text       TEXT NOT NULL DEFAULT '',
    raw_prediction   TEXT NOT NULL,
    raw_target       TEXT NOT NULL,
    predicted_letter TEXT,                              -- NULL if unparseable
    correct_letter   TEXT NOT NULL,
    is_correct       INTEGER NOT NULL,                  -- 0 or 1
    metadata_json    TEXT NOT NULL DEFAULT '{}'         -- JSON dict
);

CREATE INDEX IF NOT EXISTS idx_sp_job_id ON sample_predictions(job_id);
```

`metrics` and `metadata_json` are serialized with `json.dumps` / `json.loads`.
`timestamp` uses `.isoformat()` / `datetime.fromisoformat()`.

`exp_id` and `max_samples` were added to `evaluation_results` via `ALTER TABLE ADD COLUMN`
migrations in `__init__` (wrapped in `try/except sqlite3.OperationalError`) — these run
on every open and are a no-op on already-migrated DBs. The `sample_predictions` table is
created via `CREATE TABLE IF NOT EXISTS` so it is also a no-op on existing DBs.

---

## Constraints

- `sqlite3.connect(check_same_thread=False)` — required because Streamlit runs
  callbacks on multiple threads.
- SQLite file path injected at construction so tests can pass a temp path.
- No caching layer for the POC — query directly on every request.
