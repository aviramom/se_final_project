# fmeval/storage/ — Layer 5 (part 1): Persistence

Thin SQLite wrapper. Stores completed evaluation results and serves them to the
dashboard. No business logic here — just reads and writes.

**Dependency rule:** `storage/` must not import from `app/`, `services/`, or
`execution/`. It may import `core/` types for type hints only.

---

## Files

```
storage/
  CLAUDE.md
  repository.py        ← ResultsRepository
  models.py            ← EvaluationResult dataclass + schema definition
  migrations.py        ← schema creation / upgrade helpers
```

---

## EvaluationResult (`models.py`)

```python
@dataclass
class EvaluationResult:
    job_id: str
    model_name: str
    benchmark_name: str
    modality: str
    metrics: dict[str, float]   # e.g. {"mse": 0.42, "mae": 0.31}
    timestamp: datetime
    execution_time_seconds: float
```

This is the record stored per completed run.

---

## ResultsRepository (`repository.py`)

```python
class ResultsRepository:
    def __init__(self, db_path: Path): ...

    def save(self, result: EvaluationResult) -> None: ...

    def query(
        self,
        model_name: str | None = None,
        benchmark_name: str | None = None,
        since: datetime | None = None,
    ) -> list[EvaluationResult]: ...

    def get_by_job_id(self, job_id: str) -> EvaluationResult | None: ...
```

All queries return plain `EvaluationResult` dataclass instances. The repository
owns the SQLite connection; callers never touch SQL directly.

---

## Schema

One table: `evaluation_results`. Columns map 1-to-1 with `EvaluationResult` fields.
`metrics` is stored as a JSON string and deserialized on read.

`migrations.py` creates the table if it does not exist. Run it at application startup
via `ResultsRepository.__init__`.

---

## Constraints

- SQLite file path is injected at construction time (easy to swap in tests).
- No caching layer for the POC — query directly on every request.
- All queries must complete in < 2 s on the expected result volume (hundreds of rows).
