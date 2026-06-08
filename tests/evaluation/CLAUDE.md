# tests/evaluation/ — Integration tests for the evaluation pipeline

Tests in this directory exercise `fmeval/evaluation/` end-to-end: dataset →
pipeline → predictions → metrics → `RunResult`. All tests use an in-memory
dataset and `MockModel` — no network, no GPU.

Run: `.venv/bin/pytest tests/evaluation/ -v`

---

## Files

| File | What it tests |
|---|---|
| `test_pipeline.py` | `LocalEvaluationPipeline.run` — num\_samples, correctness, per-sample records, `RunResult.to_dataframe`, `breakdown_by`, `to_json`, `summary`, batching, `run_config` |

---

## Conventions

- Use the `InMemoryDataset` helper defined in `test_pipeline.py` to avoid HuggingFace calls.
- Add `pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")` to suppress
  expected sklearn warnings when predictions contain classes absent from a small subset.
- When a real model wrapper is integrated, add a separate test that runs the full
  pipeline with `max_samples=5` and the real model (marked `@pytest.mark.slow` so
  CI can skip it).
