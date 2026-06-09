# fmeval/evaluation/ — Local evaluation pipeline

Synchronous, cluster-free evaluation path. Runs a `ModelWrapper` against a `Dataset`
locally, collects predictions, computes metrics, and returns a rich `RunResult`.

This covers the "offline / mock" use case — demos, CI, small experiments — without
requiring Slurm. The full async path (Layer 4 `execution/` + `EvaluationService`)
will sit alongside this once implemented; both paths produce a `RunResult` that can
be analysed identically.

**Status: implemented.** Both files are written and tested
(`tests/evaluation/test_pipeline.py`, 32 tests).

---

## Files

```
evaluation/
  CLAUDE.md
  __init__.py     ← exports: LocalEvaluationPipeline, RunResult, SamplePrediction
  pipeline.py     ← ✅ LocalEvaluationPipeline
  result.py       ← ✅ RunResult, SamplePrediction
```

---

## LocalEvaluationPipeline (`pipeline.py`)

```python
pipeline = LocalEvaluationPipeline(
    model      = MockModel("A"),
    metric     = MCQMetrics(),
    batch_size = 32,
    verbose    = True,
)
result = pipeline.run(dataset)
```

Iterates the dataset once, batches calls to `model.format_input` + `model.predict`,
then calls `metric.compute`. Returns a `RunResult`.

---

## RunResult (`result.py`)

The complete output of one evaluation run. Key methods:

| Method | Returns |
|---|---|
| `result.summary()` | Human-readable text table of all aggregate metrics + per-class breakdown |
| `result.to_dataframe()` | `pd.DataFrame` — one row per sample; `Sample.metadata` fields promoted to columns |
| `result.breakdown_by(key)` | `pd.DataFrame` — recomputes all metrics per unique value of any metadata key (e.g. `"difficulty"`, `"category"`, `"num_options"`) |
| `result.to_json()` | Full serialisation including every `SamplePrediction` |

`RunResult.metrics` holds the full `dict[str, float]` from `MCQMetrics.compute`:
accuracy, balanced\_accuracy, f1\_macro/weighted, precision\_macro/weighted,
recall\_macro/weighted, n\_samples, n\_unparseable, and per-class f1/precision/recall/support.

`SamplePrediction` stores per-sample: `input_text` (prompt template from the
dataset, with `<TS_N>` placeholders), `raw_prediction`, `raw_target`,
`predicted_letter`, `correct_letter`, `is_correct`, `metadata` (all benchmark
fields so slicing never requires re-joining to the dataset). `input_text`
defaults to `""` for backwards compatibility but is always populated by the
pipeline.

---

## Usage example

```python
from fmeval.core.datasets.tsexam1 import TimeSeriesExam1Dataset
from fmeval.core.models.mock_model import MockModel
from fmeval.core.metrics.mcq_metrics import MCQMetrics
from fmeval.evaluation import LocalEvaluationPipeline

result = LocalEvaluationPipeline(MockModel(), MCQMetrics()).run(
    TimeSeriesExam1Dataset(max_samples=200)
)

print(result.summary())
result.breakdown_by("difficulty")
result.breakdown_by("category")
result.to_dataframe().to_csv("results.csv", index=False)
```

See `notebooks/tsexam1_demo.ipynb` for a fully executed walkthrough.
