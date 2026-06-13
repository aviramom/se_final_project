# tests/core/ — Unit tests for Layer 3 (Domain)

Tests in this directory cover `fmeval/core/` only — no cluster, no database, no UI.
All tests run offline with no external dependencies beyond numpy.

Run: `.venv/bin/pytest tests/core/ -v`

---

## Files

| File | What it tests |
|---|---|
| `test_sample.py` | `Sample` construction, field defaults, metadata isolation |
| `test_formatters.py` | `SampleFormatter.to_combined` token substitution, out-of-range raises, `to_separate` identity, custom serializer |
| `test_jsonl_dataset.py` | `JSONLMultimodalDataset` loading, `len()`, `max_samples`, modality, placeholder validation, malformed input |
| `test_tsexam1.py` | `TimeSeriesExam1Dataset` — single/two-TS layouts, option letter assignment, metadata, `max_samples` (HF dataset mocked) |
| `test_mcq_metrics.py` | `extract_letter` patterns, `MCQMetrics.compute` — all metric keys, per-class breakdown, balanced accuracy, unparseable handling |
| `test_classification_metrics.py` | `extract_label` priority rules + longest-first (`1` vs `10`), `ClassificationMetrics.compute` over a free label set, balanced accuracy, unparseable handling |
| `test_ucr_icl.py` | `UCRICLDataset` (synthetic ARFF) — lazy load, support-set size by k, `<TS_N>`/`input_ts` alignment, normalization, max_samples subsample, registry few-shot param threading, end-to-end via `RandomLabelModel` |
| `test_mock_model.py` | `MockModel` — fixed answer, `format_input` inlines TS, `predict` batch length |

---

## Adding tests for a new benchmark

When you add `fmeval/core/datasets/<benchmark>.py`, add
`tests/core/test_<benchmark>.py` covering at minimum:

1. `len()` matches the expected record count.
2. `__iter__` yields `Sample` objects with correct field types.
3. `_validate_sample` is exercised (at least one passing and one invalid case).
4. `modality == "multimodal"`.
5. Mock the data source (HuggingFace `load_dataset` or file I/O) so the test runs offline.

## Adding tests for a new model wrapper

When you add `fmeval/core/models/<model>.py`, add
`tests/core/test_<model>.py` covering at minimum:

1. `format_input` returns the expected type for a known `Sample`.
2. `input_mode` is declared and is either `"combined"` or `"separate"`.
3. `predict` returns a `list[str]` of the same length as the input batch.
4. Mock the HuggingFace model load so the test runs without GPU or network.
