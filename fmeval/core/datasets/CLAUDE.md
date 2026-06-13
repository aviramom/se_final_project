# fmeval/core/datasets/ — Dataset abstractions

`Dataset` is the ABC that wraps a benchmark in its original file format and exposes a
uniform interface. Adding a new benchmark means adding one subclass here — nothing else
in the system changes.

**Status: implemented.** `base.py`, `base_multimodal.py`, `formatters.py`, and
`template.py` are all written and tested (`tests/core/`).

---

## Files

```
datasets/
  CLAUDE.md
  __init__.py          ← public re-exports
  base.py              ← Dataset ABC
  base_multimodal.py   ← MultimodalDataset helper base (fixes modality, adds validation)
  formatters.py        ← SampleFormatter: canonical ↔ combined conversion
  template.py          ← JSONLMultimodalDataset — copy-paste template for new benchmarks
  tsexam1.py           ← ✅ AutonLab/TimeSeriesExam1 (HuggingFace, 746 MCQ questions)
  ucr_icl.py           ← ✅ UCRICLDataset — UCR ARFF few-shot ICL classification (lazy, UCR_DATA_PATH)
  <benchmark>.py       ← one file per concrete benchmark
```

---

## Dataset ABC (`base.py`)

```python
class Dataset(ABC):
    @property
    @abstractmethod
    def modality(self) -> Literal["text", "time_series", "multimodal"]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...         # registry key + stored in results

    @property
    def metric(self) -> Metric: ...    # evaluation method; default MCQMetrics, override per benchmark

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...   # lazy preferred

    @abstractmethod
    def __len__(self) -> int: ...
```

`metric` has a concrete default (`MCQMetrics`) so existing MCQ benchmarks need no
change. A benchmark with a different answer format overrides it — e.g.
`UCRICLDataset` returns `ClassificationMetrics`. The runners read `dataset.metric`
and the pipeline scores per-sample via `metric.label_predictions`, so no layer
hardcodes a metric.

---

## MultimodalDataset helper (`base_multimodal.py`)

All current benchmarks subclass this instead of `Dataset` directly. It:
- Fixes `modality` to `"multimodal"` — subclasses don't repeat it.
- Provides `_validate_sample(sample)` — call it before `yield` in `__iter__` to
  catch placeholder/array mismatches early.

```python
class MultimodalDataset(Dataset):
    @property
    def modality(self) -> Literal[...]:
        return "multimodal"

    def _validate_sample(self, sample: Sample) -> None:
        # Raises ValueError if:
        #   - input_ts is empty
        #   - any <TS_N> token references an out-of-range index
        ...
```

---

## SampleFormatter (`formatters.py`)

Converts between the canonical (separate) form and the combined form a model may need.
Datasets always emit separate form; `ModelWrapper.format_input` calls the formatter.

```python
class SampleFormatter:
    @staticmethod
    def to_combined(sample: Sample,
                    serializer: TSSerializer = DefaultTSSerializer()) -> Sample:
        """Replace <TS_N> tokens with serialized arrays; clears input_ts."""

    @staticmethod
    def to_separate(sample: Sample) -> Sample:
        """Identity — canonical form is already separate."""
```

`DefaultTSSerializer` renders arrays as `[v0, v1, ...]` (4 sig-fig). Implement the
`TSSerializer` protocol to swap in a different representation without changing any other
code.

**Never mutates the input Sample** — always returns a new one.

---

## Reference implementation (`template.py`)

`JSONLMultimodalDataset` reads a JSONL file where each line is:
```json
{"context": "...<TS_0>...", "ts": [[1.2, 3.4, ...]], "answer": "..."}
```

**This is the file to copy when adding a new benchmark.** Override `_parse_record` if
the benchmark uses different key names; override `__iter__` for non-JSONL formats.

---

## How to add a new benchmark

1. Create `fmeval/core/datasets/<benchmark>.py`.
2. Subclass `MultimodalDataset`.
3. In `__init__`: accept `data_path: Path` + benchmark-specific params; load
   index/metadata only (lazy loading in `__iter__`).
4. In `__iter__`:
   - Read one record at a time — never mutate the raw file.
   - Build `input_text` with `<TS_N>` tokens at the correct positions.
   - Put raw numpy arrays in `input_ts` in matching order.
   - Set `output` to the ground-truth text string.
   - Call `self._validate_sample(sample)` before `yield`.
5. If the answer format isn't A–D MCQ, override the `metric` property to return
   the right `Metric` (e.g. `ClassificationMetrics` for free class labels).
6. Register the class in `fmeval/config/benchmark_registry.py` — zero other changes.

`ucr_icl.py` is the reference for a non-MCQ benchmark: lazy ARFF loading from a
`UCR_DATA_PATH` root, train-only `[-1, 1]` normalization, per-class support-set
selection, `<TS_N>` placeholders for the support series + query, and a `metric`
override to `ClassificationMetrics`.

---

## Data Immutability constraint

Standardization is done entirely in `__iter__`. If a concrete subclass writes,
copies, or transforms the raw file on disk, that is a bug. The raw benchmark files
in `data/` are read-only from the perspective of this codebase.
