"""JSONLMultimodalDataset — reference implementation and copy-paste template.

HOW TO ADD A NEW BENCHMARK
───────────────────────────
1. Copy this file to fmeval/core/datasets/<your_benchmark>.py.
2. Rename the class and update `name`.
3. Replace the JSONL parsing logic in __iter__ with your benchmark's native
   file format (CSV, Parquet, HuggingFace datasets, etc.).
4. Build `input_text` with <TS_N> tokens at the positions where the time series
   should appear in the prompt.
5. Put the matching numpy arrays in `input_ts` (same index order).
6. Set `output` to the ground-truth text string for this record.
7. Call self._validate_sample(sample) before yielding.
8. Register the class in fmeval/config/benchmark_registry.py — nothing else
   needs to change.

EXPECTED JSONL FORMAT (one JSON object per line)
─────────────────────────────────────────────────
{
  "context": "The electricity consumption (kWh) for the past week is shown below. <TS_0> Identify any anomalous days.",
  "ts": [[12.3, 14.1, 9.8, 25.4, 13.0, 11.7, 10.2]],
  "answer": "Day 4 shows an anomalous spike to 25.4 kWh."
}

Rules:
- "context"  : prompt string; use <TS_N> to mark where each series belongs.
- "ts"       : list of lists (one inner list per series, values are floats).
- "answer"   : ground-truth text response.

All three keys are required for every record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from fmeval.core.datasets.base_multimodal import MultimodalDataset
from fmeval.core.sample import Sample


class JSONLMultimodalDataset(MultimodalDataset):
    """Loads a JSONL file produced in the format described at the top of this
    module.  Each line becomes one Sample; loading is lazy (one line at a time).

    Parameters
    ----------
    data_path:
        Path to the .jsonl file.
    max_samples:
        If set, only the first N records are returned.  Useful for quick
        smoke-tests without loading the full benchmark.
    """

    def __init__(self, data_path: Path, max_samples: int | None = None) -> None:
        self._data_path = Path(data_path)
        self._max_samples = max_samples
        self._length: int | None = None  # cached on first __len__ call

    @property
    def name(self) -> str:
        return self._data_path.stem  # filename without extension as identifier

    def __len__(self) -> int:
        if self._length is None:
            count = 0
            with self._data_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        count += 1
            self._length = count
        if self._max_samples is not None:
            return min(self._length, self._max_samples)
        return self._length

    def __iter__(self) -> Iterator[Sample]:
        with self._data_path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if self._max_samples is not None and i >= self._max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample = self._parse_record(record, line_number=i + 1)
                self._validate_sample(sample)
                yield sample

    # ------------------------------------------------------------------
    # Internal helpers — override in subclasses if the format differs
    # ------------------------------------------------------------------

    def _parse_record(self, record: dict, line_number: int) -> Sample:
        """Convert one parsed JSON dict into a Sample.

        Override this method (not __iter__) when adapting this template to a
        benchmark that uses the same JSONL structure but different key names.
        """
        try:
            input_text: str = record["context"]
            raw_ts: list[list[float]] = record["ts"]
            answer: str = record["answer"]
        except KeyError as exc:
            raise ValueError(
                f"{self._data_path}:{line_number} — missing required key {exc}. "
                "Each record must have 'context', 'ts', and 'answer'."
            ) from exc

        input_ts = [np.array(series, dtype=np.float32) for series in raw_ts]

        return Sample(
            input_text=input_text,
            input_ts=input_ts,
            output=answer,
            metadata={"source_file": str(self._data_path), "line": line_number},
        )
