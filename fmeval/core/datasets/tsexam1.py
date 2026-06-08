"""AutonLab/TimeSeriesExam1 dataset wrapper.

Each sample is a multiple-choice question about one or two time series.
The dataset has 746 questions covering pattern recognition, similarity analysis,
anomaly detection, and more.

Time series layout (mutually exclusive per row):
    ts  column  → single time series  → mapped to <TS_0>
    ts1 + ts2   → two time series     → mapped to <TS_0> and <TS_1>

Prompt structure emitted by this class (separate/canonical form):
    Time series [1]: <TS_0>
    [Time series 2: <TS_1>]               ← only for two-TS rows

    Question: {question}

    Options:
    A) {option_0}
    B) {option_1}
    ...

    {format_hint}

Output stored in Sample.output:
    "{letter}) {answer_text}"  e.g. "B) Decrease"
    Matches what the model is instructed to produce via format_hint.

HuggingFace dataset: https://huggingface.co/datasets/AutonLab/TimeSeriesExam1
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from datasets import load_dataset

from fmeval.core.datasets.base_multimodal import MultimodalDataset
from fmeval.core.sample import Sample

_OPTION_LETTERS = "ABCD"
_HF_DATASET_ID = "AutonLab/TimeSeriesExam1"


class TimeSeriesExam1Dataset(MultimodalDataset):
    """Wraps the AutonLab/TimeSeriesExam1 HuggingFace benchmark.

    Parameters
    ----------
    split:
        HuggingFace split to load — currently only "test" exists.
    max_samples:
        If set, only the first N records are yielded.  Useful for smoke-tests.
    """

    def __init__(
        self,
        split: str = "test",
        max_samples: int | None = None,
    ) -> None:
        self._split = split
        self._max_samples = max_samples
        self._hf_dataset = load_dataset(_HF_DATASET_ID, split=split)

    @property
    def name(self) -> str:
        return "tsexam1"

    def __len__(self) -> int:
        total = len(self._hf_dataset)
        if self._max_samples is not None:
            return min(total, self._max_samples)
        return total

    def __iter__(self) -> Iterator[Sample]:
        for i, row in enumerate(self._hf_dataset):
            if self._max_samples is not None and i >= self._max_samples:
                break
            sample = self._row_to_sample(row)
            self._validate_sample(sample)
            yield sample

    # ------------------------------------------------------------------

    def _row_to_sample(self, row: dict) -> Sample:
        input_ts, ts_header = self._extract_ts(row)

        options_block = "\n".join(
            f"{_OPTION_LETTERS[i]}) {opt}"
            for i, opt in enumerate(row["options"])
        )

        input_text = (
            f"{ts_header}\n"
            f"Question: {row['question']}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{row['format_hint']}"
        )

        correct_idx = row["options"].index(row["answer"])
        output = f"{_OPTION_LETTERS[correct_idx]}) {row['answer']}"

        metadata = {
            "id": row["id"],
            "tid": row["tid"],
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "relevant_concepts": row["relevant_concepts"],
            "answer_letter": _OPTION_LETTERS[correct_idx],
            "answer_text": row["answer"],
            "num_options": len(row["options"]),
        }

        return Sample(
            input_text=input_text,
            input_ts=input_ts,
            output=output,
            metadata=metadata,
        )

    @staticmethod
    def _extract_ts(row: dict) -> tuple[list[np.ndarray], str]:
        """Return (input_ts list, ts header string with <TS_N> tokens)."""
        if row["ts"] is not None:
            arr = np.array(row["ts"], dtype=np.float32)
            return [arr], "Time series: <TS_0>\n"
        else:
            arr1 = np.array(row["ts1"], dtype=np.float32)
            arr2 = np.array(row["ts2"], dtype=np.float32)
            return [arr1, arr2], "Time series 1: <TS_0>\nTime series 2: <TS_1>\n"
