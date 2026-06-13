"""UCR in-context-learning (ICL) time-series classification benchmark.

Given ``k`` labeled example series per class (the *support set*, drawn from the
training split) and one unlabeled *query* series from the test split, the model
must predict the query's class label — all inside a single prompt.  This is pure
in-context learning: no fine-tuning, the labeled examples live in the prompt.

Data source: the UCR Time Series Classification Archive (2018), one folder per
dataset under a ``Univariate_arff/`` root, each holding ``{Name}_TRAIN.arff`` and
``{Name}_TEST.arff``.  The archive is large and lives on the cluster, so the root
is taken from the ``UCR_DATA_PATH`` env var at registry-factory time and loading
is lazy: ``__init__`` only stores parameters, so the service can construct the
dataset locally for the modality compatibility check without the files present.

Prompt structure emitted (canonical / separate form, with <TS_N> placeholders):

    Time Series Classification.
    {description}

    --- EXAMPLES ---

    Example 1 Time Series: <TS_0>
    Label: {label_0}
    ...

    --- TARGET ---
    New Time Series: <TS_{k*C}>
    Return ONLY the label as one of: [{opts}] without any explanation

input_ts holds the support arrays followed by the query array, index-matched to
the tokens; output is the gold integer label as a string.  The model wrapper's
input_mode decides how the arrays are rendered (serialized inline for LLMs, passed
raw to a TS encoder, or plotted for a VLM), so this one prompt works everywhere.

Scored with ClassificationMetrics (balanced accuracy over the class set), not the
A–D MCQMetrics — the answer is a free class label, not a multiple-choice letter.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.io import arff

from fmeval.core.datasets.base_multimodal import MultimodalDataset
from fmeval.core.metrics.base import Metric
from fmeval.core.metrics.classification_metrics import ClassificationMetrics
from fmeval.core.sample import Sample

_VALID_STRATEGIES = ("first", "random", "reversed")

# Optional human-written domain hints injected into the prompt.  Empty string is
# fine — the benchmark works without descriptions; this just helps the model.
UCR_DESCRIPTIONS: dict[str, str] = {
    "GunPoint": (
        "Motion tracking data of an actor's hand. The task is to classify whether "
        "the actor is drawing a gun from a hip holster or simply pointing a finger."
    ),
    "ECG200": (
        "Electrocardiogram recordings of a single heartbeat. The task is to "
        "classify each beat as normal or as a myocardial infarction."
    ),
    "Coffee": (
        "Spectrographic analysis of coffee. The task is to distinguish Arabica "
        "from Robusta beans."
    ),
}


class UCRICLDataset(MultimodalDataset):
    """One UCR dataset wrapped as a few-shot ICL classification benchmark.

    Parameters
    ----------
    dataset_name:
        Folder name in the archive, e.g. ``"GunPoint"`` (case-sensitive).
    data_path:
        Path to the ``Univariate_arff/`` root that contains ``{dataset_name}/``.
    max_samples:
        If set, a seeded random subsample of this many test queries is used
        (matches the reference protocol of 250 queries per run).
    k_shots:
        Support examples per class (default 1).
    strategy:
        Support-set selection: ``"first"``, ``"random"`` (default), or
        ``"reversed"``.
    seed:
        Seed for the ``random`` strategy and the test subsample, so the same
        support set and query subset are used across models.
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str | Path,
        max_samples: int | None = None,
        k_shots: int = 1,
        strategy: str = "random",
        seed: int = 0,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose one of {_VALID_STRATEGIES}."
            )
        self._dataset_name = dataset_name
        self._data_path = Path(data_path)
        self._max_samples = max_samples
        self._k_shots = k_shots
        self._strategy = strategy
        self._seed = seed
        self._desc = UCR_DESCRIPTIONS.get(dataset_name, "")
        # Lazily-populated cache so __len__ and __iter__ don't reload the ARFF.
        self._cache: dict | None = None

    @property
    def name(self) -> str:
        return f"icl_ucr_{self._dataset_name}"

    @property
    def metric(self) -> Metric:
        return ClassificationMetrics()

    def __len__(self) -> int:
        return len(self._prepared()["query_indices"])

    def __iter__(self) -> Iterator[Sample]:
        prepared = self._prepared()
        support = prepared["support"]  # list of (np.ndarray, label_str)
        options = prepared["options"]  # sorted unique support labels
        test_data = prepared["test_data"]  # (N, T) normalized
        test_labels = prepared["test_labels"]  # list[str]

        input_prompt = self._build_prompt(support, options)
        support_arrays = [arr for arr, _ in support]

        for query_idx in prepared["query_indices"]:
            query_arr = test_data[query_idx]
            input_ts = [*support_arrays, query_arr]
            sample = Sample(
                input_text=input_prompt,
                input_ts=input_ts,
                output=test_labels[query_idx],
                metadata={
                    "dataset_name": self._dataset_name,
                    "num_classes": len(options),
                    "options": options,
                    "k_shots": self._k_shots,
                    "strategy": self._strategy,
                    "seed": self._seed,
                    "query_index": int(query_idx),
                    "gold_label": test_labels[query_idx],
                },
            )
            self._validate_sample(sample)
            yield sample

    # ------------------------------------------------------------------
    # Loading & preparation (lazy, cached)
    # ------------------------------------------------------------------

    def _prepared(self) -> dict:
        """Load the ARFF splits and build the support set + query subset once."""
        if self._cache is not None:
            return self._cache

        folder = self._data_path / self._dataset_name
        train_data, train_labels = self._load_split(folder, "TRAIN")
        test_data, test_labels = self._load_split(folder, "TEST")

        # Normalize to [-1, 1] using train-only statistics (no test leakage).
        valid = train_data[~np.isnan(train_data)]
        min_val = float(valid.min())
        max_val = float(valid.max())
        train_data = self._normalize(train_data, min_val, max_val)
        test_data = self._normalize(test_data, min_val, max_val)

        support = self._select_support(train_data, train_labels)
        options = sorted({label for _, label in support})

        query_indices = list(range(len(test_labels)))
        if self._max_samples is not None and self._max_samples < len(query_indices):
            rng = random.Random(self._seed)
            query_indices = sorted(rng.sample(query_indices, self._max_samples))

        self._cache = {
            "support": support,
            "options": options,
            "test_data": test_data,
            "test_labels": test_labels,
            "query_indices": query_indices,
        }
        return self._cache

    def _load_split(self, folder: Path, split: str) -> tuple[np.ndarray, list[str]]:
        """Return (data (N, T) float32, labels list[str]) for one ARFF split.

        Raises a clear error if the file is missing — the runner surfaces this as
        a failed-job status rather than crashing the app.
        """
        path = folder / f"{self._dataset_name}_{split}.arff"
        if not path.is_file():
            raise FileNotFoundError(
                f"UCR ARFF file not found: {path}. Set UCR_DATA_PATH to the "
                f"'Univariate_arff' root containing '{self._dataset_name}/'."
            )
        raw, _meta = arff.loadarff(str(path))
        df = pd.DataFrame(raw)
        # Last column is the class label (scipy yields bytes for nominal attrs).
        labels = [self._decode_label(v) for v in df.iloc[:, -1].tolist()]
        data = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        return data, labels

    @staticmethod
    def _decode_label(value) -> str:
        """Normalize a raw ARFF label to its canonical integer-string form."""
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        # UCR labels are integers stored variously as 1, 1.0, b'1' — canonicalize.
        return str(int(float(value)))

    @staticmethod
    def _normalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        span = max_val - min_val
        if span == 0:
            return np.zeros_like(data)
        return 2.0 * (data - min_val) / span - 1.0

    def _select_support(
        self, train_data: np.ndarray, train_labels: list[str]
    ) -> list[tuple[np.ndarray, str]]:
        """Pick k examples per class from the training split per the strategy."""
        if self._strategy == "random":
            rng = random.Random(self._seed)
            by_label: dict[str, list[int]] = defaultdict(list)
            for i, label in enumerate(train_labels):
                by_label[label].append(i)
            support: list[tuple[np.ndarray, str]] = []
            for label in sorted(by_label):
                indices = by_label[label]
                chosen = rng.sample(indices, min(self._k_shots, len(indices)))
                for idx in chosen:
                    support.append((train_data[idx], label))
            return support

        # "first" / "reversed": first k per class in dataset order.
        counts: dict[str, int] = defaultdict(int)
        support = []
        for i, label in enumerate(train_labels):
            if counts[label] < self._k_shots:
                support.append((train_data[i], label))
                counts[label] += 1
        if self._strategy == "reversed":
            support = support[::-1]
        return support

    def _build_prompt(
        self, support: list[tuple[np.ndarray, str]], options: list[str]
    ) -> str:
        """Assemble the ICL prompt with <TS_N> placeholders for each series."""
        lines = ["Time Series Classification."]
        if self._desc:
            lines.append(self._desc)
        lines.append("\n--- EXAMPLES ---\n")
        for i, (_, label) in enumerate(support):
            lines.append(f"Example {i + 1} Time Series: <TS_{i}>")
            lines.append(f"Label: {label}\n")
        lines.append("--- TARGET ---")
        lines.append(f"New Time Series: <TS_{len(support)}>")
        opts = ", ".join(options)
        lines.append(
            f"Return ONLY the label as one of: [{opts}] without any explanation"
        )
        return "\n".join(lines)
