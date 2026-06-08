"""Evaluation result dataclasses.

RunResult        — the full output of one evaluation run.
SamplePrediction — per-sample record: raw strings, extracted letters, metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from fmeval.core.metrics.mcq_metrics import extract_letter


@dataclass
class SamplePrediction:
    """Record for a single sample's model output and ground truth.

    Storing raw strings plus the extracted letter means downstream analysis
    can always re-derive correctness without re-running the model.

    metadata carries all benchmark-level fields (difficulty, category, etc.)
    so results can be sliced without joining back to the dataset.
    """

    sample_idx: int
    raw_prediction: str          # exactly what model.predict() returned
    raw_target: str              # Sample.output as stored in the dataset
    predicted_letter: str | None # extracted A-D, or None if unparseable
    correct_letter: str          # extracted from raw_target (always present)
    is_correct: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = dict(self.metadata)
        return d


@dataclass
class RunResult:
    """Complete result of one evaluation run.

    Holds aggregate metrics plus every individual SamplePrediction so the
    caller can slice by any metadata dimension (difficulty, category, etc.)
    without re-running anything.
    """

    model_name: str
    dataset_name: str
    timestamp: datetime
    num_samples: int
    metrics: dict[str, float]              # aggregate metrics from Metric.compute
    sample_predictions: list[SamplePrediction]
    run_config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Flat DataFrame: one row per sample, columns = core fields + metadata.

        Useful for ad-hoc slicing, CSV export, and chart generation.
        All Sample.metadata fields are promoted to top-level columns.
        """
        rows = []
        for sp in self.sample_predictions:
            row: dict[str, Any] = {
                "sample_idx": sp.sample_idx,
                "raw_prediction": sp.raw_prediction,
                "raw_target": sp.raw_target,
                "predicted_letter": sp.predicted_letter,
                "correct_letter": sp.correct_letter,
                "is_correct": sp.is_correct,
            }
            row.update(sp.metadata)
            rows.append(row)
        return pd.DataFrame(rows)

    def breakdown_by(self, key: str) -> pd.DataFrame:
        """Compute accuracy and per-class F1 for each unique value of a metadata key.

        Parameters
        ----------
        key:
            A metadata field name, e.g. 'difficulty', 'category', 'num_options'.

        Returns
        -------
        pd.DataFrame with one row per unique value of `key`, columns:
            {key}, n_samples, n_correct, accuracy,
            f1_macro, precision_macro, recall_macro, n_unparseable
        """
        from fmeval.core.metrics.mcq_metrics import MCQMetrics

        metric = MCQMetrics()
        df = self.to_dataframe()

        if key not in df.columns:
            raise KeyError(
                f"breakdown_by: key '{key}' not found in sample metadata. "
                f"Available: {list(df.columns)}"
            )

        rows = []
        for value, group in df.groupby(key, sort=True):
            preds = group["raw_prediction"].tolist()
            targets = group["raw_target"].tolist()
            m = metric.compute(preds, targets)
            rows.append({
                key: value,
                "n_samples": int(m["n_samples"]),
                "n_correct": int(round(m["accuracy"] * m["n_samples"])),
                "accuracy": round(m["accuracy"], 4),
                "balanced_accuracy": round(m["balanced_accuracy"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "precision_macro": round(m["precision_macro"], 4),
                "recall_macro": round(m["recall_macro"], 4),
                "n_unparseable": int(m["n_unparseable"]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full result to a JSON string."""
        return json.dumps(self._to_serialisable(), indent=indent, default=str)

    def _to_serialisable(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "run_config": self.run_config,
            "sample_predictions": [sp.to_dict() for sp in self.sample_predictions],
        }

    def summary(self) -> str:
        """Human-readable one-page summary of aggregate metrics."""
        lines = [
            f"Model:    {self.model_name}",
            f"Dataset:  {self.dataset_name}",
            f"Samples:  {self.num_samples}",
            f"Time:     {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "── Aggregate metrics ─────────────────────────",
        ]
        for k, v in self.metrics.items():
            if not k.startswith(("f1_", "precision_", "recall_", "support_")) or \
               k in ("f1_macro", "f1_weighted", "precision_macro",
                     "precision_weighted", "recall_macro", "recall_weighted"):
                lines.append(f"  {k:<26} {v:.4f}" if isinstance(v, float) else
                             f"  {k:<26} {int(v)}")
        lines += [
            "",
            "── Per-class breakdown ───────────────────────",
        ]
        seen_letters = sorted({
            k.split("_")[1] for k in self.metrics
            if k.startswith("support_") and len(k) == 9
        })
        for lbl in seen_letters:
            sup = int(self.metrics.get(f"support_{lbl}", 0))
            f1 = self.metrics.get(f"f1_{lbl}", 0.0)
            pr = self.metrics.get(f"precision_{lbl}", 0.0)
            rc = self.metrics.get(f"recall_{lbl}", 0.0)
            lines.append(
                f"  {lbl}  support={sup:4d}  "
                f"f1={f1:.3f}  prec={pr:.3f}  rec={rc:.3f}"
            )
        return "\n".join(lines)
