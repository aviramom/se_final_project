"""LocalEvaluationPipeline — runs a model against a dataset synchronously.

This is the local / offline execution path (no Slurm, no async queue).
It covers:
  - Mock runs for development and CI
  - Small-scale experiments that fit on a single machine
  - Pre-computed log replay (feed predictions in directly)

For large-scale cluster runs the full EvaluationService + SlurmRunner path
will eventually replace this, but both paths produce a RunResult that can be
analysed identically.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from fmeval.core.datasets.base import Dataset
from fmeval.core.metrics.base import Metric
from fmeval.core.metrics.mcq_metrics import extract_letter
from fmeval.core.models.base import ModelWrapper
from fmeval.evaluation.result import RunResult, SamplePrediction

logger = logging.getLogger(__name__)


class LocalEvaluationPipeline:
    """Runs model inference and metric computation locally, sample by sample.

    Parameters
    ----------
    model:
        Any ModelWrapper implementation (real or mock).
    metric:
        The Metric to compute after collecting all predictions.
    batch_size:
        Number of formatted inputs passed to model.predict() at once.
        Has no effect on the mock model but matters for GPU-backed models.
    verbose:
        If True, log a progress line every `log_every` samples.
    log_every:
        Progress logging interval (samples).
    """

    def __init__(
        self,
        model: ModelWrapper,
        metric: Metric,
        batch_size: int = 32,
        verbose: bool = True,
        log_every: int = 100,
    ) -> None:
        self._model = model
        self._metric = metric
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_every = log_every

    def run(self, dataset: Dataset) -> RunResult:
        """Iterate through the dataset, collect predictions, compute metrics.

        Parameters
        ----------
        dataset:
            Any Dataset implementation. Iteration is lazy — the dataset is
            consumed once and not held in memory beyond the current batch.

        Returns
        -------
        RunResult with aggregate metrics and every SamplePrediction.
        """
        timestamp = datetime.now()
        run_config = {
            "model_name": self._model.model_name,
            "dataset_name": dataset.name,
            "input_mode": self._model.input_mode,
            "batch_size": self._batch_size,
            "metric": self._metric.name,
            "timestamp": timestamp.isoformat(),
        }

        all_samples_list = list(dataset)   # consume iterator once
        n_total = len(all_samples_list)
        if n_total == 0:
            raise ValueError("Dataset is empty — nothing to evaluate.")

        raw_predictions: list[str] = []
        raw_targets: list[str] = [s.output for s in all_samples_list]

        # Batch inference
        for batch_start in range(0, n_total, self._batch_size):
            batch = all_samples_list[batch_start: batch_start + self._batch_size]
            formatted = [self._model.format_input(s) for s in batch]
            preds = self._model.predict(formatted)
            raw_predictions.extend(preds)

            if self._verbose and (batch_start + len(batch)) % self._log_every == 0:
                done = batch_start + len(batch)
                logger.info("[%s] %d / %d samples processed", dataset.name, done, n_total)

        if self._verbose:
            logger.info("[%s] inference complete (%d samples)", dataset.name, n_total)

        # Aggregate metrics
        metrics = self._metric.compute(raw_predictions, raw_targets)

        # Per-sample records
        sample_predictions = self._build_sample_predictions(
            all_samples_list, raw_predictions, raw_targets
        )

        return RunResult(
            model_name=self._model.model_name,
            dataset_name=dataset.name,
            timestamp=timestamp,
            num_samples=n_total,
            metrics=metrics,
            sample_predictions=sample_predictions,
            run_config=run_config,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_sample_predictions(samples, raw_predictions, raw_targets):
        results = []
        for idx, (sample, pred, target) in enumerate(
            zip(samples, raw_predictions, raw_targets)
        ):
            pred_letter = extract_letter(pred)
            correct_letter = extract_letter(target)
            results.append(
                SamplePrediction(
                    sample_idx=idx,
                    raw_prediction=pred,
                    raw_target=target,
                    predicted_letter=pred_letter,
                    correct_letter=correct_letter,
                    is_correct=(pred_letter == correct_letter),
                    input_text=sample.input_text,
                    metadata=dict(sample.metadata),
                )
            )
        return results
