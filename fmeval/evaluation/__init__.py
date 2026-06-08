"""fmeval.evaluation — local synchronous evaluation pipeline."""

from fmeval.evaluation.pipeline import LocalEvaluationPipeline
from fmeval.evaluation.result import RunResult, SamplePrediction

__all__ = ["LocalEvaluationPipeline", "RunResult", "SamplePrediction"]
