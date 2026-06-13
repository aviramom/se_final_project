#!/usr/bin/env python
"""Cluster evaluation worker.

Runs a single fmeval evaluation job on the cluster and writes the result as JSON.
This file is uploaded by SlurmRunner to {remote_job_dir}/worker.py and invoked
by the sbatch script with the arguments below.

Usage:
    python worker.py --dataset <name> --model <name> [--max-samples N] --output <path>
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("cluster_worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="fmeval cluster evaluation worker")
    parser.add_argument("--dataset", required=True, help="Benchmark registry key")
    parser.add_argument("--model", required=True, help="Model registry key")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Sample limit passed to the dataset factory. Omit to use all samples.",
    )
    # Optional few-shot ICL options (used by benchmarks that support them).
    parser.add_argument("--num-shots", type=int, default=None, dest="num_shots")
    parser.add_argument(
        "--picking-strategy", type=str, default=None, dest="picking_strategy"
    )
    parser.add_argument("--random-seed", type=int, default=None, dest="random_seed")
    parser.add_argument("--output", required=True, help="Path to write result.json")
    args = parser.parse_args()

    logger.info(
        "Worker started: dataset=%s  model=%s  max_samples=%s",
        args.dataset,
        args.model,
        args.max_samples,
    )

    # Only forward params that were actually supplied, so the dataset factory
    # falls back to its own defaults otherwise.
    dataset_params = {
        key: getattr(args, key)
        for key in ("num_shots", "picking_strategy", "random_seed")
        if getattr(args, key) is not None
    }

    from fmeval.config.benchmark_registry import build_default_benchmark_registry
    from fmeval.config.model_registry import build_default_model_registry
    from fmeval.evaluation.pipeline import LocalEvaluationPipeline

    benchmark_registry = build_default_benchmark_registry()
    model_registry = build_default_model_registry()

    dataset = benchmark_registry.get(
        args.dataset, max_samples=args.max_samples, dataset_params=dataset_params
    )
    model = model_registry.get(args.model)

    # The dataset chooses its own metric (MCQ letters, class labels, etc.).
    pipeline = LocalEvaluationPipeline(model=model, metric=dataset.metric, verbose=True)
    logger.info("Running pipeline...")
    result = pipeline.run(dataset)

    logger.info(
        "Pipeline complete: %d samples  accuracy=%.4f",
        result.num_samples,
        result.metrics.get("accuracy", 0.0),
    )

    with open(args.output, "w") as f:
        f.write(result.to_json())

    logger.info("Result written to %s", args.output)
    # Sentinel line parsed by SlurmRunner to confirm the file was actually written.
    print(f"FMEVAL_RESULT_WRITTEN:{args.output}", flush=True)


if __name__ == "__main__":
    main()
