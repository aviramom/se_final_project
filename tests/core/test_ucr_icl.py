"""Tests for UCRICLDataset — synthetic ARFF, no cluster or real archive needed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fmeval.core.datasets.ucr_icl import UCRICLDataset
from fmeval.core.metrics.classification_metrics import ClassificationMetrics
from fmeval.core.models.random_label_model import RandomLabelModel
from fmeval.evaluation.pipeline import LocalEvaluationPipeline

_TS_TOKEN_RE = re.compile(r"<TS_(\d+)>")


def _write_arff(
    path: Path, rows: list[tuple[list[float], str]], labels: list[str]
) -> None:
    n_att = len(rows[0][0])
    lines = ["@relation Synthetic", ""]
    lines += [f"@attribute att{i + 1} numeric" for i in range(n_att)]
    lines.append(f"@attribute target {{{','.join(labels)}}}")
    lines += ["", "@data"]
    for values, label in rows:
        lines.append(",".join(f"{v:.4f}" for v in values) + f",{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def ucr_root(tmp_path: Path) -> Path:
    """A two-class synthetic UCR dataset folder named 'Synthetic'."""
    folder = tmp_path / "Synthetic"
    folder.mkdir()
    # 2 classes, 2 train examples each, length-3 series.
    _write_arff(
        folder / "Synthetic_TRAIN.arff",
        rows=[
            ([0.0, 0.1, 0.2], "1"),
            ([0.1, 0.2, 0.3], "1"),
            ([5.0, 5.1, 5.2], "2"),
            ([5.1, 5.2, 5.3], "2"),
        ],
        labels=["1", "2"],
    )
    _write_arff(
        folder / "Synthetic_TEST.arff",
        rows=[
            ([0.05, 0.15, 0.25], "1"),
            ([5.05, 5.15, 5.25], "2"),
            ([0.2, 0.3, 0.4], "1"),
        ],
        labels=["1", "2"],
    )
    return tmp_path


def _dataset(ucr_root: Path, **kw) -> UCRICLDataset:
    return UCRICLDataset("Synthetic", data_path=ucr_root, **kw)


def test_lazy_construction_reads_no_files(tmp_path: Path):
    # Pointing at a non-existent root must not raise until iteration.
    ds = UCRICLDataset("DoesNotExist", data_path=tmp_path / "missing")
    assert ds.name == "icl_ucr_DoesNotExist"
    assert ds.modality == "multimodal"
    assert isinstance(ds.metric, ClassificationMetrics)


def test_missing_file_raises_clear_error(tmp_path: Path):
    ds = UCRICLDataset("Nope", data_path=tmp_path)
    with pytest.raises(FileNotFoundError, match="UCR ARFF file not found"):
        list(ds)


def test_len_equals_test_size(ucr_root: Path):
    assert len(_dataset(ucr_root)) == 3


def test_support_set_size_is_k_times_classes(ucr_root: Path):
    ds = _dataset(ucr_root, k_shots=1, strategy="first")
    first = next(iter(ds))
    # input_ts = support (k * C = 1 * 2) + query = 3 arrays.
    assert len(first.input_ts) == 3
    assert first.metadata["num_classes"] == 2


def test_placeholders_match_input_ts_count(ucr_root: Path):
    for sample in _dataset(ucr_root):
        indices = {int(m) for m in _TS_TOKEN_RE.findall(sample.input_text)}
        # Tokens are 0..len(input_ts)-1 with no gaps.
        assert indices == set(range(len(sample.input_ts)))


def test_output_is_gold_label_and_options_present(ucr_root: Path):
    samples = list(_dataset(ucr_root, strategy="first"))
    assert {s.output for s in samples} <= {"1", "2"}
    assert "Return ONLY the label as one of: [1, 2]" in samples[0].input_text
    assert samples[0].metadata["options"] == ["1", "2"]


def test_normalization_is_in_range(ucr_root: Path):
    sample = next(iter(_dataset(ucr_root, strategy="first")))
    for arr in sample.input_ts:
        assert arr.min() >= -1.0 - 1e-6
        assert arr.max() <= 1.0 + 1e-6


def test_max_samples_subsamples(ucr_root: Path):
    ds = _dataset(ucr_root, max_samples=2)
    assert len(ds) == 2
    assert len(list(ds)) == 2


def test_k_shots_grows_support_set(ucr_root: Path):
    ds1 = _dataset(ucr_root, k_shots=1, strategy="first")
    ds2 = _dataset(ucr_root, k_shots=2, strategy="first")
    # 2 classes: k=1 → 2 support + 1 query = 3; k=2 → 4 support + 1 query = 5.
    assert len(next(iter(ds1)).input_ts) == 3
    assert len(next(iter(ds2)).input_ts) == 5


def test_registry_threads_few_shot_params(monkeypatch, ucr_root: Path):
    # The dataset_params dict from EvaluationConfig must reach the UCR factory.
    from fmeval.config.benchmark_registry import build_default_benchmark_registry

    monkeypatch.setenv("UCR_DATA_PATH", str(ucr_root))
    # Register a synthetic dataset name by reusing the GunPoint folder name.
    (ucr_root / "GunPoint").mkdir(exist_ok=True)
    for split in ("TRAIN", "TEST"):
        src = (ucr_root / "Synthetic" / f"Synthetic_{split}.arff").read_text()
        (ucr_root / "GunPoint" / f"GunPoint_{split}.arff").write_text(src)

    registry = build_default_benchmark_registry()
    ds = registry.get(
        "icl_ucr_GunPoint",
        max_samples=None,
        dataset_params={"num_shots": 2, "picking_strategy": "first", "random_seed": 3},
    )
    first = next(iter(ds))
    assert first.metadata["k_shots"] == 2
    assert first.metadata["strategy"] == "first"
    assert first.metadata["seed"] == 3


def test_evaluation_config_dataset_params():
    from fmeval.services.types import EvaluationConfig

    cfg = EvaluationConfig(
        model_name="m",
        benchmark_name="b",
        num_shots=4,
        picking_strategy="reversed",
        random_seed=9,
    )
    assert cfg.dataset_params() == {
        "num_shots": 4,
        "picking_strategy": "reversed",
        "random_seed": 9,
    }


def test_end_to_end_with_random_label_model(ucr_root: Path):
    # Full pipeline: dataset.metric (ClassificationMetrics) + a model that emits
    # valid labels. Verifies the metric-driven per-sample scoring path.
    ds = _dataset(ucr_root, strategy="first")
    result = LocalEvaluationPipeline(
        model=RandomLabelModel(seed=0), metric=ds.metric, verbose=False
    ).run(ds)
    assert result.num_samples == 3
    assert result.metrics["n_unparseable"] == 0.0
    # Each sample's correct_letter holds the gold class label, not an A-D letter.
    assert all(sp.correct_letter in {"1", "2"} for sp in result.sample_predictions)
