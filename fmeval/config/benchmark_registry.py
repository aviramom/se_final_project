from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from fmeval.core.datasets.base import Dataset
from fmeval.core.datasets.tsexam1 import TimeSeriesExam1Dataset
from fmeval.core.datasets.ucr_icl import UCRICLDataset

# Default archive root on the cluster; overridden by the UCR_DATA_PATH env var.
# Resolution is lazy (read at factory call time) so a missing path locally does
# not break registry construction or the modality compatibility check.
_DEFAULT_UCR_ROOT = "/cs/azencot_fsas/multimodal_ts/datasets/Univariate_arff"

# The 94 feasible UCR datasets (fixed-length, fit a k=1 prompt, integer labels),
# grouped by domain as in UCR_ICL_BENCHMARK.md. The category is shown in the UI.
_UCR_DATASETS: list[tuple[str, str]] = [
    # Image / Shape
    *[
        (n, "Image/Shape")
        for n in (
            "ArrowHead",
            "BeetleFly",
            "BirdChicken",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "Fish",
            "Herring",
            "MedicalImages",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxOutlineCorrect",
            "MiddlePhalanxTW",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxTW",
            "SwedishLeaf",
            "Symbols",
            "Yoga",
            "Crop",
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
        )
    ],
    # Sensor / Device
    *[
        (n, "Sensor/Device")
        for n in (
            "Car",
            "ChlorineConcentration",
            "Computers",
            "Earthquakes",
            "ElectricDevices",
            "FordA",
            "FordB",
            "ItalyPowerDemand",
            "LargeKitchenAppliances",
            "Lightning2",
            "Lightning7",
            "MoteStrain",
            "Plane",
            "RefrigerationDevices",
            "ScreenType",
            "SmallKitchenAppliances",
            "SonyAIBORobotSurface1",
            "SonyAIBORobotSurface2",
            "StarLightCurves",
            "Trace",
            "Wafer",
            "BME",
            "Chinatown",
            "DodgerLoopDay",
            "DodgerLoopGame",
            "DodgerLoopWeekend",
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "HouseTwenty",
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
            "InsectWingbeatSound",
            "MelbournePedestrian",
            "PowerCons",
            "SemgHandGenderCh2",
            "SmoothSubspace",
        )
    ],
    # Motion / HAR
    *[
        (n, "Motion/HAR")
        for n in (
            "CricketX",
            "CricketY",
            "CricketZ",
            "GunPoint",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Haptics",
            "InlineSkate",
            "PickupGestureWiimoteZ",
            "ShakeGestureWiimoteZ",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "UWaveGestureLibraryAll",
            "Worms",
            "WormsTwoClass",
        )
    ],
    # ECG / Medical
    *[
        (n, "ECG/Medical")
        for n in (
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "TwoLeadECG",
        )
    ],
    # Spectrographic / Chemometrics
    *[
        (n, "Spectro/Chemometrics")
        for n in (
            "Beef",
            "Coffee",
            "EthanolLevel",
            "Ham",
            "Meat",
            "OliveOil",
            "Strawberry",
            "Wine",
        )
    ],
    # Simulated / Synthetic
    *[
        (n, "Simulated/Synthetic")
        for n in (
            "CBF",
            "SyntheticControl",
            "TwoPatterns",
            "UMD",
        )
    ],
]


def _ucr_root() -> str:
    """Archive root for UCR datasets, from UCR_DATA_PATH or the cluster default."""
    return os.environ.get("UCR_DATA_PATH", _DEFAULT_UCR_ROOT)


# A factory takes (max_samples, dataset_params) and returns a Dataset.
# dataset_params carries optional, benchmark-specific construction hints from
# EvaluationConfig (e.g. num_shots / picking_strategy / random_seed for ICL);
# factories ignore keys they don't use.
DatasetFactory = Callable[[int | None, dict], Dataset]


def _ucr_factory(dataset_name: str) -> DatasetFactory:
    """Build a registry factory bound to one UCR dataset (avoids loop late-binding)."""

    def factory(max_samples: int | None, params: dict) -> Dataset:
        return UCRICLDataset(
            dataset_name,
            data_path=_ucr_root(),
            max_samples=max_samples,
            k_shots=params.get("num_shots", 1),
            strategy=params.get("picking_strategy", "random"),
            seed=params.get("random_seed", 0),
        )

    return factory


@dataclass
class BenchmarkInfo:
    """UI-facing metadata for a registered benchmark."""

    name: str  # registry key used in EvaluationConfig
    display_name: str  # shown in dropdowns
    modality: str
    group: str = ""  # picker group; many UCR datasets share one group
    short_name: str = ""  # label within a group (e.g. the dataset name)
    supports_few_shot: bool = False  # True → UI exposes k / strategy / seed controls

    def __post_init__(self) -> None:
        # Fallbacks so single-benchmark families need no extra wiring: each
        # becomes its own group keyed by display_name.
        if not self.group:
            self.group = self.display_name
        if not self.short_name:
            self.short_name = self.display_name


class BenchmarkRegistry:
    """Maps benchmark names to factories that produce Dataset instances.

    The factory receives max_samples plus a dataset_params dict so a dataset is
    constructed with the right size limit and any benchmark-specific options
    without the registry needing to store config state.
    """

    def __init__(self) -> None:
        self._registry: dict[str, tuple[BenchmarkInfo, DatasetFactory]] = {}

    def register(self, info: BenchmarkInfo, factory: DatasetFactory) -> None:
        self._registry[info.name] = (info, factory)

    def list(self) -> list[BenchmarkInfo]:
        return [info for info, _ in self._registry.values()]

    def get(
        self,
        name: str,
        max_samples: int | None = None,
        dataset_params: dict | None = None,
    ) -> Dataset:
        if name not in self._registry:
            available = list(self._registry)
            raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
        _, factory = self._registry[name]
        return factory(max_samples, dataset_params or {})


def build_default_benchmark_registry() -> BenchmarkRegistry:
    """Return a registry pre-populated with the benchmarks available for the POC."""
    registry = BenchmarkRegistry()
    registry.register(
        BenchmarkInfo(
            name="tsexam1",
            display_name="TimeSeriesExam1 (HuggingFace)",
            modality="multimodal",
        ),
        factory=lambda max_s, _params: TimeSeriesExam1Dataset(max_samples=max_s),
    )

    # All feasible UCR in-context-learning classification datasets. Each is its
    # own entry because EvaluationConfig selects a benchmark by name only; the
    # UI groups them by category and exposes the few-shot controls (k/strategy/
    # seed) for them.
    for ds, category in _UCR_DATASETS:
        registry.register(
            BenchmarkInfo(
                name=f"icl_ucr_{ds}",
                display_name=f"UCR ICL: {ds} ({category})",
                modality="multimodal",
                group=f"UCR ICL — {category}",
                short_name=ds,
                supports_few_shot=True,
            ),
            factory=_ucr_factory(ds),
        )
    return registry
