# UCR In-Context Learning (ICL) Benchmark

## Overview

This benchmark evaluates language models on **few-shot time series classification** using datasets from the [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). The task: given `k` labeled example time series (the "support set") and an unlabeled query time series, predict the class of the query — all within a single LLM prompt.

**Task format:** Pure in-context learning (ICL). No fine-tuning. The model receives the support examples and the query as plain text (numeric arrays), and must respond with exactly one class label.

**Scope:** 94 UCR datasets that are (a) fixed-length, (b) fit within a model's context window at k=1, and (c) have numeric labels that parse cleanly.

---

## Data Source

**Download URL:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

Download the UCR Time Series Classification Archive (2018). After extraction, the directory structure is:

```
Univariate_arff/
├── GunPoint/
│   ├── GunPoint_TRAIN.arff
│   └── GunPoint_TEST.arff
├── Beef/
│   ├── Beef_TRAIN.arff
│   └── Beef_TEST.arff
└── ... (128 datasets total)
```

Each `.arff` file contains one time series per row. All columns except the last are the numeric time-series values; the last column is the integer class label.

---

## Dataset Format

### ARFF Structure

```
@relation GunPoint

@attribute att1 numeric
@attribute att2 numeric
...
@attribute attT numeric
@attribute classAttribute {1,2}

@data
-0.64,-0.64,...,1.00,1
-0.32,-0.45,...,0.89,2
...
```

### UCRDataset

The dataset class loads an ARFF file, normalizes the time series, and yields `(ts_tensor, label_tensor)` pairs.

**Normalization:** Min-max to `[-1, 1]` using statistics computed from the **training split only** (to avoid test leakage):

```
ts_normalized = 2 * (ts - min_val) / (max_val - min_val) - 1
```

where `min_val` and `max_val` are the global min/max of the training data (ignoring NaN values).

**Fields:**
- `data`: `torch.FloatTensor` of shape `(N, T)` — unnormalized raw values
- `labels`: `torch.LongTensor` of shape `(N,)` — integer class labels
- `min_val`, `max_val`: float scalars for normalization
- `desc`: string domain description from `ucr_descriptions/{DatasetName}/description.txt`

---

## Prompt Format

Each evaluation sample is a single text prompt. The model must respond with only a class label.

### Template

```
Time Series Classification.
{description}

--- EXAMPLES ---

Example 1 Time Series: [{v1:.4f}, {v2:.4f}, ..., {vT:.4f}]
Label: {class_a}

Example 2 Time Series: [{v1:.4f}, {v2:.4f}, ..., {vT:.4f}]
Label: {class_b}

--- TARGET ---
New Time Series: [{v1:.4f}, {v2:.4f}, ..., {vT:.4f}]
Return ONLY the label as one of: [{class_a}, {class_b}] without any explanation
```

### Rules

- `{description}`: Domain description string. Empty string if `--use_label_desc 0` and no description file exists.
- Time series values are formatted to **4 decimal places**, comma-separated, wrapped in `[...]`.
- `options` list in the final line is derived from the **unique labels in the support set**, not the full label set.
- The prompt is wrapped in a chat template via `tokenizer.apply_chat_template()` before inference.
- Qwen3/3.6 thinking mode is explicitly disabled (`enable_thinking=False`).

### Concrete Example (GunPoint, k=1)

```
Time Series Classification.
Motion tracking data of an actor's hand. The task is to classify whether the actor is drawing a gun from a hip holster or simply pointing a finger.

--- EXAMPLES ---

Example 1 Time Series: [-0.4832, -0.4832, -0.4671, ..., 0.9812]
Label: 1

Example 2 Time Series: [0.1204, 0.0932, 0.0814, ..., -0.2341]
Label: 2

--- TARGET ---
New Time Series: [-0.1023, -0.0987, -0.0811, ..., 0.4512]
Return ONLY the label as one of: [1, 2] without any explanation
```

---

## Support Set Selection Strategies

The support set is `k` examples per class drawn from the **training split**.

| Strategy | Description |
|----------|-------------|
| `first` | Take the first `k` examples per class in dataset order |
| `random` | Uniformly sample `k` examples per class (seeded by `--random_seed`) |
| `medoid` | Select the `k` examples per class with minimum total L2 distance to all other class members |
| `medoid_dtw` | Same as `medoid` but using DTW distance |
| `reversed` | Same as `first` but in reversed order |

**Default in production experiments:** `random`, `k=1`.

The random strategy uses `random.Random(seed)` for reproducibility — the same seed produces the same support set across all models.

---

## Label Extraction

After the model generates a response, the predicted label is extracted by matching against the known class labels from the prompt. Rules are tried in priority order:

1. **Exact match:** `response == label`
2. **Class statement:** `"The class is {label}"` or `"The class is <{label}>"` in response
3. **Predicted Label:** regex `Predicted\s*Label\s*:\s*["'<[]?\s*{label}(?!\d)` (case-insensitive)
4. **Predicted:** regex `Predicted\s*:\s*["'<[]?\s*{label}(?!\d)` (case-insensitive)
5. **Label colon:** regex `(?<!\w)label\s*:\s*["'<[]?\s*{label}(?!\d)` (case-insensitive)
6. **Label is:** regex `(?:correct\s+)?label\s+is\s+["'<[]?\s*{label}(?!\d)` (case-insensitive)

If no rule matches any label, the predicted label is `"INVALID_PREDICTION"`.

The options are parsed from the prompt by extracting the list in `"Return ONLY the label as one of: [...]"`.

---

## Metrics

All metrics are computed via `sklearn.metrics`.

| Metric | Description |
|--------|-------------|
| `balanced_accuracy` | **Primary metric.** Average per-class recall. Handles class imbalance. Equivalent to macro recall in multiclass settings. |
| `f1_macro` | F1 score, macro-averaged over classes |
| `f1_weighted` | F1 score, weighted by class frequency |
| `precision_macro` | Precision, macro-averaged |
| `precision_weighted` | Precision, weighted |
| `recall_macro` | Recall, macro-averaged |
| `recall_weighted` | Recall, weighted |

`INVALID_PREDICTION` responses count as wrong for all metrics.

---

## The 94 Feasible Datasets

These are the datasets used in the primary experiment `random_k1_comparison_full`. Excluded datasets are: variable-length series, datasets where a k=1 prompt exceeds the model context window, and datasets with non-integer labels that cause parsing issues.

Task IDs use the prefix `icl_ucr_` followed by the dataset name (case-sensitive, matching the folder name in the archive).

### Image / Shape (27)

```
icl_ucr_ArrowHead           icl_ucr_BeetleFly            icl_ucr_BirdChicken
icl_ucr_DiatomSizeReduction icl_ucr_DistalPhalanxOutlineAgeGroup
icl_ucr_DistalPhalanxOutlineCorrect  icl_ucr_DistalPhalanxTW
icl_ucr_FaceAll             icl_ucr_FaceFour             icl_ucr_FacesUCR
icl_ucr_Fish                icl_ucr_Herring               icl_ucr_MedicalImages
icl_ucr_MiddlePhalanxOutlineAgeGroup icl_ucr_MiddlePhalanxOutlineCorrect
icl_ucr_MiddlePhalanxTW     icl_ucr_OSULeaf              icl_ucr_PhalangesOutlinesCorrect
icl_ucr_ProximalPhalanxOutlineAgeGroup icl_ucr_ProximalPhalanxOutlineCorrect
icl_ucr_ProximalPhalanxTW   icl_ucr_SwedishLeaf          icl_ucr_Symbols
icl_ucr_Yoga                icl_ucr_Crop                 icl_ucr_MixedShapesRegularTrain
icl_ucr_MixedShapesSmallTrain
```

### Sensor / Device (36)

```
icl_ucr_Car                 icl_ucr_ChlorineConcentration icl_ucr_Computers
icl_ucr_Earthquakes         icl_ucr_ElectricDevices      icl_ucr_FordA
icl_ucr_FordB               icl_ucr_ItalyPowerDemand     icl_ucr_LargeKitchenAppliances
icl_ucr_Lightning2          icl_ucr_Lightning7           icl_ucr_MoteStrain
icl_ucr_Plane               icl_ucr_RefrigerationDevices icl_ucr_ScreenType
icl_ucr_SmallKitchenAppliances icl_ucr_SonyAIBORobotSurface1 icl_ucr_SonyAIBORobotSurface2
icl_ucr_StarLightCurves     icl_ucr_Trace                icl_ucr_Wafer
icl_ucr_BME                 icl_ucr_Chinatown            icl_ucr_DodgerLoopDay
icl_ucr_DodgerLoopGame      icl_ucr_DodgerLoopWeekend    icl_ucr_FreezerRegularTrain
icl_ucr_FreezerSmallTrain   icl_ucr_HouseTwenty          icl_ucr_InsectEPGRegularTrain
icl_ucr_InsectEPGSmallTrain icl_ucr_InsectWingbeatSound  icl_ucr_MelbournePedestrian
icl_ucr_PowerCons           icl_ucr_SemgHandGenderCh2    icl_ucr_SmoothSubspace
```

### Motion / HAR (16)

```
icl_ucr_CricketX            icl_ucr_CricketY             icl_ucr_CricketZ
icl_ucr_GunPoint            icl_ucr_GunPointAgeSpan      icl_ucr_GunPointMaleVersusFemale
icl_ucr_GunPointOldVersusYoung icl_ucr_Haptics           icl_ucr_InlineSkate
icl_ucr_PickupGestureWiimoteZ icl_ucr_ShakeGestureWiimoteZ icl_ucr_ToeSegmentation1
icl_ucr_ToeSegmentation2    icl_ucr_UWaveGestureLibraryAll icl_ucr_Worms
icl_ucr_WormsTwoClass
```

### ECG / Medical (4)

```
icl_ucr_ECG200              icl_ucr_ECG5000              icl_ucr_ECGFiveDays
icl_ucr_TwoLeadECG
```

### Spectrographic / Chemometrics (7)

```
icl_ucr_Beef                icl_ucr_Coffee               icl_ucr_EthanolLevel
icl_ucr_Ham                 icl_ucr_Meat                 icl_ucr_OliveOil
icl_ucr_Strawberry          icl_ucr_Wine
```

### Simulated / Synthetic (4)

```
icl_ucr_CBF                 icl_ucr_SyntheticControl     icl_ucr_TwoPatterns
icl_ucr_UMD
```

---

## Domain Descriptions

Each dataset can optionally have a human-written domain description injected into the prompt. Descriptions live in:

```
ucr_descriptions/
└── {DatasetName}/
    └── description.txt
```

Example (`ucr_descriptions/GunPoint/description.txt`):
```
Motion tracking data of an actor's hand. The task is to classify whether the actor is drawing a gun from a hip holster or simply pointing a finger.
```

Descriptions are loaded at module import time into the `UCR_DESCRIPTIONS` dict. When `--use_label_desc 0` (default), the description from this dict is still included in the prompt (as `desc`). When `--use_label_desc 1`, the description is read fresh from the file at runtime, enabling per-run customization.

If no description is available, the description field in the prompt is an empty string.

---

## Experimental Protocol

### Primary Experiment: `random_k1_comparison_full`

- **Datasets:** 94 feasible UCR datasets listed above
- **Seeds:** 8 independent runs (seeds 0–7)
- **k:** 1 (one support example per class)
- **Strategy:** `random`
- **Max test samples:** 250 per run (seeded random subsample, same subset across all models)
- **Batch size:** 1

For each `(method, seed, dataset)` triplet, a separate job is submitted. Results are written to `outputs/evaluation_results_{task_id}_{num_samples}_{method}_{exp_id}.json` and logged to W&B.

### Result File Format

```json
{
  "args": {
    "task_id": "icl_ucr_GunPoint",
    "method": "Qwen/Qwen3-4B-Instruct-2507",
    "num_shots": 1,
    "random_seed": 0,
    "exp_id": "random_k1_comparison_full"
  },
  "metrics": {
    "balanced_accuracy": 0.87,
    "f1_macro": 0.85,
    "f1_weighted": 0.86,
    "precision_macro": 0.88,
    "precision_weighted": 0.86,
    "recall_macro": 0.87,
    "recall_weighted": 0.87,
    "total_test_size": 250,
    "num_of_classes": 2
  },
  "sample_count": 250
}
```

---

## Design Notes

- **Why balanced accuracy?** UCR datasets are often class-imbalanced. Balanced accuracy (average per-class recall) avoids inflating scores on majority classes. It equals macro recall.
- **Why integer labels?** UCR labels are originally arbitrary numeric/string values. We cast to `int64` on load. The prompt uses the raw integer (e.g., `1`, `2`) — no remapping.
- **Why normalize to [-1, 1]?** Standard preprocessing for LLM consumption. Keeps all values in a consistent range regardless of original scale. Stats computed on train split to simulate real deployment.
- **Why `return_full_text=False`?** The HF pipeline strips the prompt from the output. We only want the model's generated continuation (the predicted label), not the full prompt echoed back.
- **Why disable thinking mode?** Qwen3/3.6 models have a thinking mode that prepends a `<think>...</think>` reasoning block. This is stripped before label extraction to prevent the reasoning text from interfering with pattern matching.
