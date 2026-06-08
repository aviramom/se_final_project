"""Generates tsexam1_demo.ipynb — run once to create the notebook."""

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14.0"},
}


def md(source):
    return nbformat.v4.new_markdown_cell(source)


def code(source):
    return nbformat.v4.new_code_cell(source)


# ── Title ─────────────────────────────────────────────────────────────────────
nb.cells.append(md("""\
# TimeSeriesExam1 — Dataset Inspection & Evaluation Demo

This notebook shows:
1. What a **`Sample`** from `TimeSeriesExam1Dataset` looks like in both the
   **canonical (separate)** form and the **combined** form an LLM sees.
2. How to **run an evaluation** with the `LocalEvaluationPipeline`.
3. How to **read the results**: aggregate metrics, per-sample predictions,
   and breakdowns by difficulty / category / number of options.
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
nb.cells.append(md("## 1 — Setup"))
nb.cells.append(code("""\
import sys, pathlib
# Make sure the project root is on the path when the notebook runs from notebooks/
_root = pathlib.Path().resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.float_format", "{:.4f}".format)

from fmeval.core.datasets.tsexam1 import TimeSeriesExam1Dataset
from fmeval.core.datasets.formatters import SampleFormatter
from fmeval.core.models.mock_model import MockModel
from fmeval.core.metrics.mcq_metrics import MCQMetrics
from fmeval.evaluation.pipeline import LocalEvaluationPipeline

print("All imports OK.")
"""))

# ── Part 1 header ─────────────────────────────────────────────────────────────
nb.cells.append(md("""\
---
## Part 1 — What a Sample Looks Like

We load a small slice of the dataset and inspect each field of the `Sample` dataclass.
"""))

# ── Cell 2: Load ──────────────────────────────────────────────────────────────
nb.cells.append(code("""\
# Load 10 samples — enough to guarantee both 1-TS and 2-TS rows appear
ds = TimeSeriesExam1Dataset(max_samples=10)
samples = list(ds)

one_ts_sample = next(s for s in samples if len(s.input_ts) == 1)
two_ts_sample  = next(s for s in samples if len(s.input_ts) == 2)

print(f"Total samples loaded : {len(samples)}")
print(f"1-TS samples          : {sum(len(s.input_ts) == 1 for s in samples)}")
print(f"2-TS samples          : {sum(len(s.input_ts) == 2 for s in samples)}")
"""))

# ── Cell 3: input_text ────────────────────────────────────────────────────────
nb.cells.append(md("""\
### 1.1 — `input_text` (canonical / separate form)

`<TS_0>` is a placeholder — the raw numpy array lives in `input_ts[0]`.
This is what a model with a **dedicated TS encoder** receives as-is.
"""))
nb.cells.append(code("""\
print(one_ts_sample.input_text)
"""))

# ── Cell 4: input_ts ──────────────────────────────────────────────────────────
nb.cells.append(md("### 1.2 — `input_ts` (raw time-series arrays)"))
nb.cells.append(code("""\
ts = one_ts_sample.input_ts[0]
print(f"  number of arrays : {len(one_ts_sample.input_ts)}")
print(f"  dtype            : {ts.dtype}")
print(f"  shape            : {ts.shape}")
print(f"  min / mean / max : {ts.min():.3f} / {ts.mean():.3f} / {ts.max():.3f}")
print(f"  first 12 values  : {ts[:12].tolist()}")
"""))

# ── Cell 5: plot 1-TS ─────────────────────────────────────────────────────────
nb.cells.append(md("### 1.3 — Visualising the time series"))
nb.cells.append(code("""\
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(one_ts_sample.input_ts[0], lw=0.7, color="steelblue")
ax.set_title(
    f"input_ts[0]  |  category: {one_ts_sample.metadata['category']} "
    f"— {one_ts_sample.metadata['subcategory']}  |  difficulty: {one_ts_sample.metadata['difficulty']}"
)
ax.set_xlabel("time step")
ax.set_ylabel("value")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ── Cell 6: output + metadata ─────────────────────────────────────────────────
nb.cells.append(md("### 1.4 — `output` and `metadata`"))
nb.cells.append(code("""\
print("output (ground-truth answer):")
print(" ", repr(one_ts_sample.output))
print()
print("metadata:")
for k, v in one_ts_sample.metadata.items():
    print(f"  {k:<22} {v}")
"""))

# ── Cell 7: combined form ─────────────────────────────────────────────────────
nb.cells.append(md("""\
### 1.5 — Combined form (what an LLM sees)

`SampleFormatter.to_combined()` replaces every `<TS_N>` token with the serialised
float values. This is what a standard LLM receives — **a single text string** with
the time series inline.
"""))
nb.cells.append(code("""\
combined = SampleFormatter.to_combined(one_ts_sample)

print("=== combined.input_text (first 600 chars) ===")
print(combined.input_text[:600])
print("  ... (truncated)")
print()
print(f"combined.input_ts : {combined.input_ts}  ← empty; TS is now inside the text")
"""))

# ── Cell 8: 2-TS sample ───────────────────────────────────────────────────────
nb.cells.append(md("""\
### 1.6 — Sample with two time series

Some questions compare two series. Both placeholders appear in `input_text`;
both arrays are in `input_ts`.
"""))
nb.cells.append(code("""\
print("input_text:")
print(two_ts_sample.input_text)
print()
print(f"len(input_ts) = {len(two_ts_sample.input_ts)}")
print(f"input_ts[0].shape = {two_ts_sample.input_ts[0].shape}")
print(f"input_ts[1].shape = {two_ts_sample.input_ts[1].shape}")
"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
colors = ["steelblue", "darkorange"]
for i, ax in enumerate(axes):
    ax.plot(two_ts_sample.input_ts[i], lw=0.7, color=colors[i])
    ax.set_ylabel(f"input_ts[{i}]")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    f"2-TS sample  |  category: {two_ts_sample.metadata['category']}  "
    f"|  answer: {two_ts_sample.output}",
    fontsize=11,
)
plt.tight_layout()
plt.show()
"""))

# ── Part 2 header ─────────────────────────────────────────────────────────────
nb.cells.append(md("""\
---
## Part 2 — Running the Evaluation Pipeline

```
TimeSeriesExam1Dataset
        │  yields Sample objects (separate form)
        ▼
LocalEvaluationPipeline
    ├─ model.format_input(sample)  →  calls SampleFormatter.to_combined (MockModel)
    ├─ model.predict(batch)        →  returns list[str]   ("A)" for every input)
    └─ MCQMetrics.compute(preds, targets)
        │
        ▼
RunResult  (metrics + per-sample SamplePrediction list)
```

The **`MockModel`** always answers `"A)"` — it is the simplest possible baseline
and lets us verify the full pipeline is wired correctly before plugging in a real model.
"""))

# ── Cell 9: run eval ──────────────────────────────────────────────────────────
nb.cells.append(code("""\
# Load 200 samples to get a meaningful metric distribution
ds_eval = TimeSeriesExam1Dataset(max_samples=200)

model    = MockModel(answer="A")
metric   = MCQMetrics()
pipeline = LocalEvaluationPipeline(model, metric, batch_size=32, verbose=False)

result = pipeline.run(ds_eval)
print(f"Run complete — {result.num_samples} samples evaluated.")
print(f"run_config: {result.run_config}")
"""))

# ── Part 3 header ─────────────────────────────────────────────────────────────
nb.cells.append(md("""\
---
## Part 3 — Reading the Results
"""))

# ── Cell 10: summary ─────────────────────────────────────────────────────────
nb.cells.append(md("### 3.1 — Text summary"))
nb.cells.append(code("""\
print(result.summary())
"""))

# ── Cell 11: dataframe ───────────────────────────────────────────────────────
nb.cells.append(md("""\
### 3.2 — Per-sample DataFrame

`result.to_dataframe()` produces one row per sample. All `Sample.metadata`
fields (difficulty, category, etc.) are promoted to top-level columns so you
can filter and slice freely.
"""))
nb.cells.append(code("""\
df = result.to_dataframe()
print(f"Shape: {df.shape}  columns: {list(df.columns)}")
df[["sample_idx", "predicted_letter", "correct_letter", "is_correct",
    "difficulty", "category", "num_options"]].head(12)
"""))

# ── Cell 12: breakdown difficulty ─────────────────────────────────────────────
nb.cells.append(md("### 3.3 — Breakdown by difficulty"))
nb.cells.append(code("""\
bd_difficulty = result.breakdown_by("difficulty")
display(bd_difficulty)
"""))

# ── Cell 13: breakdown category ───────────────────────────────────────────────
nb.cells.append(md("### 3.4 — Breakdown by category"))
nb.cells.append(code("""\
bd_category = result.breakdown_by("category").sort_values("accuracy", ascending=False)
display(bd_category)
"""))

# ── Cell 14: breakdown num_options ────────────────────────────────────────────
nb.cells.append(md("### 3.5 — Breakdown by number of options"))
nb.cells.append(code("""\
bd_options = result.breakdown_by("num_options")
display(bd_options)
"""))

# ── Cell 15: charts ───────────────────────────────────────────────────────────
nb.cells.append(md("### 3.6 — Charts"))
nb.cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── Accuracy by difficulty ──────────────────────────────────────────────────
ax = axes[0]
bd_d = bd_difficulty.sort_values("accuracy", ascending=False)
bars = ax.bar(bd_d["difficulty"], bd_d["accuracy"], color="steelblue", alpha=0.85)
ax.axhline(result.metrics["accuracy"], color="crimson", ls="--", lw=1.5, label=f"overall {result.metrics['accuracy']:.3f}")
ax.set_ylim(0, 1)
ax.set_xlabel("difficulty")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy by Difficulty")
ax.legend(fontsize=8)
for bar, val in zip(bars, bd_d["accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

# ── F1 macro by difficulty ──────────────────────────────────────────────────
ax = axes[1]
bars = ax.bar(bd_d["difficulty"], bd_d["f1_macro"], color="darkorange", alpha=0.85)
ax.axhline(result.metrics["f1_macro"], color="crimson", ls="--", lw=1.5, label=f"overall {result.metrics['f1_macro']:.3f}")
ax.set_ylim(0, 1)
ax.set_xlabel("difficulty")
ax.set_ylabel("F1 macro")
ax.set_title("F1 Macro by Difficulty")
ax.legend(fontsize=8)
for bar, val in zip(bars, bd_d["f1_macro"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

# ── Accuracy by num_options ─────────────────────────────────────────────────
ax = axes[2]
bd_o = bd_options.sort_values("num_options")
x_labels = [str(v) for v in bd_o["num_options"]]
bars = ax.bar(x_labels, bd_o["accuracy"], color="seagreen", alpha=0.85)
ax.axhline(result.metrics["accuracy"], color="crimson", ls="--", lw=1.5, label=f"overall {result.metrics['accuracy']:.3f}")
ax.set_ylim(0, 1)
ax.set_xlabel("number of options")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy by Number of Options\\n(random baseline: 1/n_options)")
for n_opts, bar, val in zip(bd_o["num_options"], bars, bd_o["accuracy"]):
    baseline = 1 / n_opts
    ax.axhline(baseline, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
ax.legend(fontsize=8)

plt.suptitle(
    f"MockModel (always A)  |  n={result.num_samples} samples  |  dataset: tsexam1",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.show()
"""))

# ── Cell 16: export ───────────────────────────────────────────────────────────
nb.cells.append(md("### 3.7 — Export to JSON / CSV"))
nb.cells.append(code("""\
import json, pathlib

out_dir = pathlib.Path("results")
out_dir.mkdir(exist_ok=True)

# JSON — full result with every sample prediction
json_path = out_dir / "mock_tsexam1.json"
json_path.write_text(result.to_json())
print(f"Saved: {json_path}  ({json_path.stat().st_size // 1024} KB)")

# CSV — flat per-sample table
csv_path = out_dir / "mock_tsexam1_samples.csv"
result.to_dataframe().to_csv(csv_path, index=False)
print(f"Saved: {csv_path}  ({csv_path.stat().st_size // 1024} KB)")

# Preview the JSON structure
print("\\nJSON top-level keys:", list(json.loads(json_path.read_text()).keys()))
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
import pathlib
pathlib.Path("notebooks").mkdir(exist_ok=True)

output_path = pathlib.Path("notebooks/tsexam1_demo.ipynb")
with output_path.open("w") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {output_path}  ({len(nb.cells)} cells)")
