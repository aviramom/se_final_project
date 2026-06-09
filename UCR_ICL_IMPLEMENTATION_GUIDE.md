# UCR ICL Benchmark — Implementation Guide for Claude Code

This guide tells you exactly how to implement the UCR in-context learning benchmark from scratch in a new repository. Read `UCR_ICL_BENCHMARK.md` first for the high-level design.

---

## Directory Structure

Create this layout:

```
your_repo/
├── data_provider/
│   ├── __init__.py
│   ├── dataset_iclucr.py       # UCRDataset class + domain descriptions dict
│   └── data_utils.py           # load_dataset_for_task() dispatcher
├── models/
│   ├── __init__.py
│   ├── base_model.py           # Abstract BaseModelWrapper
│   ├── instruct_model.py       # InstructModel (HF pipeline) + LargeInstructModel (vLLM)
│   └── baselines.py            # RandomBaseline, KNNBaseline
├── evaluations/
│   ├── __init__.py
│   └── icl_ucr_eval.py         # run_evaluation_icl_ucr(), _extract_predicted_label()
├── utils/
│   ├── __init__.py
│   ├── args.py                 # argparse CLI definitions
│   ├── formatting.py           # icl_classification_format()
│   └── model.py                # method_wrapper_dict, load_base_model()
├── picking_strategy.py         # get_support_set() + 5 strategies
├── dataset.py                  # MultiTSDataset + from_icl_ucr_dataset()
├── eval.py                     # Entry point: run_eval()
├── ucr_descriptions/           # Optional: domain description text files
│   └── {DatasetName}/
│       └── description.txt
└── icl_experiment_scripts/
    ├── run_single_task_gpu.sh
    ├── run_single_task_gpu_large.sh
    ├── run_single_task_cpu.sh
    └── icl_ucr_comparison_full.sh
```

---

## Step 1 — UCR Dataset Loader (`data_provider/dataset_iclucr.py`)

This module loads UCR ARFF files and provides an optional domain description dictionary.

```python
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy.io import arff

_UCR_DESC_DIR = os.path.join(os.path.dirname(__file__), '..', 'ucr_descriptions')

def _load_ucr_txt_descriptions():
    """Scan ucr_descriptions/{DatasetName}/description.txt and return a dict."""
    descriptions = {}
    if not os.path.isdir(_UCR_DESC_DIR):
        return descriptions, {}
    for dataset_name in os.listdir(_UCR_DESC_DIR):
        txt_path = os.path.join(_UCR_DESC_DIR, dataset_name, 'description.txt')
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()
        if text.strip():
            descriptions[dataset_name] = text
    return descriptions, {}

UCR_DESCRIPTIONS, UCR_LABEL_DESCRIPTIONS = _load_ucr_txt_descriptions()


class UCRDataset(Dataset):
    """Loads a UCR dataset from ARFF files with train/test split support."""

    # Hardcoded fallback descriptions — embed your dataset descriptions here
    # (or rely solely on ucr_descriptions/ files above)
    UCR_DESCRIPTIONS = {
        "GunPoint": "Motion tracking data of an actor's hand. The task is to classify whether the actor is drawing a gun from a hip holster or simply pointing a finger.",
        "Beef": "Spectrographic analysis of beef. The task is to classify whether the beef is pure or adulterated with offal.",
        # ... add all datasets you need (see full list in UCR_ICL_BENCHMARK.md)
    }

    def __init__(self, ucr_path: str, split: str = None, name: str = None):
        """
        Args:
            ucr_path: Path to directory containing {DatasetName}_TRAIN.arff and _TEST.arff
            split: 'train' or 'test'
            name: Dataset name string (used to look up description)
        """
        dataset_name = os.path.basename(ucr_path)
        if split == "train":
            df = self._load_file(os.path.join(ucr_path, f"{dataset_name}_TRAIN.arff"))
        elif split == "test":
            df = self._load_file(os.path.join(ucr_path, f"{dataset_name}_TEST.arff"))
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.desc = self.UCR_DESCRIPTIONS.get(name or dataset_name, "")

        # Last column is the label; all others are the time series
        labels_raw = df.iloc[:, -1].astype('int64')
        self.labels = torch.tensor(labels_raw.values, dtype=torch.long)
        self.data = torch.tensor(df.iloc[:, :-1].values.astype('float32'), dtype=torch.float32)

        # Compute normalization stats on non-NaN values
        valid = self.data[~torch.isnan(self.data)]
        self.min_val = float(valid.min())
        self.max_val = float(valid.max())

    def _load_file(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('.arff'):
            return self._load_arff(file_path)
        return pd.read_csv(file_path, header=None, sep='\t')

    def _load_arff(self, arff_path: str) -> pd.DataFrame:
        data, meta = arff.loadarff(arff_path)
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns (normalized_ts, label). Normalization: min-max to [-1, 1]."""
        ts = self.data[idx]
        ts = 2 * (ts - self.min_val) / (self.max_val - self.min_val) - 1
        return ts, self.labels[idx]
```

**Critical:** The normalization range (`min_val`, `max_val`) is computed from the **training split only** and applied to both train and test. Always instantiate the training split first if you need the stats.

---

## Step 2 — Dataset Loading Dispatcher (`data_provider/data_utils.py`)

Add the `icl_ucr` case to your dataset loading dispatcher:

```python
import os
from data_provider import dataset_iclucr as dataset_ucr

def load_dataset_for_task(task_id: str, data_path: str) -> dict:
    """
    Returns {task_id: dataset_object}.
    For ICL UCR tasks, dataset_object is {"train": UCRDataset, "test": UCRDataset}.
    """
    if "icl_ucr" in task_id.lower():
        dataset_name = task_id.replace("ICL_UCR_", "").replace("icl_ucr_", "")
        ucr_path = os.path.join(data_path, "Univariate_arff", dataset_name)

        train_dataset = dataset_ucr.UCRDataset(ucr_path, split="train")
        test_dataset = dataset_ucr.UCRDataset(ucr_path, split="test")

        return {task_id: {"train": train_dataset, "test": test_dataset}}

    else:
        raise ValueError(f"Unsupported task_id: {task_id}")
```

The UCR archive must be placed at `{data_path}/Univariate_arff/`. Each dataset folder is named exactly as in the archive (e.g., `GunPoint/`, `Beef/`).

---

## Step 3 — Support Set Selection (`picking_strategy.py`)

```python
import random
import numpy as np
from collections import defaultdict


def get_support_set(train_ds, strategy="first", k_shots=1, seed=None):
    """
    Select k examples per class from the training dataset.

    Returns:
        List of (ts_list, label) where ts_list = [flat_list_of_floats]
    """
    if strategy == "first":
        return _first(train_ds, k_shots)
    elif strategy == "random":
        return _random(train_ds, k_shots, seed)
    elif strategy == "medoid":
        return _medoid(train_ds, k_shots)
    elif strategy == "medoid_dtw":
        return _medoid_dtw(train_ds, k_shots)
    elif strategy == "reversed":
        return _reversed(train_ds, k_shots)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_array(ts):
    if hasattr(ts, "numpy"):
        return ts.numpy().flatten().astype(float)
    return np.array(ts).flatten().astype(float)


def _to_list(ts):
    """Wrap ts into [list_of_floats] — the expected format for input_ts."""
    if hasattr(ts, "tolist"):
        return [ts.tolist()]
    return [ts]


def _label_val(label):
    if hasattr(label, "item"):
        return label.item()
    return label


# ── strategies ────────────────────────────────────────────────────────────────

def _first(train_ds, k_shots):
    examples = []
    label_counts = {}
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        label = _label_val(label_i)
        if label_counts.get(label, 0) < k_shots:
            examples.append((_to_list(ts_i), label))
            label_counts[label] = label_counts.get(label, 0) + 1
    return examples


def _reversed(train_ds, k_shots):
    return _first(train_ds, k_shots)[::-1]


def _random(train_ds, k_shots, seed=None):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        _, label_i = train_ds[i]
        by_label[_label_val(label_i)].append(i)
    examples = []
    for label, indices in by_label.items():
        chosen = rng.sample(indices, min(k_shots, len(indices)))
        for idx in chosen:
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
    return examples


def _medoid(train_ds, k_shots):
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        by_label[_label_val(label_i)].append((i, _to_array(ts_i)))
    examples = []
    for label, items in by_label.items():
        if len(items) == 1:
            idx, _ = items[0]
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
            continue
        arrays = np.stack([arr for _, arr in items])   # (N, T)
        dist_sums = np.zeros(len(items))
        for j in range(len(items)):
            diffs = arrays - arrays[j]
            dist_sums[j] = np.sqrt((diffs ** 2).sum(axis=1)).sum()
        best = np.argsort(dist_sums)[:k_shots]
        for b in best:
            orig_idx, _ = items[b]
            ts_i, _ = train_ds[orig_idx]
            examples.append((_to_list(ts_i), label))
    return examples


def _dtw_distance(a, b):
    """Sakoe-Chiba DTW (no window constraint)."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return np.sqrt(dtw[n, m])


def _medoid_dtw(train_ds, k_shots):
    by_label = defaultdict(list)
    for i in range(len(train_ds)):
        ts_i, label_i = train_ds[i]
        by_label[_label_val(label_i)].append((i, _to_array(ts_i)))
    examples = []
    for label, items in by_label.items():
        if len(items) == 1:
            idx, _ = items[0]
            ts_i, _ = train_ds[idx]
            examples.append((_to_list(ts_i), label))
            continue
        dist_sums = np.zeros(len(items))
        for j in range(len(items)):
            for k in range(len(items)):
                if k != j:
                    dist_sums[j] += _dtw_distance(items[j][1], items[k][1])
        best = np.argsort(dist_sums)[:k_shots]
        for b in best:
            orig_idx, _ = items[b]
            ts_i, _ = train_ds[orig_idx]
            examples.append((_to_list(ts_i), label))
    return examples
```

---

## Step 4 — Prompt Formatter (`utils/formatting.py`)

```python
def icl_classification_format(desc: str, examples: str, target: str, options: list) -> str:
    """Assemble the full few-shot classification prompt."""
    instruction = f"""{desc.strip()}

{examples.strip()}

{target.strip()}
Return ONLY the label as one of: {options} without any explanation
"""
    return "Time Series Classification.\n" + instruction
```

The `examples` argument is a pre-built string with `<ts><ts/>` placeholders; `target` is the query placeholder string. The caller (MultiTSDataset) handles placeholder substitution.

---

## Step 5 — Dataset Wrapper (`dataset.py`)

The `MultiTSDataset` class wraps the raw UCR data into a structured format for the dataloader. The key method is `from_icl_ucr_dataset`.

```python
import os
import numpy as np
from datasets import Dataset, Features, Value, Sequence
from utils.formatting import icl_classification_format
from picking_strategy import get_support_set

FORMAT_KEYS = {
    "input_text":  Value("string"),
    "output_text": Value("string"),
    "input_ts":    Sequence(Sequence(Value("float32"))),  # 2D: list of series
    "options":     Sequence(Value("string")),
    "task_id":     Value("string"),
    "mean":        Sequence(Value("float32")),
    "std":         Sequence(Value("float32")),
    # Add other fields as needed (task, domain, metadata, etc.)
}


class MultiTSDataset:
    def __init__(self, input_mode="combined", method="", ts_place_holder=None):
        self.input_mode = input_mode
        self.method = method
        self.ts_place_holder = ts_place_holder or "<ts><ts/>"
        self.data_format = {key: [] for key in FORMAT_KEYS.keys()}

    def __len__(self):
        return len(self.data_format["input_text"])

    def add_sample(self, **kwargs):
        for key in FORMAT_KEYS.keys():
            self.data_format[key].append(kwargs.get(key, None))

    def to_hf_dataset(self) -> Dataset:
        return Dataset.from_dict(self.data_format, features=Features(FORMAT_KEYS))

    def _compute_mean_std(self, ts_list):
        means, stds = [], []
        for series in ts_list:
            if series is None or len(series) == 0:
                means.append(None); stds.append(None)
                continue
            arr = np.array([np.nan if x is None else x for x in series], dtype=float)
            means.append(float(np.nanmean(arr)))
            stds.append(float(np.nanstd(arr)))
        return means, stds

    def from_dataset(self, dataset, task_id="icl_ucr_GunPoint", args=None):
        """Route to the correct loader based on task_id prefix."""
        task_key = task_id.split("_")[0].lower()
        if task_key == "icl":
            self.from_icl_ucr_dataset(dataset, task_id=task_id, args=args)
        else:
            raise ValueError(f"Unsupported task_id: {task_id}")

    def from_icl_ucr_dataset(self, dataset: dict, task_id: str = "icl_ucr_GunPoint", args=None):
        """
        Build MultiTSDataset from a {"train": UCRDataset, "test": UCRDataset} dict.

        For each test sample: assemble a prompt with k support examples + the query,
        substitute all <ts><ts/> placeholders with actual numeric values.
        """
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # 1. Build support set
        examples = get_support_set(
            train_dataset,
            strategy=getattr(args, "picking_strategy", "random"),
            k_shots=getattr(args, "num_shots", 1),
            seed=getattr(args, "random_seed", None),
        )

        # 2. Load description
        if getattr(args, "use_label_desc", 0):
            _name = task_id.replace("ICL_UCR_", "").replace("icl_ucr_", "")
            _desc_dir = getattr(args, "desc_dir", "ucr_descriptions")
            _desc_path = os.path.join(os.path.dirname(__file__), _desc_dir, _name, "description.txt")
            if os.path.isfile(_desc_path):
                with open(_desc_path, encoding="utf-8") as f:
                    description = f.read().strip()
            else:
                description = "Time Series Classification."
        else:
            description = getattr(train_dataset, "desc", "")

        # 3. Extract unique labels from the support set
        options = list(set([ex[1] for ex in examples]))

        # 4. Build the base prompt template (with <ts><ts/> placeholders)
        def build_input(examples, desc, opts):
            input_text = "\n--- EXAMPLES ---\n"
            support_ts = []
            for i, (ts, label) in enumerate(examples):
                input_text += f"\nExample {i+1} Time Series: <ts><ts/>\nLabel: {label}\n"
                support_ts.append(ts[0])  # unwrap the [list] wrapper
            target = "\n--- TARGET ---\n"
            target += "New Time Series: <ts><ts/>\n"
            full_prompt = icl_classification_format(desc, input_text, target, opts)
            return full_prompt, support_ts

        # 5. TS placeholder substitution helper
        def combine_ts_text(text, ts_list):
            placeholder = "<ts><ts/>"
            for ts in ts_list:
                if isinstance(ts[0], list):
                    ts_str = ", ".join([f"{x:.4f}" for x in ts[0]])
                else:
                    ts_str = ", ".join([f"{x:.4f}" for x in ts])
                text = text.replace(placeholder, f"[{ts_str}]", 1)
            return text

        input_prompt, support_ts_list = build_input(examples, description, options)

        # 6. Iterate test set and build one sample per query
        for i in range(len(test_dataset)):
            ts_i, label_i = test_dataset[i]
            ts_query = ts_i.tolist() if hasattr(ts_i, "tolist") else ts_i
            label_val = label_i.item() if hasattr(label_i, "item") else label_i

            current_ts = support_ts_list.copy()
            current_ts.append(ts_query)

            current_text = input_prompt
            if "combined" in self.input_mode:
                current_text = combine_ts_text(current_text, current_ts)

            mean, std = self._compute_mean_std(current_ts)
            self.add_sample(
                input_text=current_text,
                output_text=str(label_val),
                input_ts=current_ts,
                options=[str(o) for o in options],
                task_id=task_id,
                mean=mean,
                std=std,
            )
```

**Key implementation note:** `options` are stored as strings in the HF dataset (the `Sequence(Value("string"))` type). The `output_text` (gold label) is also stored as a string via `str(label_val)`. Both must be strings for comparison in the evaluation loop.

---

## Step 6 — Model Interface (`models/base_model.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModelWrapper(ABC):

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        """
        Args:
            batch: dict with at minimum key "input_text" -> List[str]
        Returns:
            List[str]: one predicted string per input
        """
        pass

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {}

    @classmethod
    def get_relevant_args(cls, args, parser):
        """Override in subclasses to filter or augment args."""
        return args
```

---

## Step 7 — InstructModel (`models/instruct_model.py`)

This is the standard wrapper for HuggingFace causal language models.

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any, List, Optional
from models.base_model import BaseModelWrapper


class InstructModel(BaseModelWrapper):

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method = getattr(args, "method", None)
        self.cache_dir = getattr(args, "cache_dir", None) or None
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    @staticmethod
    def get_args_dict():
        return {
            "model_type": "instruct",
            "device": "cuda",
            "max_seq_length": 4096,
            "max_new_tokens": 50,
            "format": "chat",
            "input_mode": "combined",
        }

    def load_model(self, model_path=None, cache_dir=None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir
        quantization = getattr(self.args, "quantization", None)

        hf_kwargs = dict(trust_remote_code=True, cache_dir=cache, torch_dtype=torch.bfloat16)
        if quantization == "8bit":
            hf_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            hf_kwargs.pop("torch_dtype")
        elif quantization == "4bit":
            hf_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            hf_kwargs.pop("torch_dtype")

        if quantization:
            self.model = AutoModelForCausalLM.from_pretrained(path, **hf_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **hf_kwargs).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, return_tensors="pt",
            max_length=self.args.max_seq_length, padding=True, truncation=True,
            cache_dir=cache,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pipe_kwargs = dict(model=self.model, tokenizer=self.tokenizer)
        if quantization:
            pipe_kwargs["device_map"] = "auto"
        else:
            pipe_kwargs["device"] = self.device
        self.pipeline = transformers.pipeline("text-generation", **pipe_kwargs)

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if self.model is None:
            self.load_model()

        def _apply_template(q):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )

        def _strip_thinking(text: str) -> str:
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()

        formatted = [_apply_template(q) for q in prompts]
        outputs = self.pipeline(
            formatted,
            max_new_tokens=getattr(self.args, "max_new_tokens", max_new_tokens),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
            batch_size=getattr(self.args, "batch_size", 1),
            do_sample=False,
        )
        return [_strip_thinking(o[0]["generated_text"]) for o in outputs]
```

### LargeInstructModel (vLLM, for 27B+ models)

```python
class LargeInstructModel(InstructModel):
    """Uses vLLM with tensor parallelism across all available GPUs."""

    def __init__(self, args, device="cuda"):
        super().__init__(args, device)
        self._use_vllm = False
        self._vllm_llm = None

    def load_model(self, model_path=None, cache_dir=None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir
        try:
            from vllm import LLM
            n_gpus = torch.cuda.device_count() or 1
            self._vllm_llm = LLM(
                model=path, download_dir=cache,
                tensor_parallel_size=n_gpus, dtype="bfloat16",
                max_model_len=max(getattr(self.args, "max_seq_length", 4096), 16384),
                trust_remote_code=True,
                enforce_eager=True,           # skip 30-60 min CUDA graph compilation
                disable_custom_all_reduce=True,  # required for non-adjacent PCIe GPUs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._use_vllm = True
            self.model = self._vllm_llm
        except ImportError:
            super().load_model(model_path, cache_dir)

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        if not self._use_vllm:
            return super().generate(batch, max_new_tokens, **kwargs)
        if self.model is None:
            self.load_model()
        from vllm import SamplingParams

        def _apply_template(q):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )

        prompts = [_apply_template(q) for q in batch["input_text"]]
        sp = SamplingParams(temperature=0.0, max_tokens=getattr(self.args, "max_new_tokens", max_new_tokens))
        outputs = self._vllm_llm.generate(prompts, sp)

        def _strip_thinking(text):
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()

        return [_strip_thinking(o.outputs[0].text) for o in outputs]
```

---

## Step 8 — Baselines (`models/baselines.py`)

### RandomBaseline

```python
import re, random
from typing import Any, Dict, List
from models.base_model import BaseModelWrapper

class RandomBaseline(BaseModelWrapper):
    def __init__(self, args: Any, device: str = "cpu"):
        self.args = args
        self.rng = random.Random(getattr(args, "random_seed", None))

    def load_model(self): pass

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        return [str(self.rng.choice(self._parse_options(p)) or "") for p in prompts]

    @staticmethod
    def _parse_options(prompt: str) -> List[str]:
        match = re.search(r'Return ONLY the label as one of:\s*\[([^\]]+)\]', prompt)
        if not match:
            return []
        return [opt.strip() for opt in match.group(1).split(',')]
```

### KNNBaseline (1-NN with DTW)

```python
import re
import numpy as np
from typing import Any, List
from tslearn.metrics import dtw
from models.base_model import BaseModelWrapper

class KNNBaseline(BaseModelWrapper):
    def __init__(self, args: Any, device: str = "cpu"):
        self.args = args

    def load_model(self): pass

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        results = []
        for i, prompt in enumerate(prompts):
            support_labels = self._parse_support_labels(prompt)
            ts_list = batch["input_ts"][i]  # [support_0, ..., support_k, query]
            if not support_labels or len(ts_list) < 2:
                results.append("")
                continue
            support_arrays = [self._to_array(ts_list[j]) for j in range(len(support_labels))]
            query_array = self._to_array(ts_list[-1])
            dists = [dtw(query_array, s) for s in support_arrays]
            results.append(str(support_labels[int(np.argmin(dists))]))
        return results

    @staticmethod
    def _parse_support_labels(prompt: str) -> List[str]:
        return re.findall(r'Label:\s*(\S+)', prompt)

    @staticmethod
    def _to_array(ts) -> np.ndarray:
        if hasattr(ts, "numpy"):
            arr = ts.numpy().flatten().astype(float)
        else:
            arr = np.array(ts).flatten().astype(float)
        if np.isnan(arr).any():
            nans = np.isnan(arr)
            if nans.all():
                arr = np.zeros_like(arr)
            else:
                idxs = np.arange(len(arr))
                arr[nans] = np.interp(idxs[nans], idxs[~nans], arr[~nans])
        return arr
```

**Dependency:** `tslearn` (`pip install tslearn`). Alternatively, implement DTW yourself using the `_dtw_distance` function from `picking_strategy.py`.

---

## Step 9 — Evaluation Loop (`evaluations/icl_ucr_eval.py`)

```python
import re
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


def _parse_options(prompt: str) -> list:
    """Extract class options from 'Return ONLY the label as one of: [a, b, ...]'."""
    match = re.search(r'Return ONLY the label as one of:\s*\[([^\]]+)\]', prompt)
    if not match:
        return []
    return [opt.strip() for opt in match.group(1).split(',')]


def _extract_predicted_label(response: str, options: list) -> str:
    """Return the first matching option or 'INVALID_PREDICTION'."""
    for opt in options:
        if response == opt:
            return opt
        if f'The class is {opt}' in response or f'The class is <{opt}>' in response:
            return opt
        if re.search(r'Predicted\s*Label\s*:\s*["\'<\[]?\s*' + re.escape(opt) + r'(?!\d)', response, re.IGNORECASE):
            return opt
        if re.search(r'Predicted\s*:\s*["\'<\[]?\s*' + re.escape(opt) + r'(?!\d)', response, re.IGNORECASE):
            return opt
        if re.search(r'(?<!\w)label\s*:\s*["\'<\[]?\s*' + re.escape(opt) + r'(?!\d)', response, re.IGNORECASE):
            return opt
        if re.search(r'(?:correct\s+)?label\s+is\s+["\'<\[]?\s*' + re.escape(opt) + r'(?!\d)', response, re.IGNORECASE):
            return opt
    return "INVALID_PREDICTION"


def run_evaluation_icl_ucr(model, dataloader, args):
    """
    Iterate over all batches, run model inference, extract labels, compute metrics.

    Returns:
        results: dict of aggregate metrics
        input_output: dict of per-sample details for logging/display
    """
    if "ucr" not in args.task_id.lower():
        raise ValueError("task_id must contain 'ucr'")

    accuracy_scores, gold_answers, predicted_answers = [], [], []
    generated_texts, questions, input_ts_all = [], [], []

    for batch in tqdm(dataloader, desc=f"Evaluating {args.task_id}"):
        batch_prompts = batch["input_text"]
        gen_out = model.generate(batch)

        for i in range(len(batch_prompts)):
            answer = str(batch["output_text"][i]).strip()
            response = str(gen_out[i]).strip()

            score = 0
            if answer == response:
                score = 1
            elif f'The class is {answer}' in response or f'The class is <{answer}>' in response:
                score = 1
            elif re.search(r'Predicted\s*Label\s*:\s*["\'<\[]?\s*' + re.escape(answer) + r'(?!\d)', response, re.IGNORECASE):
                score = 1
            elif re.search(r'Predicted\s*:\s*["\'<\[]?\s*' + re.escape(answer) + r'(?!\d)', response, re.IGNORECASE):
                score = 1
            elif re.search(r'(?<!\w)label\s*:\s*["\'<\[]?\s*' + re.escape(answer) + r'(?!\d)', response, re.IGNORECASE):
                score = 1
            elif re.search(r'(?:correct\s+)?label\s+is\s+["\'<\[]?\s*' + re.escape(answer) + r'(?!\d)', response, re.IGNORECASE):
                score = 1

            accuracy_scores.append(score)
            questions.append(batch_prompts[i])
            generated_texts.append(response)
            gold_answers.append(answer)
            input_ts_all.append(batch["input_ts"][i])

            options = _parse_options(batch_prompts[i])
            predicted_answers.append(_extract_predicted_label(response, options))

    results = {
        "accuracy_scores": accuracy_scores,
        "num_of_classes": len(set(gold_answers)),
        "total_test_size": len(accuracy_scores),
        "balanced_accuracy": balanced_accuracy_score(gold_answers, predicted_answers),
        "f1_macro": f1_score(gold_answers, predicted_answers, average="macro", zero_division=0),
        "f1_weighted": f1_score(gold_answers, predicted_answers, average="weighted", zero_division=0),
        "precision_macro": precision_score(gold_answers, predicted_answers, average="macro", zero_division=0),
        "precision_weighted": precision_score(gold_answers, predicted_answers, average="weighted", zero_division=0),
        "recall_macro": recall_score(gold_answers, predicted_answers, average="macro", zero_division=0),
        "recall_weighted": recall_score(gold_answers, predicted_answers, average="weighted", zero_division=0),
    }
    input_output = {
        "questions": questions,
        "generated_texts": generated_texts,
        "gold_answers": gold_answers,
        "input_ts": input_ts_all,
    }
    return results, input_output
```

---

## Step 10 — CLI Arguments (`utils/args.py`)

Add these arguments to your argparse parser. All are needed by the ICL pipeline:

```python
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment identity
    parser.add_argument("--exp_id", type=str, default="1")
    parser.add_argument("--random_seed", type=int, default=2021)

    # Model
    parser.add_argument("--method", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--format", type=str, default="chat")
    parser.add_argument("--input_mode", type=str, default="combined")
    parser.add_argument("--device", type=str, default="cuda")

    # Data
    parser.add_argument("--data_path", type=str, default="/path/to/UCR/data/")
    parser.add_argument("--task_id", type=str, default="icl_ucr_GunPoint")

    # ICL-specific
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument("--picking_strategy", type=str, default="random",
                        choices=["first", "random", "medoid", "medoid_dtw", "reversed"])
    parser.add_argument("--use_label_desc", type=int, default=0)
    parser.add_argument("--desc_dir", type=str, default="ucr_descriptions")

    # Evaluation
    parser.add_argument("--num_samples", type=int, default=np.inf)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--display_samples", type=int, default=5)

    # Logging
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--project", type=str, default="my-project/multits")

    return parser
```

After parsing, convert sentinel strings to `None`:

```python
args = parser.parse_args()
if args.quantization == "none":
    args.quantization = None
if args.cache_dir == "":
    args.cache_dir = None
```

---

## Step 11 — Model Registry (`utils/model.py`)

```python
from models.instruct_model import InstructModel, LargeInstructModel
from models.baselines import RandomBaseline, KNNBaseline

method_wrapper_dict = {
    # HuggingFace instruct models (single GPU)
    "Qwen/Qwen3-4B-Instruct-2507": InstructModel,
    "Qwen/Qwen2.5-7B-Instruct": InstructModel,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": InstructModel,

    # Large models (multi-GPU vLLM)
    "Qwen/Qwen3.6-27B": LargeInstructModel,

    # Baselines
    "random_baseline": RandomBaseline,
    "knn_baseline": KNNBaseline,
}
```

---

## Step 12 — Entry Point (`eval.py`)

```python
import os, sys, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.args import get_parser
from utils.model import method_wrapper_dict
from data_provider.data_utils import load_dataset_for_task
from dataset import MultiTSDataset
from evaluations.icl_ucr_eval import run_evaluation_icl_ucr


def collate_fn(batch):
    """Simple collate: stack list fields, keep others as lists."""
    out = {}
    for key in batch[0].keys():
        out[key] = [item[key] for item in batch]
    return out


def run_eval(args=None):
    if args is None:
        parser = get_parser()
        args, _ = parser.parse_known_args()
        if args.quantization == "none":
            args.quantization = None
        if not args.cache_dir:
            args.cache_dir = None

    # Reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Load model
    wrapper_class = method_wrapper_dict[args.method]
    model = wrapper_class(args, device=getattr(args, "device", "cuda"))

    # Load dataset
    dataset_dict = load_dataset_for_task(args.task_id, args.data_path)
    dataset = dataset_dict[args.task_id]

    # Build MultiTSDataset
    multits = MultiTSDataset(input_mode=args.input_mode, method=args.method)
    multits.from_dataset(dataset, task_id=args.task_id, args=args)

    # Convert to HF Dataset and optionally subsample
    hf_ds = multits.to_hf_dataset()
    if args.num_samples is not None and args.num_samples < len(multits):
        rng = random.Random(args.random_seed)
        inds = sorted(rng.sample(range(len(multits)), args.num_samples))
        hf_ds = hf_ds.select(inds)

    dataloader = DataLoader(hf_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Run evaluation
    if "icl_ucr" in args.task_id.lower():
        results, input_output = run_evaluation_icl_ucr(model, dataloader, args)
    else:
        raise ValueError(f"Unsupported task: {args.task_id}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    method_slug = args.method.replace("/", "_").replace(".", "v")
    out_file = f"outputs/evaluation_results_{args.task_id}_{args.num_samples}_{method_slug}_exp_{args.exp_id}.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "metrics": results, "sample_count": len(input_output["questions"])}, f, indent=2)

    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Results saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_eval()
```

---

## Step 13 — SLURM Scripts

### `icl_experiment_scripts/run_single_task_gpu.sh`

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:45:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

conda activate multits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval.py "$@"
```

### `icl_experiment_scripts/run_single_task_gpu_large.sh`

For 27B+ models (vLLM, 2× GPUs):

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:30:00
#SBATCH --gpus=rtx_6000:2
#SBATCH --mem=128G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

conda activate multits_large
module load cuda/12.4
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python eval.py "$@"
```

### `icl_experiment_scripts/run_single_task_cpu.sh`

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

conda activate multits

python eval.py "$@"
```

### `icl_experiment_scripts/icl_ucr_comparison_full.sh`

```bash
#!/bin/bash
# Run all methods on all 94 UCR datasets across 8 seeds.
# Submits one SLURM job per (method, seed, dataset) triplet.

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/your/model/cache"
project="your-wandb-org/multits"
exp_id="random_k1_comparison_full"
strategy="random"
k_shots=1

llm_methods=("Qwen/Qwen3-4B-Instruct-2507")
large_llm_methods=("Qwen/Qwen3.6-27B")
baseline_methods=("random_baseline" "knn_baseline")

seeds=(0 1 2 3 4 5 6 7)

tasks=(
    "icl_ucr_GunPoint"
    "icl_ucr_Beef"
    "icl_ucr_ECG200"
    # ... add all 94 datasets from UCR_ICL_BENCHMARK.md
)

# GPU LLMs
for seed in "${seeds[@]}"; do
    for method in "${llm_methods[@]}"; do
        for task in "${tasks[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_task_gpu.sh" \
                --cache_dir "$cache_dir" --method "$method" \
                --use_wandb 1 --batch_size 1 --project "$project" \
                --exp_id "$exp_id" --picking_strategy "$strategy" \
                --num_shots "$k_shots" --num_samples 250 \
                --random_seed "$seed" --task_id "$task"
        done
    done
done

# Large LLMs
for seed in "${seeds[@]}"; do
    for method in "${large_llm_methods[@]}"; do
        for task in "${tasks[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_task_gpu_large.sh" \
                --cache_dir "$cache_dir" --method "$method" \
                --use_wandb 1 --batch_size 1 --project "$project" \
                --exp_id "$exp_id" --picking_strategy "$strategy" \
                --num_shots "$k_shots" --num_samples 250 \
                --random_seed "$seed" --task_id "$task"
        done
    done
done

# CPU Baselines
for seed in "${seeds[@]}"; do
    for method in "${baseline_methods[@]}"; do
        for task in "${tasks[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_task_cpu.sh" \
                --cache_dir "$cache_dir" --method "$method" \
                --use_wandb 1 --batch_size 1 --project "$project" \
                --exp_id "$exp_id" --picking_strategy "$strategy" \
                --num_shots "$k_shots" --num_samples 250 \
                --random_seed "$seed" --task_id "$task"
        done
    done
done
```

---

## Step 14 — Dependencies

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0          # for arff loading
numpy>=1.24.0
tqdm
tslearn                # for KNNBaseline DTW
bitsandbytes           # for 8bit/4bit quantization
vllm>=0.4.0            # for LargeInstructModel (optional, only for 27B+)
wandb                  # optional experiment tracking
```

Recommended: two separate conda environments:
- `multits` — for small models on RTX 4090
- `multits_large` — for large models via vLLM on multi-GPU nodes (requires newer `transformers` and `vllm`)

---

## Step 15 — Verification

### Smoke test (no GPU needed)

```bash
python eval.py \
    --task_id icl_ucr_GunPoint \
    --method random_baseline \
    --data_path /path/to/UCR/parent_dir \
    --num_samples 50 \
    --random_seed 0 \
    --use_wandb 0
```

**Expected:** balanced_accuracy ≈ 0.50 (random chance on binary GunPoint). A result file appears in `outputs/`.

### KNN baseline test

```bash
python eval.py \
    --task_id icl_ucr_GunPoint \
    --method knn_baseline \
    --data_path /path/to/UCR/parent_dir \
    --num_samples 50 \
    --random_seed 0 \
    --use_wandb 0
```

**Expected:** balanced_accuracy ≥ 0.85 (KNN is a strong baseline on GunPoint).

### LLM test (requires GPU)

```bash
python eval.py \
    --task_id icl_ucr_GunPoint \
    --method Qwen/Qwen3-4B-Instruct-2507 \
    --data_path /path/to/UCR/parent_dir \
    --cache_dir /your/model/cache \
    --num_samples 20 \
    --random_seed 0 \
    --batch_size 1 \
    --use_wandb 0
```

**Expected:** No errors. balanced_accuracy printed at the end. Result JSON in `outputs/`.

### Debug a single prompt

To inspect the prompt a model receives, add this after building `multits`:

```python
sample = multits[0]
print(sample["input_text"])
print("Gold label:", sample["output_text"])
```

---

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'icl_ucr_GunPoint'` in `load_dataset_for_task` | The return dict key must match `task_id` exactly | Return `{task_id: ...}` not `{"train": ..., "test": ...}` |
| All predictions are `INVALID_PREDICTION` | Options in the prompt are integers but `_extract_predicted_label` is matching strings | Cast `options` to `str` in `add_sample()` |
| `options` field in `add_sample` is `[1, 2]` (ints) but `gold_answers` are `"1"`, `"2"` (strings) | Type mismatch | Always use `[str(o) for o in options]` and `str(label_val)` |
| ARFF loading error: byte strings in labels | `scipy.io.arff` returns byte strings for non-numeric attributes | Cast with `.astype(str)` then strip `b''` prefix, or ensure labels are numeric in the ARFF |
| `OOM` on RTX 4090 with 8B model | Context window too long at k=1 | Reduce `--max_seq_length` or use `--quantization 8bit` |
| Qwen3 response contains `<think>...</think>` | Thinking mode not disabled | Pass `enable_thinking=False` to `apply_chat_template`; strip `</think>` split as fallback |
| Reproducibility broken | `random` module state shared across calls | Always use `random.Random(seed)` instances, not the global `random.seed()` |
