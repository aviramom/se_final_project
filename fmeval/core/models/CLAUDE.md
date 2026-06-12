# fmeval/core/models/ — ModelWrapper abstractions

`ModelWrapper` is the ABC that hides every HuggingFace model's unique loading and
inference API behind a single interface. Adding a new model = one new file here + one
registry entry. Nothing else changes.

**All models are HuggingFace models.** No external REST APIs, no API keys.

**All models produce text output** — the research focus is multimodal TS+text tasks
where both LLMs and specialized multimodal models generate a natural-language answer.

---

**Status: two real models.** `base.py`, `mock_model.py`, `chatts_model.py`, and
`qwen_vl_model.py` are written and cluster-verified. Registry lives in
`fmeval/config/model_registry.py` (`build_default_model_registry()`), not a separate
`registry.py`.

---

## Files

```
models/
  CLAUDE.md
  __init__.py         ← ✅ exports: ModelWrapper, MockModel, ChatTSModel, QwenVLModel
  base.py             ← ✅ ModelWrapper ABC
  mock_model.py       ← ✅ MockModel (fixed-answer baseline for pipeline testing)
  chatts_model.py     ← ✅ ChatTSModel (bytedance-research/ChatTS-8B, input_mode="separate")
  qwen_vl_model.py    ← ✅ QwenVLModel (Qwen/Qwen3-VL-8B-Instruct, input_mode="image")
```

---

## ModelWrapper ABC (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Literal
from fmeval.core.sample import Sample

class ModelWrapper(ABC):
    """
    Wraps one HuggingFace model family. Handles model-specific loading,
    input formatting, and inference. Never knows about metrics or benchmarks.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier, e.g. 'llama3-8b'."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        """All current models declare ["multimodal"]."""
        ...

    @property
    @abstractmethod
    def input_mode(self) -> Literal["combined", "separate", "image"]:
        """
        How this model consumes a Sample's time series data.

        "combined"  — SampleFormatter.to_combined() is called before format_input;
                      the model receives a single text string with TS values inlined.
                      Use for standard LLMs (LLaMA, Mistral, etc.).

        "separate"  — Sample is passed as-is; the model receives both input_text
                      (with <TS_N> tokens) and input_ts (raw arrays) and handles
                      the fusion itself.
                      Use for models with a dedicated TS encoder (e.g. ChatTS).

        "image"     — format_input renders each TS array as a matplotlib chart (PNG)
                      and builds an interleaved content list (text ↔ images) for a
                      vision-language model. SampleFormatter is not called.
                      Use for VLMs like Qwen3-VL.
        """
        ...

    @abstractmethod
    def format_input(self, sample: Sample) -> Any:
        """
        Convert a Sample into the model's expected input format.
        Must call SampleFormatter based on self.input_mode before further formatting.
        Returns whatever structure predict() expects (string, tensor, dict, etc.).
        """
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> list[str]:
        """
        Run inference on a batch of formatted inputs (output of format_input).
        Always returns list[str] — all models output text.
        This is the code that runs on the cluster; in mock mode it runs locally.
        """
        ...
```

---

## Registry

Registration lives in `fmeval/config/model_registry.py` → `build_default_model_registry()`.
Adding a model = one new file here + one `registry.register(...)` call there.

---

## Concrete wrapper pattern

Each model file must:

1. Subclass `ModelWrapper`.
2. Accept `checkpoint_path: str` in `__init__` (local dir or HuggingFace repo id).
3. **Lazy weight loading** — `__init__` stores config only; implement `_load_if_needed()`
   that imports `torch`/`transformers` and loads weights, and call it at the top of both
   `format_input` and `predict`. This lets the service construct the model locally for
   compatibility checks without a GPU or cluster path on the dev machine.
4. Declare `input_mode` — `"combined"` for standard LLMs (TS serialized to text),
   `"separate"` for models with a dedicated TS encoder (raw arrays passed through),
   `"image"` for VLMs that receive TS as rendered chart images.
5. In `format_input`: handle the TS data according to `self.input_mode`. For `"combined"`
   and `"separate"` call the appropriate `SampleFormatter` method; for `"image"` plot
   each array and build an interleaved text/image content list.
6. Implement `predict(inputs: list[Any]) -> list[str]`.
7. **`model_name` must return a fully lowercase string.** It is passed as a CLI
   argument to `cluster_worker.py` on the cluster and looked up in the registry there —
   a case mismatch causes a `KeyError` at runtime.

---

## MockModel (`mock_model.py`) — **implemented**

```python
model = MockModel(answer="A")   # always returns "A)" for every input
model = MockModel(answer="B")   # always returns "B)"
```

`input_mode = "combined"` — calls `SampleFormatter.to_combined()` in `format_input`.
Used to verify the full pipeline end-to-end without loading any real weights.

---

## ChatTSModel (`chatts_model.py`) — **implemented & cluster-verified**

```python
model = ChatTSModel(checkpoint_path="/home/aviramom/models/chatts-8b")
model = ChatTSModel()  # falls back to HF Hub download
```

`input_mode = "separate"` — passes raw numpy arrays directly to `AutoProcessor`.

**Lazy weight loading.** `__init__` only stores `checkpoint_path`; weights, tokenizer,
and processor are loaded on the first call to `_load_if_needed()`, which is invoked at
the top of both `format_input` and `predict`. This keeps construction cheap so
`EvaluationService` can instantiate the model locally for compatibility checks without
touching the GPU or requiring cluster paths to exist on the dev machine.

**`format_input` details.** Replaces `<TS_N>` placeholders with `<ts><ts/>` — the
ChatTS processor's native token — then applies the Qwen chat template. Do **not** use
plain text like `[time series]`; the processor splits on `<ts><ts/>` to interleave
encoded arrays and will raise `ValueError` if the count mismatches.

**`predict` details.** Flattens the per-sample `timeseries` lists into one flat list
before passing to the processor (`[ts for inp in inputs for ts in inp["timeseries"]]`).
The processor uses the per-prompt `<ts><ts/>` count to slice the flat list correctly.
Runs `model.generate(do_sample=False)` and slices new tokens from the left-padded
output (Qwen uses left-padding for batched generation).

Checkpoint path is read from the `CHATTS_MODEL_PATH` env var at registry factory
time; set it in the Slurm launch command to point to a pre-downloaded path on the
cluster (`/home/aviramom/models/chatts-8b`). Without the env var the constructor
downloads from HuggingFace Hub (~16 GB).

**Local testing:** use `MockModel` (`FMEVAL_RUNNER=mock`). The module imports safely on
any machine; only the first `predict()` call loads weights, which only happens inside
the cluster worker.

---

---

## QwenVLModel (`qwen_vl_model.py`) — **implemented & cluster-verified**

```python
model = QwenVLModel(checkpoint_path="/home/aviramom/models/qwen3-vl-8b")
model = QwenVLModel()  # falls back to HF Hub download of Qwen/Qwen3-VL-8B-Instruct
```

`input_mode = "image"` — converts each time-series array to a matplotlib PNG chart
before calling the Qwen3-VL vision-language processor.

**TS → image conversion (`_plot_ts`).** Each `np.ndarray` is plotted as a line chart
with a dashed red mean line and an orange ±1σ shaded band. Mean and std values appear
in the legend so the model can read them directly. Uses the `Agg` (non-interactive)
matplotlib backend — safe on headless cluster nodes.

**`format_input` details.** Splits `input_text` on `<TS_N>` tokens and builds an
interleaved content list: text chunk → `{"type": "image", "image": <PIL.Image>}` →
text chunk → … The result is a `{"messages": [{"role": "user", "content": [...]}]}`
dict ready for `processor.apply_chat_template`.

**`predict` details.** Processes each sample individually (variable image counts per
prompt make batching complex). Uses `process_vision_info` from `qwen-vl-utils` to
extract image tensors, then `AutoProcessor` + `AutoModelForImageTextToText.generate`
with `do_sample=False`. Decodes only newly generated tokens.

**Local path handling.** `_load_if_needed` checks `os.path.isabs(checkpoint_path)`:
if True it passes `local_files_only=True` to avoid Hub repo-ID validation on absolute
paths. If the directory doesn't exist, a `FileNotFoundError` with a download hint is
raised before the Hub call.

**Prerequisites on the cluster.** Requires `torchvision` (for `Qwen3VLVideoProcessor`)
in addition to the standard `torch`/`transformers` stack.

Checkpoint path is read from the `QWEN_VL_MODEL_PATH` env var. Download once:
```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/home/aviramom/models/qwen3-vl-8b')"
```

---

## Public exports (`__init__.py`)

```python
from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.mock_model import MockModel
from fmeval.core.models.chatts_model import ChatTSModel
from fmeval.core.models.qwen_vl_model import QwenVLModel
```
