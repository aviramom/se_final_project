"""Qwen3-VL-8B-Instruct model wrapper.

Converts each time-series array into a matplotlib chart image (annotated with
mean and ±1σ) and feeds the images + question text to Qwen3-VL's standard
vision-language pipeline.  This lets a general VLM reason about time series
visually without requiring a domain-specific TS encoder.

input_mode is "image": format_input splits the prompt on <TS_N> tokens,
renders each array as a PNG, and builds an interleaved content list
(text chunk → image → text chunk …) that the Qwen VL processor consumes.

Weights are expected on the Slurm cluster.  Point to them at runtime with the
QWEN_VL_MODEL_PATH env var; omit it to fall back to HuggingFace Hub download.
"""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fmeval.core.models.base import ModelWrapper
from fmeval.core.sample import Sample

if TYPE_CHECKING:
    from PIL import Image as PILImage

_TS_PLACEHOLDER_RE = re.compile(r"<TS_(\d+)>")


class QwenVLModel(ModelWrapper):
    """Wrapper for Qwen/Qwen3-VL-8B-Instruct.

    Heavy imports (torch, transformers, matplotlib, PIL) are deferred to their
    first use so that importing this module on a CPU-only machine does not fail.
    Only constructing the model AND calling predict() requires GPU + weights.
    format_input() only requires matplotlib + PIL (CPU-safe).
    """

    def __init__(
        self,
        checkpoint_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 100,
    ) -> None:
        """Store configuration; weights are loaded lazily on first predict().

        Args:
            checkpoint_path: Local directory with model weights or a
                HuggingFace Hub model ID.  On the cluster, set
                QWEN_VL_MODEL_PATH to a pre-downloaded path so the worker
                does not re-download on every job.
            max_new_tokens: Maximum tokens to generate per sample.
        """
        self._checkpoint_path = checkpoint_path
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = None

    def _load_if_needed(self) -> None:
        """Load model weights and processor on first inference call."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "torch and transformers>=4.45.0 are required to use QwenVLModel. "
                "Install them in the cluster environment."
            ) from exc

        import os
        local = os.path.isabs(self._checkpoint_path)
        if local and not os.path.isdir(self._checkpoint_path):
            raise FileNotFoundError(
                f"QwenVLModel: model directory not found: '{self._checkpoint_path}'.\n"
                f"Download it on the cluster:\n"
                f"  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('Qwen/Qwen3-VL-8B-Instruct', "
                f"local_dir='{self._checkpoint_path}')\""
            )

        self._model = AutoModelForImageTextToText.from_pretrained(
            self._checkpoint_path,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
            local_files_only=local,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self._checkpoint_path,
            trust_remote_code=True,
            local_files_only=local,
        )
        self._device = next(self._model.parameters()).device

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "qwen3-vl-8b"

    @property
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["multimodal"]

    @property
    def input_mode(self) -> Literal["combined", "separate", "image"]:
        return "image"

    def _plot_ts(self, ts: np.ndarray) -> "PILImage.Image":
        """Render one time-series array as an annotated PNG (PIL Image).

        The image shows: line plot of values vs. index, a dashed red horizontal
        line at the mean, and a shaded orange band spanning ±1σ.  Mean and std
        values are shown in the legend so the VLM can read them directly.
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, safe on headless servers
        import matplotlib.pyplot as plt
        from PIL import Image

        ts_flat = ts.flatten()
        mean = float(np.mean(ts_flat))
        std = float(np.std(ts_flat))
        indices = np.arange(len(ts_flat))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(indices, ts_flat, color="steelblue", linewidth=1.5, label="value")
        ax.axhline(
            mean,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={mean:.3g}",
        )
        ax.fill_between(
            indices,
            mean - std,
            mean + std,
            alpha=0.2,
            color="orange",
            label=f"±1σ={std:.3g}",
        )
        ax.set_xlabel("Index (t)")
        ax.set_ylabel("Value")
        ax.set_title("Time Series")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.load()  # force pixel decode before buffer is released
        buf.close()
        return img

    def format_input(self, sample: Sample) -> dict[str, Any]:
        """Build an interleaved content list: text chunks ↔ TS chart images.

        Each <TS_N> token is replaced by the PNG chart for sample.input_ts[N].
        The surrounding text is preserved in order.  The result is a Qwen VL
        message dict ready for processor.apply_chat_template.

        Returns:
            {"messages": [{"role": "user", "content": [...]}}]}
            where content entries alternate between {"type": "text", "text": ...}
            and {"type": "image", "image": <PIL.Image>}.
        """
        content: list[dict[str, Any]] = []
        last_end = 0

        for match in _TS_PLACEHOLDER_RE.finditer(sample.input_text):
            idx = int(match.group(1))
            # Text segment before this placeholder
            if match.start() > last_end:
                content.append(
                    {"type": "text", "text": sample.input_text[last_end : match.start()]}
                )
            # Plot the TS if the array exists (guard against out-of-range tokens)
            if idx < len(sample.input_ts):
                img = self._plot_ts(sample.input_ts[idx])
                content.append({"type": "image", "image": img})
            last_end = match.end()

        # Remaining text after the last placeholder (or entire text if no tokens)
        if last_end < len(sample.input_text):
            content.append({"type": "text", "text": sample.input_text[last_end:]})

        return {"messages": [{"role": "user", "content": content}]}

    def predict(self, inputs: list[dict[str, Any]]) -> list[str]:
        """Run vision-language inference on a list of formatted inputs.

        Processes each sample individually to handle variable numbers of images
        per prompt without padding complexity.

        Args:
            inputs: Output of format_input — each element is a dict with
                "messages" (list with one user turn containing text + image blocks).

        Returns:
            list[str]: Raw generated text per input.  Letter extraction for MCQ
            scoring is handled downstream by MCQMetrics.extract_letter().
        """
        self._load_if_needed()
        import torch
        from qwen_vl_utils import process_vision_info

        results: list[str] = []
        for inp in inputs:
            messages = inp["messages"]

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            proc_inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            input_len = proc_inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out_ids = self._model.generate(
                    **proc_inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )

            new_tokens = out_ids[0][input_len:]
            decoded = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(decoded)

        return results
