"""ChatTS-8B model wrapper.

Wraps bytedance-research/ChatTS-8B, a multimodal LLM that encodes raw
time-series arrays natively via a patch-based MLP encoder (Qwen3-8B backbone).

input_mode is "separate": the AutoProcessor consumes both the text prompt and
the raw numpy arrays directly, so we never need to serialize TS values into
text. The <TS_N> placeholders from the dataset are stripped before the prompt
reaches the processor.

Weights are expected on the Slurm cluster. Point to them at runtime with the
CHATTS_MODEL_PATH env var; omit it to fall back to HuggingFace Hub download.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from fmeval.core.datasets.formatters import SampleFormatter
from fmeval.core.models.base import ModelWrapper
from fmeval.core.sample import Sample

# Matches <TS_0>, <TS_1>, … placeholders emitted by all dataset classes.
_TS_PLACEHOLDER_RE = re.compile(r"<TS_\d+>")


class ChatTSModel(ModelWrapper):
    """Wrapper for bytedance-research/ChatTS-8B.

    Heavy imports (torch, transformers) are deferred to __init__ so that
    importing this module on a machine without GPU or transformers installed
    does not fail — only actually calling the constructor does.
    """

    def __init__(
        self,
        checkpoint_path: str = "bytedance-research/ChatTS-8B",
        max_new_tokens: int = 100,
    ) -> None:
        """Load model, tokenizer, and processor from checkpoint_path.

        Args:
            checkpoint_path: Local directory with model weights or a
                HuggingFace Hub model ID (e.g. "bytedance-research/ChatTS-8B").
                On the cluster, set CHATTS_MODEL_PATH to a pre-downloaded path
                so the worker does not re-download on every job.
            max_new_tokens: Maximum tokens to generate per sample. 100 is
                enough for MCQ answers; increase for open-ended generation.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required to use ChatTSModel. "
                "Install them in the cluster environment."
            ) from exc

        self._max_new_tokens = max_new_tokens

        self._model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )

        # Resolve the actual device after device_map="auto" placement.
        self._device = next(self._model.parameters()).device

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "chatts-8b"

    @property
    def supported_modalities(self) -> list[Literal["text", "time_series", "multimodal"]]:
        return ["multimodal"]

    @property
    def input_mode(self) -> Literal["combined", "separate"]:
        return "separate"

    def format_input(self, sample: Sample) -> dict[str, Any]:
        """Prepare one sample for ChatTS inference.

        Steps:
        1. Confirm canonical (separate) form via SampleFormatter.to_separate.
        2. Strip <TS_N> placeholders — the processor handles TS-text fusion
           internally and does not expect these tokens in the text input.
        3. Apply Qwen chat template so the instruction-tuned model sees the
           prompt in the format it was fine-tuned on.

        Returns a dict with "text" (formatted prompt string) and "timeseries"
        (list of raw numpy arrays), ready for batch assembly in predict().
        """
        sample = SampleFormatter.to_separate(sample)

        cleaned = _TS_PLACEHOLDER_RE.sub("[time series]", sample.input_text)

        messages = [{"role": "user", "content": cleaned}]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {"text": prompt, "timeseries": sample.input_ts}

    def predict(self, inputs: list[dict[str, Any]]) -> list[str]:
        """Run batched inference on a list of formatted inputs.

        Args:
            inputs: Output of format_input — each element is a dict with
                "text" (str) and "timeseries" (list[np.ndarray]).

        Returns:
            list[str]: Raw generated text per input (one string per sample).
            Letter extraction for MCQ scoring is handled downstream by
            MCQMetrics.extract_letter().
        """
        import torch

        texts = [inp["text"] for inp in inputs]
        timeseries_list = [inp["timeseries"] for inp in inputs]

        proc = self._processor(
            text=texts,
            timeseries=timeseries_list,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Qwen uses left-padding for batched generation, so all samples in the
        # batch are padded to the same length on the left. New tokens from
        # generate() are appended after the (padded) input sequence length.
        input_seq_len = proc["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **proc,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        results: list[str] = []
        for out in output_ids:
            new_tokens = out[input_seq_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(text)

        return results
