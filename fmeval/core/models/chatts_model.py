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
        """Store configuration; weights are loaded lazily on first predict().

        Args:
            checkpoint_path: Local directory with model weights or a
                HuggingFace Hub model ID (e.g. "bytedance-research/ChatTS-8B").
                On the cluster, set CHATTS_MODEL_PATH to a pre-downloaded path
                so the worker does not re-download on every job.
            max_new_tokens: Maximum tokens to generate per sample. 100 is
                enough for MCQ answers; increase for open-ended generation.
        """
        self._checkpoint_path = checkpoint_path
        self._max_new_tokens = max_new_tokens
        # Loaded on first call to _load_if_needed() — keeps construction cheap
        # so the service can instantiate this object locally for compatibility
        # checks without requiring GPU or cluster paths to exist.
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device = None

    def _load_if_needed(self) -> None:
        """Load model weights, tokenizer, and processor on first inference call."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required to use ChatTSModel. "
                "Install them in the cluster environment."
            ) from exc

        self._model = AutoModelForCausalLM.from_pretrained(
            self._checkpoint_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._checkpoint_path,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._checkpoint_path,
            trust_remote_code=True,
        )
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
        self._load_if_needed()
        sample = SampleFormatter.to_separate(sample)

        # ChatTS processor splits on <ts><ts/> to interleave encoded arrays.
        cleaned = _TS_PLACEHOLDER_RE.sub("<ts><ts/>", sample.input_text)

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
        self._load_if_needed()
        import torch

        texts = [inp["text"] for inp in inputs]
        # Processor expects a flat list across all samples; it uses <ts><ts/>
        # placeholder counts per prompt to know how many arrays belong to each sample.
        timeseries_flat = [ts for inp in inputs for ts in inp["timeseries"]]

        proc = self._processor(
            text=texts,
            timeseries=timeseries_flat,
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
