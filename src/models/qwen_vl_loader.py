"""Load Qwen2-VL / Qwen2.5-VL weights via Hugging Face ``transformers``.

The same loaded bundle is shared by the vision analyzer and the answer
generator when both are configured with the same model name, so the weights
are only held in memory once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.models.errors import ModelUnavailableError
from src.utils.device import preferred_dtype, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class QwenVLBundle:
    """A loaded vision-language model and its processor."""

    model: Any
    processor: Any
    device: str
    model_name: str

    def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ) -> str:
        """Run one chat-style generation with optional images.

        This is model mechanics (chat template, tensor plumbing, decoding),
        not business logic; prompts are composed by callers.
        """
        import torch

        content: list[dict[str, Any]] = [{"type": "image"} for _ in (images or [])]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(self.model.device)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0].strip() if decoded else ""


@lru_cache(maxsize=2)
def load_qwen_vl(model_name: str, device_preference: str = "auto") -> QwenVLBundle:
    """Load a Qwen-VL style image-text-to-text model.

    Works with ``Qwen/Qwen2-VL-*-Instruct`` and ``Qwen/Qwen2.5-VL-*-Instruct``
    (any model supported by ``AutoModelForImageTextToText``).

    Raises:
        ModelUnavailableError: if dependencies or weights are unavailable.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ModelUnavailableError(
            f"Cannot load '{model_name}': torch/transformers not installed ({exc})"
        ) from exc

    device = resolve_device(device_preference)
    dtype = preferred_dtype(device)
    logger.info("Loading vision-language model %s on %s (%s)", model_name, device, dtype)
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, device_map=device if device != "cpu" else None
        )
        if device == "cpu":
            model.to(torch.device("cpu"))
    except Exception as exc:
        raise ModelUnavailableError(
            f"Cannot load '{model_name}': {type(exc).__name__}: {exc}. Check the model name, "
            "network access to huggingface.co and available memory, or switch the backend "
            "(MRAG_VISION_BACKEND / MRAG_LLM_BACKEND) to 'openai_compatible' or a fallback."
        ) from exc
    model.eval()
    return QwenVLBundle(model=model, processor=processor, device=device, model_name=model_name)


def clear_model_cache() -> None:
    """Release cached models (used by tests and the API shutdown hook)."""
    load_qwen_vl.cache_clear()
