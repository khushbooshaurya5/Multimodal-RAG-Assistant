"""Load Whisper speech-recognition weights from Hugging Face ``transformers``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.errors import ModelUnavailableError
from src.utils.device import preferred_dtype, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class WhisperBundle:
    """A loaded Whisper model, its processor and the resolved device."""

    model: Any
    processor: Any
    device: str
    model_name: str


def load_whisper(model_name: str, device_preference: str = "auto") -> WhisperBundle:
    """Load a Whisper checkpoint (e.g. ``openai/whisper-small``).

    Raises:
        ModelUnavailableError: if ``transformers``/``torch`` are missing or the
            weights cannot be fetched (offline, blocked network, bad name).
    """
    try:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
    except ImportError as exc:
        raise ModelUnavailableError(
            f"Cannot load Whisper '{model_name}': torch/transformers not installed ({exc})"
        ) from exc

    device = resolve_device(device_preference)
    dtype = preferred_dtype(device)
    logger.info("Loading Whisper model %s on %s (%s)", model_name, device, dtype)
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    except Exception as exc:  # network errors, missing repo, corrupt cache ...
        raise ModelUnavailableError(
            f"Cannot load Whisper '{model_name}': {type(exc).__name__}: {exc}. "
            "Check the model name, network access to huggingface.co, or set "
            "MRAG_WHISPER_BACKEND=none to disable audio."
        ) from exc
    model.to(torch.device(device))
    model.eval()
    return WhisperBundle(model=model, processor=processor, device=device, model_name=model_name)
