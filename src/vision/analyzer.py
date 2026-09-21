"""Image understanding backends.

:class:`ImageAnalyzer` is the interface. Implementations:

* :class:`QwenVLAnalyzer` – local Qwen2-VL / Qwen2.5-VL via ``transformers``.
* :class:`OpenAICompatibleAnalyzer` – any OpenAI-style vision endpoint (vLLM, Ollama, DashScope).
* :class:`MetadataImageAnalyzer` – **offline fallback**: records only what can be read
  without a model (dimensions, format, EXIF). It is explicitly flagged as a fallback and
  never claims to describe image content.

Design decision – what gets embedded for an image
-------------------------------------------------
A raw caption ("a diagram") retrieves poorly. Instead every analyzer produces a
structured :class:`ImageAnalysis` (kind, dense description, verbatim visible text,
objects) and :meth:`ImageAnalysis.to_searchable_text` renders it into labelled
sections. Visible text is kept verbatim because for documents, screenshots, charts
and diagrams it carries the entities and numbers users actually search for; the
description adds relationships that OCR alone would miss.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from src.config import Settings
from src.models.errors import ModelUnavailableError
from src.schemas import ImageAnalysis
from src.utils.logging import get_logger
from src.vision.prompts import ANALYSIS_PROMPT, ANALYSIS_SYSTEM_PROMPT, VQA_SYSTEM_PROMPT

logger = get_logger(__name__)

VALID_KINDS = {"document", "diagram", "chart", "table", "screenshot", "photo", "other"}


@runtime_checkable
class ImageAnalyzer(Protocol):
    """Interface for image understanding backends."""

    name: str
    is_fallback: bool

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        """Produce a structured, searchable analysis of ``image``."""
        ...

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        """Answer a free-form question about ``image`` (visual question answering)."""
        ...


# --- Shared parsing ---------------------------------------------------------


def parse_analysis_json(raw: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply; tolerate code fences and prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"description": text}


def analysis_from_reply(raw: str, image: Image.Image, source: str, backend: str) -> ImageAnalysis:
    """Build an :class:`ImageAnalysis` from a (possibly malformed) model reply."""
    data = parse_analysis_json(raw)
    kind = str(data.get("image_kind", "") or "").strip().lower() or None
    if kind not in VALID_KINDS:
        kind = None
    objects_raw = data.get("objects", [])
    if isinstance(objects_raw, str):
        objects = [o.strip() for o in objects_raw.split(",") if o.strip()]
    elif isinstance(objects_raw, list):
        objects = [str(o).strip() for o in objects_raw if str(o).strip()]
    else:
        objects = []
    visible = data.get("visible_text", "")
    if isinstance(visible, list):
        visible = "\n".join(str(v) for v in visible)
    return ImageAnalysis(
        source=source,
        description=str(data.get("description", "") or "").strip(),
        extracted_text=str(visible or "").strip(),
        objects=objects,
        image_kind=kind,
        width=image.width,
        height=image.height,
        backend=backend,
        is_fallback=False,
    )


def _vqa_prompt(question: str, context: str | None) -> str:
    if context:
        return (
            "Use the image and, where relevant, the following retrieved notes to answer.\n\n"
            f"Retrieved notes:\n{context}\n\nQuestion: {question}"
        )
    return question


# --- Backends ---------------------------------------------------------------


class QwenVLAnalyzer:
    """Local Qwen-VL through Hugging Face ``transformers``."""

    name = "qwen_vl"
    is_fallback = False

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 512) -> None:
        from src.models.qwen_vl_loader import load_qwen_vl

        self.bundle = load_qwen_vl(model_name, device)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        raw = self.bundle.generate(
            ANALYSIS_PROMPT,
            images=[image],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
        )
        return analysis_from_reply(raw, image, source, f"{self.name}:{self.model_name}")

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return self.bundle.generate(
            _vqa_prompt(question, context),
            images=[image],
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            system_prompt=VQA_SYSTEM_PROMPT,
        )


class OpenAICompatibleAnalyzer:
    """Vision model behind an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai_compatible"
    is_fallback = False

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_new_tokens: int = 512,
    ) -> None:
        from src.models.openai_client import OpenAICompatibleClient

        self.client = OpenAICompatibleClient(base_url, api_key=api_key, model=model, timeout_s=timeout_s)
        self.model_name = model
        self.max_new_tokens = max_new_tokens

    def _chat(self, system: str, prompt: str, image: Image.Image, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.client.build_user_content(prompt, [image])},
        ]
        return self.client.chat(messages, max_tokens=self.max_new_tokens, temperature=temperature)

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        raw = self._chat(ANALYSIS_SYSTEM_PROMPT, ANALYSIS_PROMPT, image, 0.0)
        return analysis_from_reply(raw, image, source, f"{self.name}:{self.model_name}")

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return self._chat(VQA_SYSTEM_PROMPT, _vqa_prompt(question, context), image, 0.1)


class MetadataImageAnalyzer:
    """Offline fallback: no vision model. Records only file-level facts.

    The resulting text is still indexable (filename, dimensions, EXIF fields)
    so a user can retrieve an image by name, but the analysis is flagged with
    ``is_fallback=True`` and the description states plainly that no model saw
    the image.
    """

    name = "metadata_only"
    is_fallback = True

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        exif_bits: list[str] = []
        try:
            exif = image.getexif()
            for tag, label in ((0x010F, "camera make"), (0x0110, "camera model"), (0x9003, "captured")):
                if tag in exif:
                    exif_bits.append(f"{label}: {exif[tag]}")
        except Exception:  # pragma: no cover - EXIF parsing is best effort
            pass
        description = (
            f"No vision-language model available: image '{source}' was NOT analysed for content. "
            f"Size {image.width}x{image.height} px, mode {image.mode}."
        )
        if exif_bits:
            description += " " + "; ".join(exif_bits) + "."
        logger.warning("Vision fallback active: %s indexed by metadata only", source)
        return ImageAnalysis(
            source=source,
            description=description,
            extracted_text="",
            objects=[],
            image_kind=None,
            width=image.width,
            height=image.height,
            backend=self.name,
            is_fallback=True,
        )

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return (
            "No vision-language model is configured, so the image content cannot be analysed. "
            "Set MRAG_VISION_BACKEND to 'transformers' or 'openai_compatible' to enable image "
            "question answering."
        )


def build_image_analyzer(settings: Settings) -> ImageAnalyzer:
    """Instantiate the configured vision backend.

    Raises:
        ModelUnavailableError: when a neural backend is configured but cannot load.
    """
    backend = settings.vision_backend
    if backend == "transformers":
        return QwenVLAnalyzer(settings.vision_model, settings.device, settings.vision_max_new_tokens)
    if backend == "openai_compatible":
        return OpenAICompatibleAnalyzer(
            settings.openai_base_url,
            settings.vision_model,
            api_key=settings.openai_api_key,
            timeout_s=settings.openai_timeout_s,
            max_new_tokens=settings.vision_max_new_tokens,
        )
    if backend == "metadata":
        logger.warning("Vision backend 'metadata': images will NOT be understood by a model")
        return MetadataImageAnalyzer()
    raise ModelUnavailableError(f"Unknown vision backend: {backend}")
