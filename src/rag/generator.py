"""Answer generation backends.

* :class:`TransformersGenerator` – local Qwen-VL (shares weights with the vision analyzer).
* :class:`OpenAICompatibleGenerator` – OpenAI-style endpoint (vLLM / Ollama / DashScope ...).
* :class:`ExtractiveGenerator` – **offline fallback without any language model**. It
  selects the evidence sentences most lexically related to the question and returns
  them with citations. It cannot reason, paraphrase or synthesise, and it says so.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from src.config import Settings
from src.models.errors import ModelUnavailableError
from src.rag.prompts import SYSTEM_PROMPT
from src.schemas import ImageAnalysis, RetrievedChunk
from src.text.chunker import split_sentences
from src.utils.logging import get_logger

logger = get_logger(__name__)

_WORD = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been this that these those it its with as by "
    "from at what which who whom how why when where does do did can could would should about into "
    "than then there their they them we you your our i me my he she his her not no yes also".split()
)


@runtime_checkable
class AnswerGenerator(Protocol):
    """Interface for answer generation backends."""

    name: str
    is_fallback: bool
    supports_images: bool

    def generate(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
        retrieved: list[RetrievedChunk] | None = None,
        question: str | None = None,
        image_analysis: ImageAnalysis | None = None,
    ) -> str:
        """Produce the answer text for an assembled prompt."""
        ...


class TransformersGenerator:
    """Local Qwen-VL via ``transformers`` (text and image inputs)."""

    is_fallback = False
    supports_images = True

    def __init__(
        self, model_name: str, device: str = "auto", max_new_tokens: int = 512, temperature: float = 0.2
    ) -> None:
        from src.models.qwen_vl_loader import load_qwen_vl

        self.bundle = load_qwen_vl(model_name, device)
        self.model_name = model_name
        self.name = f"transformers:{model_name}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str, *, images: list[Any] | None = None, **_: Any) -> str:
        return self.bundle.generate(
            prompt,
            images=images,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            system_prompt=SYSTEM_PROMPT,
        )


class OpenAICompatibleGenerator:
    """Chat model behind an OpenAI-compatible endpoint."""

    is_fallback = False
    supports_images = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        from src.models.openai_client import OpenAICompatibleClient

        self.client = OpenAICompatibleClient(base_url, api_key=api_key, model=model, timeout_s=timeout_s)
        self.name = f"openai_compatible:{model}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str, *, images: list[Any] | None = None, **_: Any) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.client.build_user_content(prompt, images)},
        ]
        return self.client.chat(messages, max_tokens=self.max_new_tokens, temperature=self.temperature)


class ExtractiveGenerator:
    """Evidence-only answer without a language model (documented fallback).

    Algorithm: score every sentence of every retrieved passage by overlap of
    content words with the question (weighted by the passage's retrieval
    score), return the best sentences with their [n] citations, and prefix a
    notice explaining that no LLM produced the text.
    """

    name = "extractive"
    is_fallback = True
    supports_images = False
    NOTICE = (
        "⚠️ No language model is configured (MRAG_LLM_BACKEND=extractive). The text below is "
        "extracted verbatim from the retrieved evidence, not generated or reasoned about."
    )

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    @staticmethod
    def _content_words(text: str) -> set[str]:
        return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}

    def generate(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
        retrieved: list[RetrievedChunk] | None = None,
        question: str | None = None,
        image_analysis: ImageAnalysis | None = None,
    ) -> str:
        retrieved = retrieved or []
        question_words = self._content_words(question or "")
        scored: list[tuple[float, int, str]] = []
        for idx, item in enumerate(retrieved, start=1):
            for sentence in split_sentences(item.text):
                words = self._content_words(sentence)
                if not words:
                    continue
                overlap = len(words & question_words) / (len(question_words) or 1)
                score = overlap + 0.25 * max(item.score, 0.0)
                if overlap > 0 or not question_words:
                    scored.append((score, idx, sentence.strip()))
        scored.sort(key=lambda t: (-t[0], t[1]))

        lines: list[str] = [self.NOTICE, ""]
        if image_analysis is not None:
            if image_analysis.is_fallback:
                lines.append(
                    "Attached image: no vision model was available, so its content could not be analysed."
                )
            else:
                lines.append(f"Attached image analysis ({image_analysis.backend}): {image_analysis.description}")
                if image_analysis.extracted_text:
                    lines.append(f"Visible text in image: {image_analysis.extracted_text}")
            lines.append("")

        if not retrieved:
            lines.append("No retrieved evidence met the similarity threshold, so there is nothing to extract.")
            return "\n".join(lines).strip()
        if not scored:
            lines.append("The retrieved passages do not contain sentences overlapping with the question. "
                         "Evidence is insufficient to answer.")
            return "\n".join(lines).strip()

        lines.append("Most relevant evidence sentences:")
        seen: set[str] = set()
        for score, idx, sentence in scored:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {sentence} [{idx}]")
            if len(seen) >= self.max_sentences:
                break
        return "\n".join(lines).strip()


def build_generator(settings: Settings) -> AnswerGenerator:
    """Instantiate the configured answer generator.

    Raises:
        ModelUnavailableError: if a neural backend is configured but cannot load.
    """
    backend = settings.llm_backend
    if backend == "transformers":
        return TransformersGenerator(
            settings.llm_model, settings.device, settings.llm_max_new_tokens, settings.llm_temperature
        )
    if backend == "openai_compatible":
        return OpenAICompatibleGenerator(
            settings.openai_base_url,
            settings.llm_model,
            api_key=settings.openai_api_key,
            timeout_s=settings.openai_timeout_s,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
    if backend == "extractive":
        logger.warning("LLM backend 'extractive': answers are evidence excerpts, no language model")
        return ExtractiveGenerator()
    raise ModelUnavailableError(f"Unknown LLM backend: {backend}")
