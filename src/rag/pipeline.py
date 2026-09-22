"""End-to-end query pipeline: multimodal inputs → retrieval → grounded answer."""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.audio.loader import AudioClip
from src.audio.transcriber import AudioDisabledError, Transcriber
from src.rag.context import ContextAssembler
from src.rag.evidence import compute_evidence
from src.rag.generator import AnswerGenerator
from src.rag.ingestion import IngestionPipeline
from src.retrieval.filters import RetrievalFilters
from src.retrieval.retriever import Retriever
from src.schemas import ImageAnalysis, RAGResponse, Transcript
from src.utils.logging import get_logger
from src.utils.timing import Timer
from src.vision.analyzer import ImageAnalyzer
from src.vision.processor import analysis_to_document

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates transcription, image analysis, retrieval and generation."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        image_analyzer: ImageAnalyzer | None = None,
        transcriber: Transcriber | None = None,
        ingestion: IngestionPipeline | None = None,
        context_assembler: ContextAssembler | None = None,
        max_image_query_chars: int = 600,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.image_analyzer = image_analyzer
        self.transcriber = transcriber
        self.ingestion = ingestion
        self.context = context_assembler or ContextAssembler()
        self.max_image_query_chars = max_image_query_chars

    # --- Public API -------------------------------------------------------

    def answer(
        self,
        question: str | None = None,
        *,
        image: Image.Image | None = None,
        image_source: str = "uploaded_image",
        audio: AudioClip | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        rerank: bool | None = None,
        index_image: bool = False,
        include_prompt: bool = False,
    ) -> RAGResponse:
        """Answer a question given any combination of text, image and audio.

        Args:
            question: Typed question (may be empty when ``audio`` carries it).
            image: Attached image (PIL). Analysed by the vision backend and, when the
                generator supports images, also passed to it directly.
            image_source: Display name for the image (filename).
            audio: Spoken question; transcribed and appended to ``question``.
            top_k / similarity_threshold / filters / rerank: retrieval overrides.
            index_image: Also add the image analysis to the vector index.
            include_prompt: Return the assembled prompt for inspection.

        Raises:
            ValueError: if neither text nor audio provides a question.
            AudioDisabledError: if audio is given but no transcriber exists.
        """
        timer = Timer()
        transcript = self._transcribe(audio, timer)
        full_question = self._merge_question(question, transcript)

        image_analysis = self._analyse_image(image, image_source, timer)
        if image_analysis is not None and index_image and self.ingestion is not None:
            with timer.track("index_image"):
                self.ingestion.index_document(analysis_to_document(image_analysis))

        retrieval_query = self._retrieval_query(full_question, image_analysis)
        with timer.track("retrieval"):
            retrieved = self.retriever.retrieve(
                retrieval_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters,
                rerank=rerank,
            )

        attach_image = image is not None and self.generator.supports_images
        assembled = self.context.build(
            full_question,
            retrieved,
            image_analysis=image_analysis,
            transcript=transcript,
            image_attached_to_model=attach_image,
        )

        with timer.track("generation"):
            answer_text = self.generator.generate(
                assembled.prompt,
                images=[image] if attach_image else None,
                retrieved=assembled.included,
                question=full_question,
                image_analysis=image_analysis,
            )

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.retriever.similarity_threshold
        )
        evidence = compute_evidence(
            assembled.included,
            similarity_threshold=threshold,
            lexical_backend=self.retriever.embedder.is_fallback,
        )
        logger.info(
            "Answered query (%d chars) with %d passages, evidence=%s, backend=%s, %.0f ms",
            len(full_question),
            len(assembled.included),
            evidence.level.value,
            self.generator.name,
            timer.total_ms,
        )
        return RAGResponse(
            query=full_question,
            answer=answer_text,
            citations=assembled.citations,
            retrieved=assembled.included,
            evidence=evidence,
            image_analysis=image_analysis,
            transcript=transcript,
            generator_backend=self.generator.name,
            is_fallback_generation=self.generator.is_fallback,
            timings_ms=timer.timings_ms,
            prompt=assembled.prompt if include_prompt else None,
        )

    # --- Steps ------------------------------------------------------------

    def _transcribe(self, audio: AudioClip | None, timer: Timer) -> Transcript | None:
        if audio is None:
            return None
        if self.transcriber is None:
            raise AudioDisabledError("Audio input given but no speech backend is configured")
        with timer.track("transcription"):
            return self.transcriber.transcribe(audio)

    @staticmethod
    def _merge_question(question: str | None, transcript: Transcript | None) -> str:
        parts = [
            p.strip()
            for p in (question or "", transcript.text if transcript else "")
            if p and p.strip()
        ]
        if not parts:
            raise ValueError("Provide a question as text or as audio")
        return " ".join(parts)

    def _analyse_image(
        self, image: Image.Image | None, source: str, timer: Timer
    ) -> ImageAnalysis | None:
        if image is None:
            return None
        if self.image_analyzer is None:
            raise RuntimeError("Image given but no vision backend is configured")
        with timer.track("image_analysis"):
            return self.image_analyzer.analyze(image, source)

    def _retrieval_query(self, question: str, image_analysis: ImageAnalysis | None) -> str:
        """Enrich the retrieval query with the image's content so related docs surface.

        The question alone ("explain this architecture") carries little
        retrievable signal; the visible text and description of the image do.
        """
        if image_analysis is None or image_analysis.is_fallback:
            return question
        extra = " ".join(
            p for p in (image_analysis.extracted_text, image_analysis.description) if p
        ).strip()
        if not extra:
            return question
        return f"{question} {extra[: self.max_image_query_chars]}"

    def describe_backends(self) -> dict[str, Any]:
        return {
            "embedder": self.retriever.embedder.name,
            "embedder_fallback": self.retriever.embedder.is_fallback,
            "generator": self.generator.name,
            "generator_fallback": self.generator.is_fallback,
            "vision": getattr(self.image_analyzer, "name", None),
            "vision_fallback": getattr(self.image_analyzer, "is_fallback", None),
            "transcriber": getattr(self.transcriber, "name", None),
            "reranker": getattr(self.retriever.reranker, "name", None),
        }
