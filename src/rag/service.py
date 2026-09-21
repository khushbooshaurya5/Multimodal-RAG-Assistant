"""Application service: wires every component from :class:`Settings`.

Both the Streamlit UI and the FastAPI server use this single entry point so
model loading, fallbacks and index persistence behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.audio.loader import AudioClip, AudioLoader
from src.audio.processor import AudioProcessor
from src.audio.transcriber import Transcriber, build_transcriber
from src.config import Settings, get_settings
from src.embeddings.base import Embedder
from src.embeddings.factory import build_embedder
from src.embeddings.hashing import HashingEmbedder
from src.models.errors import ModelUnavailableError
from src.rag.generator import AnswerGenerator, ExtractiveGenerator, build_generator
from src.rag.ingestion import IngestionPipeline, IngestResult
from src.rag.pipeline import RAGPipeline
from src.retrieval.filters import RetrievalFilters
from src.retrieval.reranker import CrossEncoderReranker, Reranker
from src.retrieval.retriever import Retriever
from src.schemas import RAGResponse
from src.text.chunker import TextChunker
from src.utils.logging import get_logger
from src.vectorstore.faiss_store import FaissVectorStore
from src.vision.analyzer import ImageAnalyzer, MetadataImageAnalyzer, build_image_analyzer
from src.vision.processor import ImageProcessor

logger = get_logger(__name__)


@dataclass(slots=True)
class BackendStatus:
    """What actually got loaded, plus any degradation warnings."""

    embedder: str
    generator: str
    vision: str | None
    transcriber: str | None
    reranker: str | None
    device: str
    warnings: list[str] = field(default_factory=list)
    fallbacks: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "generator": self.generator,
            "vision": self.vision,
            "transcriber": self.transcriber,
            "reranker": self.reranker,
            "device": self.device,
            "warnings": list(self.warnings),
            "fallbacks": dict(self.fallbacks),
        }


class AssistantService:
    """Builds and owns the pipeline, index and processors for one process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._warnings: list[str] = []

        self.embedder: Embedder = self._build("embedder", lambda: build_embedder(self.settings), self._fallback_embedder)
        self.generator: AnswerGenerator = self._build("generator", lambda: build_generator(self.settings), ExtractiveGenerator)
        self.image_analyzer: ImageAnalyzer = self._build("vision", lambda: build_image_analyzer(self.settings), MetadataImageAnalyzer)
        self.transcriber: Transcriber | None = self._build("transcriber", lambda: build_transcriber(self.settings), lambda: None)
        self.reranker: Reranker | None = None
        if self.settings.rerank:
            self.reranker = self._build(
                "reranker",
                lambda: CrossEncoderReranker(self.settings.rerank_model, self.settings.device),
                lambda: None,
            )

        self.store = self._open_store()
        self.chunker = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.image_processor = ImageProcessor(self.image_analyzer)
        self.audio_processor = AudioProcessor(self.transcriber, AudioLoader())
        self.ingestion = IngestionPipeline(
            self.embedder,
            self.store,
            self.chunker,
            image_processor=self.image_processor,
            audio_processor=self.audio_processor,
            index_dir=self.settings.index_dir,
        )
        self.retriever = Retriever(
            self.embedder,
            self.store,
            top_k=self.settings.top_k,
            similarity_threshold=self.settings.similarity_threshold,
            reranker=self.reranker,
        )
        self.pipeline = RAGPipeline(
            self.retriever,
            self.generator,
            image_analyzer=self.image_analyzer,
            transcriber=self.transcriber,
            ingestion=self.ingestion,
        )
        logger.info("Assistant ready: %s", self.status().as_dict())

    # --- Construction helpers --------------------------------------------

    def _fallback_embedder(self) -> Embedder:
        return HashingEmbedder(self.settings.embedding_dim)

    def _build(self, label: str, primary, fallback):
        """Build a component; on :class:`ModelUnavailableError` degrade if allowed."""
        try:
            return primary()
        except ModelUnavailableError as exc:
            if not self.settings.fallback_on_error:
                raise
            message = f"{label}: {exc} → using offline fallback"
            logger.error(message)
            self._warnings.append(message)
            return fallback()

    def _open_store(self) -> FaissVectorStore:
        index_dir = self.settings.index_dir
        try:
            return FaissVectorStore.load_or_create(index_dir, self.embedder.dimension, self.embedder.name)
        except Exception as exc:
            message = (
                f"index: could not load {index_dir} ({exc}); starting with an empty in-memory index. "
                "Delete or rebuild the index directory to persist again."
            )
            logger.error(message)
            self._warnings.append(message)
            return FaissVectorStore(self.embedder.dimension, self.embedder.name)

    # --- Status -----------------------------------------------------------

    def status(self) -> BackendStatus:
        from src.utils.device import resolve_device

        return BackendStatus(
            embedder=self.embedder.name,
            generator=self.generator.name,
            vision=self.image_analyzer.name,
            transcriber=getattr(self.transcriber, "name", None),
            reranker=getattr(self.reranker, "name", None),
            device=resolve_device(self.settings.device),
            warnings=list(self._warnings),
            fallbacks={
                "embedder": self.embedder.is_fallback,
                "generator": self.generator.is_fallback,
                "vision": self.image_analyzer.is_fallback,
                "audio_disabled": self.transcriber is None,
            },
        )

    def index_stats(self) -> dict[str, Any]:
        return {
            "num_vectors": len(self.store),
            "dimension": self.store.dimension,
            "sources": self.store.sources(),
            "index_dir": str(self.settings.index_dir),
        }

    # --- Ingestion --------------------------------------------------------

    def ingest_files(self, paths: list[Path]) -> list[IngestResult]:
        return self.ingestion.ingest_paths([Path(p) for p in paths])

    def ingest_directory(self, directory: Path) -> list[IngestResult]:
        return self.ingestion.ingest_directory(Path(directory))

    def ingest_upload(self, filename: str, data: bytes) -> IngestResult:
        """Persist uploaded bytes under ``data/raw`` and ingest the file."""
        safe_name = Path(filename).name or "upload"
        target = self.settings.raw_dir / safe_name
        target.write_bytes(data)
        return self.ingestion.ingest_file(target)

    def remove_source(self, source: str) -> int:
        removed = self.store.remove_source(source)
        self.ingestion.save()
        return removed

    def clear_index(self) -> None:
        self.store.clear()
        self.ingestion.save()

    # --- Query ------------------------------------------------------------

    def transcribe(self, data: bytes, filename: str):
        return self.audio_processor.transcribe_bytes(data, filename)

    def load_audio(self, data: bytes, filename: str) -> AudioClip:
        return self.audio_processor.loader.load_bytes(data, filename)

    def query(
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
        return self.pipeline.answer(
            question,
            image=image,
            image_source=image_source,
            audio=audio,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
            rerank=rerank,
            index_image=index_image,
            include_prompt=include_prompt,
        )
