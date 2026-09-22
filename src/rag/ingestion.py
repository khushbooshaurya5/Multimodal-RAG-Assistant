"""Ingestion pipeline: file → Document → chunks → embeddings → FAISS.

Routing by extension:

* ``.txt .md .pdf ...`` → :class:`TextExtractor`
* images → :class:`ImageProcessor` (Qwen-VL analysis becomes searchable text)
* audio → :class:`AudioProcessor` (Whisper transcript with timestamps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.audio.loader import AudioLoader
from src.audio.processor import AudioProcessor
from src.audio.transcriber import AudioDisabledError
from src.embeddings.base import Embedder
from src.models.errors import ModelUnavailableError
from src.schemas import Chunk, ContentType, Document
from src.text.chunker import TextChunker
from src.text.extractor import TextExtractor
from src.utils.logging import get_logger
from src.utils.timing import Timer
from src.vectorstore.faiss_store import FaissVectorStore
from src.vision import supports as supports_image
from src.vision.processor import ImageProcessor

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    """Outcome of ingesting one file."""

    source: str
    doc_id: str | None
    content_type: ContentType | None
    num_chunks: int
    skipped: bool = False
    error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    document: Document | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class IngestionPipeline:
    """Extract, chunk, embed and index files of any supported modality."""

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        chunker: TextChunker,
        *,
        image_processor: ImageProcessor | None = None,
        audio_processor: AudioProcessor | None = None,
        text_extractor: TextExtractor | None = None,
        index_dir: Path | None = None,
        embed_batch_size: int = 64,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.text_extractor = text_extractor or TextExtractor()
        self.index_dir = index_dir
        self.embed_batch_size = embed_batch_size

    # --- Routing ----------------------------------------------------------

    def supports(self, path: Path) -> bool:
        path = Path(path)
        return (
            self.text_extractor.supports(path) or supports_image(path) or AudioLoader.supports(path)
        )

    def extract(self, path: Path) -> Document:
        """Turn a file into a :class:`Document` using the right modality processor."""
        path = Path(path)
        if self.text_extractor.supports(path):
            return self.text_extractor.extract(path)
        if supports_image(path):
            if self.image_processor is None:
                raise RuntimeError("No image processor configured")
            return self.image_processor.process_file(path)
        if AudioLoader.supports(path):
            if self.audio_processor is None or not self.audio_processor.enabled:
                raise AudioDisabledError(
                    f"Cannot ingest {path.name}: no speech-recognition backend configured"
                )
            return self.audio_processor.process_file(path)
        raise ValueError(f"Unsupported file type: {path.name}")

    # --- Indexing ---------------------------------------------------------

    def index_document(
        self, document: Document, *, save: bool = True
    ) -> tuple[list[Chunk], list[int]]:
        """Chunk, embed and add a document to the store (skips already-indexed docs)."""
        if self.store.contains_doc(document.doc_id):
            logger.info(
                "Document %s (%s) already indexed; skipping", document.source, document.doc_id
            )
            return [], []
        chunks = self.chunker.chunk_document(document)
        if not chunks:
            logger.warning("Document %s produced no chunks (empty content)", document.source)
            return [], []
        vectors = self._embed([c.text for c in chunks])
        ids = self.store.add(chunks, vectors)
        if save:
            self.save()
        return chunks, ids

    def ingest_file(self, path: Path, *, save: bool = True) -> IngestResult:
        """Ingest a single file, never raising: errors are returned in the result."""
        path = Path(path)
        timer = Timer()
        try:
            with timer.track("extract"):
                document = self.extract(path)
            if self.store.contains_doc(document.doc_id):
                return IngestResult(
                    source=path.name,
                    doc_id=document.doc_id,
                    content_type=document.content_type,
                    num_chunks=0,
                    skipped=True,
                    timings_ms=timer.timings_ms,
                    document=document,
                )
            with timer.track("chunk_embed_index"):
                chunks, _ = self.index_document(document, save=save)
            return IngestResult(
                source=path.name,
                doc_id=document.doc_id,
                content_type=document.content_type,
                num_chunks=len(chunks),
                timings_ms=timer.timings_ms,
                document=document,
            )
        except (AudioDisabledError, ModelUnavailableError, ValueError, FileNotFoundError) as exc:
            logger.error("Cannot ingest %s: %s", path.name, exc)
            return IngestResult(
                source=path.name,
                doc_id=None,
                content_type=None,
                num_chunks=0,
                error=f"{type(exc).__name__}: {exc}",
                timings_ms=timer.timings_ms,
            )
        except Exception as exc:
            logger.exception("Failed to ingest %s", path.name)
            return IngestResult(
                source=path.name,
                doc_id=None,
                content_type=None,
                num_chunks=0,
                error=f"{type(exc).__name__}: {exc}",
                timings_ms=timer.timings_ms,
            )

    def ingest_paths(self, paths: list[Path], *, save: bool = True) -> list[IngestResult]:
        """Ingest many files; the index is saved once at the end."""
        results = [self.ingest_file(p, save=False) for p in paths]
        if save and any(r.ok and not r.skipped for r in results):
            self.save()
        return results

    def ingest_directory(
        self, directory: Path, *, recursive: bool = True, save: bool = True
    ) -> list[IngestResult]:
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        paths = sorted(p for p in directory.glob(pattern) if p.is_file() and self.supports(p))
        logger.info("Ingesting %d supported files from %s", len(paths), directory)
        return self.ingest_paths(paths, save=save)

    def save(self) -> None:
        if self.index_dir is not None:
            self.store.save(self.index_dir)

    def _embed(self, texts: list[str]):
        import numpy as np

        batches = [
            self.embedder.embed_texts(texts[i : i + self.embed_batch_size])
            for i in range(0, len(texts), self.embed_batch_size)
        ]
        return (
            np.concatenate(batches, axis=0)
            if batches
            else np.zeros((0, self.embedder.dimension), dtype=np.float32)
        )
