"""Shared data models (Pydantic) passed between pipeline stages.

Keeping every cross-module payload here makes the interfaces between the
audio, vision, embedding, retrieval and generation layers explicit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(StrEnum):
    """Origin modality of a piece of indexed content."""

    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"


class ChunkMetadata(BaseModel):
    """Metadata retained for every indexed chunk (persisted next to the FAISS index)."""

    source: str = Field(description="Original filename or identifier")
    content_type: ContentType
    chunk_id: str = Field(description="Stable unique chunk identifier")
    doc_id: str = Field(description="Identifier of the parent document")
    chunk_index: int = Field(0, ge=0, description="Position of the chunk within the document")
    page: int | None = Field(None, description="1-based page number for paginated sources")
    start_time: float | None = Field(None, description="Audio segment start (seconds)")
    end_time: float | None = Field(None, description="Audio segment end (seconds)")
    language: str | None = None
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A unit of searchable text plus its metadata."""

    text: str
    metadata: ChunkMetadata


class Document(BaseModel):
    """Extracted content of one ingested file before chunking."""

    doc_id: str
    source: str
    content_type: ContentType
    text: str = Field(description="Full extracted text (transcript, VLM description, ...)")
    pages: list[str] | None = Field(None, description="Per-page text for paginated sources")
    segments: list[TranscriptSegment] | None = Field(
        None, description="Timed segments for audio sources"
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever together with its similarity score."""

    chunk: Chunk
    score: float = Field(description="Cosine similarity in [-1, 1]")
    rank: int = Field(ge=1)
    rerank_score: float | None = None

    @property
    def metadata(self) -> ChunkMetadata:
        return self.chunk.metadata

    @property
    def text(self) -> str:
        return self.chunk.text


# --- Audio -------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    """A timed span of a transcript."""

    text: str
    start: float | None = None
    end: float | None = None


class Transcript(BaseModel):
    """Result of speech recognition."""

    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    duration_s: float | None = None
    backend: str = Field(description="Which transcription backend produced this")


# --- Vision ------------------------------------------------------------------


class ImageAnalysis(BaseModel):
    """Structured understanding of one image produced by the vision backend."""

    source: str
    description: str = Field(description="Dense semantic description of the image")
    extracted_text: str = Field("", description="OCR-like text visible in the image")
    objects: list[str] = Field(default_factory=list)
    image_kind: str | None = Field(
        None, description="Coarse category: document, diagram, chart, table, photo, screenshot"
    )
    width: int | None = None
    height: int | None = None
    backend: str = Field(description="Which vision backend produced this")
    is_fallback: bool = Field(
        False, description="True when no vision-language model was used (metadata only)"
    )

    def to_searchable_text(self) -> str:
        """Compose the text that gets chunked and embedded for this image.

        Design decision: rather than embedding the raw caption only, we build a
        structured representation (kind, description, visible text, objects).
        Visible text is the most valuable retrieval signal for documents,
        diagrams and screenshots, so it is kept verbatim and labelled.
        """
        parts: list[str] = []
        if self.image_kind:
            parts.append(f"Image type: {self.image_kind}.")
        if self.description:
            parts.append(f"Description: {self.description.strip()}")
        if self.extracted_text.strip():
            parts.append(f"Visible text: {self.extracted_text.strip()}")
        if self.objects:
            parts.append("Objects: " + ", ".join(dict.fromkeys(self.objects)) + ".")
        return "\n".join(parts).strip()


# --- Generation --------------------------------------------------------------


class EvidenceLevel(StrEnum):
    """Coarse, *uncalibrated* indicator of how much retrieved evidence supports an answer."""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class EvidenceIndicator(BaseModel):
    """Heuristic evidence summary shown alongside an answer.

    This is derived from retrieval statistics (number of hits above the
    threshold, best score, score spread). It is **not** a calibrated
    probability that the answer is correct.
    """

    level: EvidenceLevel
    num_sources: int
    best_score: float | None
    mean_score: float | None
    note: str


class Citation(BaseModel):
    """A reference to a retrieved chunk displayed with the answer."""

    index: int = Field(ge=1, description="1-based citation number used in the answer")
    source: str
    content_type: ContentType
    chunk_id: str
    score: float
    page: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    snippet: str


class RAGResponse(BaseModel):
    """Final answer plus everything needed to make the RAG process transparent."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    evidence: EvidenceIndicator
    image_analysis: ImageAnalysis | None = None
    transcript: Transcript | None = None
    generator_backend: str
    is_fallback_generation: bool = Field(
        False, description="True if the answer came from the extractive (no-LLM) fallback"
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)
    prompt: str | None = Field(None, description="Prompt sent to the generator (for debugging)")


Document.model_rebuild()
