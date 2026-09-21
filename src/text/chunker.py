"""Sentence-aware text chunking with overlap.

Why chunking matters for retrieval quality
------------------------------------------
* Embedding models have a limited effective context (MiniLM ≈ 256 word
  pieces); long passages get truncated or diluted.
* Small, focused chunks give sharper similarity scores and let the answer
  cite a precise location (page / timestamp).
* **Overlap** keeps sentences that straddle a chunk boundary retrievable from
  both neighbours so a fact split across the boundary is not lost.

The chunker works on characters (a language-agnostic proxy for tokens) and
prefers to split on sentence boundaries, then on whitespace, and only as a last
resort mid-word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas import Chunk, ChunkMetadata, ContentType, Document, TranscriptSegment
from src.utils.hashing import sha256_text
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-like units (paragraph breaks also split)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts


@dataclass(slots=True)
class TextChunker:
    """Split documents into overlapping chunks.

    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of trailing context carried into the next chunk.
        min_chunk_size: Chunks shorter than this are merged into the previous one.
    """

    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_size: int = 40

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    # --- Public API -------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Return overlapping chunk strings for ``text``."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        units = self._units(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit) + (1 if current else 0)
            if current and current_len + unit_len > self.chunk_size:
                chunks.append(" ".join(current))
                current, current_len = self._carry_overlap(current)
                unit_len = len(unit) + (1 if current else 0)
            current.append(unit)
            current_len += unit_len

        if current:
            tail = " ".join(current)
            if chunks and len(tail) < self.min_chunk_size:
                chunks[-1] = chunks[-1] + " " + tail
            else:
                chunks.append(tail)

        return self._dedupe_adjacent(chunks)

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a :class:`Document`, preserving page / timestamp metadata."""
        if document.content_type == ContentType.AUDIO and document.segments:
            chunks = self._chunk_segments(document, document.segments)
        elif document.pages:
            chunks = self._chunk_pages(document, document.pages)
        else:
            chunks = self._make_chunks(document, self.split_text(document.text))
        logger.info("Chunked %s into %d chunks", document.source, len(chunks))
        return chunks

    # --- Internals --------------------------------------------------------

    def _units(self, text: str) -> list[str]:
        """Sentence units, with oversized sentences split further on whitespace."""
        units: list[str] = []
        for sentence in split_sentences(text):
            if len(sentence) <= self.chunk_size:
                units.append(sentence)
                continue
            words = sentence.split(" ")
            buffer: list[str] = []
            length = 0
            for word in words:
                if len(word) > self.chunk_size:  # pathological token: hard split
                    if buffer:
                        units.append(" ".join(buffer))
                        buffer, length = [], 0
                    units.extend(
                        word[i : i + self.chunk_size] for i in range(0, len(word), self.chunk_size)
                    )
                    continue
                extra = len(word) + (1 if buffer else 0)
                if buffer and length + extra > self.chunk_size:
                    units.append(" ".join(buffer))
                    buffer, length = [], 0
                    extra = len(word)
                buffer.append(word)
                length += extra
            if buffer:
                units.append(" ".join(buffer))
        return units

    def _carry_overlap(self, current: list[str]) -> tuple[list[str], int]:
        """Keep trailing units whose total length fits within ``chunk_overlap``."""
        if self.chunk_overlap <= 0:
            return [], 0
        carried: list[str] = []
        length = 0
        for unit in reversed(current):
            extra = len(unit) + (1 if carried else 0)
            if length + extra > self.chunk_overlap:
                break
            carried.insert(0, unit)
            length += extra
        return carried, length

    @staticmethod
    def _dedupe_adjacent(chunks: list[str]) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            if out and (chunk == out[-1] or chunk in out[-1]):
                continue
            out.append(chunk)
        return out

    def _make_chunks(
        self,
        document: Document,
        texts: list[str],
        *,
        page: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        start_index: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for offset, text in enumerate(texts):
            index = start_index + offset
            chunk_id = f"{document.doc_id}:{index}:{sha256_text(text, 8)}"
            metadata = ChunkMetadata(
                source=document.source,
                content_type=document.content_type,
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                chunk_index=index,
                page=page,
                start_time=start_time,
                end_time=end_time,
                language=document.extra.get("language"),
                extra={k: v for k, v in document.extra.items() if k != "language"},
            )
            chunks.append(Chunk(text=text, metadata=metadata))
        return chunks

    def _chunk_pages(self, document: Document, pages: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page_number, page_text in enumerate(pages, start=1):
            texts = self.split_text(page_text)
            chunks.extend(
                self._make_chunks(document, texts, page=page_number, start_index=len(chunks))
            )
        return chunks

    def _chunk_segments(self, document: Document, segments: list[TranscriptSegment]) -> list[Chunk]:
        """Group timed transcript segments into chunks and keep their time span."""
        chunks: list[Chunk] = []
        group: list[TranscriptSegment] = []
        group_len = 0

        def flush() -> None:
            nonlocal group, group_len
            if not group:
                return
            text = " ".join(s.text.strip() for s in group if s.text.strip())
            if text:
                starts = [s.start for s in group if s.start is not None]
                ends = [s.end for s in group if s.end is not None]
                chunks.extend(
                    self._make_chunks(
                        document,
                        [text],
                        start_time=min(starts) if starts else None,
                        end_time=max(ends) if ends else None,
                        start_index=len(chunks),
                    )
                )
            group, group_len = [], 0

        for segment in segments:
            seg_len = len(segment.text) + 1
            if group and group_len + seg_len > self.chunk_size:
                flush()
            group.append(segment)
            group_len += seg_len
        flush()
        return chunks
