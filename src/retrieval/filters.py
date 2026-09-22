"""Metadata filters for retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas import ChunkMetadata, ContentType


class RetrievalFilters(BaseModel):
    """Restrict retrieval to a subset of the index.

    Why: users often know *where* the answer lives ("only in the PDFs",
    "only in this lecture recording"). Filtering removes competing chunks
    from other sources so top-k is spent on the relevant subset.
    """

    content_types: list[ContentType] | None = Field(None, description="Allowed modalities")
    sources: list[str] | None = Field(None, description="Allowed source filenames")
    doc_ids: list[str] | None = Field(None, description="Allowed document ids")

    @property
    def is_empty(self) -> bool:
        return not (self.content_types or self.sources or self.doc_ids)

    def matches(self, metadata: ChunkMetadata) -> bool:
        if self.content_types and metadata.content_type not in self.content_types:
            return False
        if self.sources and metadata.source not in self.sources:
            return False
        if self.doc_ids and metadata.doc_id not in self.doc_ids:
            return False
        return True
