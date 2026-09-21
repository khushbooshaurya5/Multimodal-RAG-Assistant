"""Optional cross-encoder reranking.

Why: a bi-encoder (embedding) scores query and chunk independently, which is
fast but coarse. A cross-encoder reads the query *and* the chunk together and
is markedly better at judging fine-grained relevance. It is too slow to run
over the whole index, so it re-scores only the top candidates.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return candidates re-ordered by relevance with ``rerank_score`` set."""
        ...


class CrossEncoderReranker:
    """``sentence-transformers`` CrossEncoder (default ``ms-marco-MiniLM-L-6-v2``)."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        from src.models.embedding_loader import load_cross_encoder

        self.model = load_cross_encoder(model_name, device)
        self.name = f"cross_encoder:{model_name}"

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        rescored = [
            c.model_copy(update={"rerank_score": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return rescored
