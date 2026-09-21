"""Query-time retrieval.

Pipeline: embed query → FAISS top-k (over-fetched) → similarity threshold →
metadata filter → near-duplicate removal → optional cross-encoder rerank →
truncate to ``top_k`` → assign ranks.

Why each stage exists
---------------------
* **Configurable top-k** – trades context size (cost, noise) against recall.
* **Similarity threshold** – stops weakly related chunks from being presented
  as evidence when the index simply does not contain an answer; this is what
  lets the assistant say "insufficient evidence" instead of guessing.
* **Metadata filtering** – scopes the search to chosen modalities / files.
* **Deduplication** – overlapping chunks and re-ingested files produce near
  identical passages; showing the same text twice wastes context and inflates
  the evidence indicator.
* **Optional reranking** – a cross-encoder re-scores the shortlist for
  precision; off by default because it needs an extra model.
"""

from __future__ import annotations

import re

from src.embeddings.base import Embedder
from src.retrieval.filters import RetrievalFilters
from src.retrieval.reranker import Reranker
from src.schemas import RetrievedChunk
from src.utils.logging import get_logger
from src.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)

_WORD = re.compile(r"\w+")


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def is_near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    """Jaccard similarity of word sets above ``threshold`` (or containment)."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return a.strip() == b.strip()
    inter = len(ta & tb)
    if inter / len(ta | tb) >= threshold:
        return True
    smaller = min(len(ta), len(tb))
    return inter / smaller >= 0.95  # one chunk almost entirely contained in the other


class Retriever:
    """Top-k semantic retrieval with quality controls."""

    def __init__(
        self,
        embedder: Embedder,
        store: FaissVectorStore,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        reranker: Reranker | None = None,
        overfetch: int = 3,
        dedup_threshold: float = 0.85,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.reranker = reranker
        self.overfetch = max(1, overfetch)
        self.dedup_threshold = dedup_threshold

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for ``query``, best first."""
        k = top_k or self.top_k
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        use_rerank = self.reranker is not None and (rerank if rerank is not None else True)
        if not query.strip() or self.store.is_empty:
            return []

        query_vector = self.embedder.embed_query(query)
        metadata_filter = None if (filters is None or filters.is_empty) else filters.matches
        raw = self.store.search(
            query_vector, k=k * self.overfetch, metadata_filter=metadata_filter, overfetch=4
        )

        above = [(chunk, score) for chunk, score in raw if score >= threshold]
        deduped: list[tuple[str, float]] = []
        kept: list[RetrievedChunk] = []
        for chunk, score in above:
            if any(is_near_duplicate(chunk.text, text, self.dedup_threshold) for text, _ in deduped):
                continue
            deduped.append((chunk.text, score))
            kept.append(RetrievedChunk(chunk=chunk, score=score, rank=len(kept) + 1))

        if use_rerank and kept:
            kept = self.reranker.rerank(query, kept)  # type: ignore[union-attr]

        results = [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(kept[:k])]
        logger.info(
            "Retrieved %d/%d candidates above threshold %.2f (dedup removed %d, k=%d, rerank=%s)",
            len(results), len(raw), threshold, len(above) - len(kept) if not use_rerank else len(above) - len(deduped), k, use_rerank,
        )
        return results
