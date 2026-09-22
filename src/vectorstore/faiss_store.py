"""FAISS-backed vector store with metadata persistence.

Vectors are L2-normalised so an inner-product index (``IndexFlatIP``) yields
cosine similarity. ``IndexIDMap2`` wraps the flat index so every vector has a
stable integer id that also keys the :class:`MetadataStore`; this enables
removal by document and keeps ids valid across save/load.

On-disk layout (``<index_dir>/``)::

    index.faiss      the FAISS index
    metadata.jsonl   one JSON record per vector: {"id": int, "chunk": {...}}
    manifest.json    dimension, embedder name, counts, timestamps
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.schemas import Chunk, ChunkMetadata
from src.utils.logging import get_logger
from src.vectorstore.metadata_store import MetadataStore

logger = get_logger(__name__)

MetadataFilter = Callable[[ChunkMetadata], bool]


class VectorStoreError(RuntimeError):
    """Raised for invalid store operations (dimension mismatch, corrupt files)."""


class FaissVectorStore:
    """Exact cosine-similarity search over chunk embeddings with metadata.

    Args:
        dimension: Embedding size.
        embedder_name: Recorded in the manifest so a store built with one
            embedder is not silently queried with another.
    """

    INDEX_FILE = "index.faiss"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, dimension: int, embedder_name: str = "unknown") -> None:
        import faiss

        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.embedder_name = embedder_name
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))
        self.metadata = MetadataStore()
        self._next_id = 0
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at

    # --- Introspection ----------------------------------------------------

    def __len__(self) -> int:
        return int(self._index.ntotal)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def sources(self) -> dict[str, int]:
        return self.metadata.sources()

    def contains_doc(self, doc_id: str) -> bool:
        return bool(self.metadata.ids_for_doc(doc_id))

    # --- Mutation ---------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        """Add chunks with their embeddings; returns assigned vector ids.

        Chunks whose ``chunk_id`` is already present are skipped (idempotent
        re-ingestion of the same file).
        """
        if len(chunks) == 0:
            return []
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Expected embeddings of shape ({len(chunks)}, {self.dimension}), got {embeddings.shape}"
            )
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} does not match store dimension {self.dimension}"
            )
        keep = [i for i, c in enumerate(chunks) if c.metadata.chunk_id not in self.metadata]
        if not keep:
            logger.info("All %d chunks already indexed; nothing added", len(chunks))
            return []
        ids = np.arange(self._next_id, self._next_id + len(keep), dtype=np.int64)
        self._index.add_with_ids(np.ascontiguousarray(embeddings[keep]), ids)
        for vid, i in zip(ids.tolist(), keep, strict=True):
            self.metadata.add(vid, chunks[i])
        self._next_id += len(keep)
        self.updated_at = datetime.now(UTC)
        logger.info("Added %d vectors (total %d)", len(keep), len(self))
        return ids.tolist()

    def remove_doc(self, doc_id: str) -> int:
        """Remove every chunk of a document; returns the number removed."""
        ids = self.metadata.ids_for_doc(doc_id)
        return self._remove_ids(ids)

    def remove_source(self, source: str) -> int:
        """Remove every chunk from a source filename; returns the number removed."""
        ids = self.metadata.ids_for_source(source)
        return self._remove_ids(ids)

    def _remove_ids(self, ids: list[int]) -> int:
        if not ids:
            return 0
        import faiss

        selector = faiss.IDSelectorArray(np.asarray(ids, dtype=np.int64))
        removed = int(self._index.remove_ids(selector))
        self.metadata.remove(ids)
        self.updated_at = datetime.now(UTC)
        logger.info("Removed %d vectors (total %d)", removed, len(self))
        return removed

    def clear(self) -> None:
        self._index.reset()
        self.metadata = MetadataStore()
        self._next_id = 0
        self.updated_at = datetime.now(UTC)

    # --- Search -----------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        *,
        metadata_filter: MetadataFilter | None = None,
        overfetch: int = 4,
    ) -> list[tuple[Chunk, float]]:
        """Return up to ``k`` ``(chunk, cosine_score)`` pairs, best first.

        Metadata filtering is applied *after* the ANN search on an over-fetched
        candidate list (``k * overfetch``), which is simple and exact for the
        flat index used here.
        """
        if self.is_empty or k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query dimension {query.shape[1]} does not match store dimension {self.dimension}"
            )
        fetch = min(len(self), k * (overfetch if metadata_filter else 1))
        scores, ids = self._index.search(np.ascontiguousarray(query), fetch)
        results: list[tuple[Chunk, float]] = []
        for score, vid in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if vid < 0:
                continue
            chunk = self.metadata.get(vid)
            if chunk is None:
                logger.warning("Vector %d has no metadata; skipping", vid)
                continue
            if metadata_filter and not metadata_filter(chunk.metadata):
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results

    # --- Persistence ------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Persist index, metadata and manifest to ``directory``."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / self.INDEX_FILE))
        self.metadata.save(directory)
        manifest = {
            "dimension": self.dimension,
            "embedder_name": self.embedder_name,
            "count": len(self),
            "next_id": self._next_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        (directory / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Saved vector store (%d vectors) to %s", len(self), directory)

    @classmethod
    def exists(cls, directory: Path) -> bool:
        directory = Path(directory)
        return (directory / cls.INDEX_FILE).is_file() and (directory / cls.MANIFEST_FILE).is_file()

    @classmethod
    def load(cls, directory: Path, *, expected_embedder: str | None = None) -> FaissVectorStore:
        """Load a store saved with :meth:`save`.

        Raises:
            FileNotFoundError: if no store exists in ``directory``.
            VectorStoreError: if files are inconsistent or the embedder differs.
        """
        import faiss

        directory = Path(directory)
        if not cls.exists(directory):
            raise FileNotFoundError(f"No vector store found in {directory}")
        manifest = json.loads((directory / cls.MANIFEST_FILE).read_text(encoding="utf-8"))
        store = cls(int(manifest["dimension"]), manifest.get("embedder_name", "unknown"))
        store._index = faiss.read_index(str(directory / cls.INDEX_FILE))
        if store._index.d != store.dimension:
            raise VectorStoreError("Manifest dimension does not match FAISS index")
        store.metadata = MetadataStore.load(directory)
        if len(store.metadata) != store._index.ntotal:
            raise VectorStoreError(
                f"Metadata has {len(store.metadata)} records but index has {store._index.ntotal} vectors"
            )
        store._next_id = int(manifest.get("next_id", store._index.ntotal))
        store.created_at = datetime.fromisoformat(manifest["created_at"])
        store.updated_at = datetime.fromisoformat(manifest["updated_at"])
        if expected_embedder and manifest.get("embedder_name") not in (
            None,
            "unknown",
            expected_embedder,
        ):
            raise VectorStoreError(
                f"Index was built with embedder '{manifest['embedder_name']}' but the current "
                f"embedder is '{expected_embedder}'. Rebuild the index or switch backends."
            )
        logger.info("Loaded vector store (%d vectors) from %s", len(store), directory)
        return store

    @classmethod
    def load_or_create(
        cls, directory: Path, dimension: int, embedder_name: str
    ) -> FaissVectorStore:
        """Load the store in ``directory`` if present, otherwise create an empty one."""
        if cls.exists(directory):
            return cls.load(directory, expected_embedder=embedder_name)
        return cls(dimension, embedder_name)
