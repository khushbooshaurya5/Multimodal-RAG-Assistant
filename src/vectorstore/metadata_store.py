"""Metadata persistence for the vector store.

FAISS only stores vectors. Chunk text and metadata are kept here, keyed by the
integer id assigned to each vector, and persisted as JSON Lines next to the
index so nothing is lost across restarts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from src.schemas import Chunk


class MetadataStore:
    """In-memory ``id -> Chunk`` map with JSONL persistence."""

    FILENAME = "metadata.jsonl"

    def __init__(self) -> None:
        self._chunks: dict[int, Chunk] = {}
        self._chunk_id_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_id_index

    def get(self, vector_id: int) -> Chunk | None:
        return self._chunks.get(int(vector_id))

    def id_for_chunk(self, chunk_id: str) -> int | None:
        return self._chunk_id_index.get(chunk_id)

    def add(self, vector_id: int, chunk: Chunk) -> None:
        self._chunks[int(vector_id)] = chunk
        self._chunk_id_index[chunk.metadata.chunk_id] = int(vector_id)

    def remove(self, vector_ids: Iterable[int]) -> None:
        for vid in vector_ids:
            chunk = self._chunks.pop(int(vid), None)
            if chunk is not None:
                self._chunk_id_index.pop(chunk.metadata.chunk_id, None)

    def items(self) -> Iterator[tuple[int, Chunk]]:
        return iter(self._chunks.items())

    def ids_for_doc(self, doc_id: str) -> list[int]:
        return [vid for vid, chunk in self._chunks.items() if chunk.metadata.doc_id == doc_id]

    def ids_for_source(self, source: str) -> list[int]:
        return [vid for vid, chunk in self._chunks.items() if chunk.metadata.source == source]

    def sources(self) -> dict[str, int]:
        """Return ``source -> number of chunks``."""
        counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            counts[chunk.metadata.source] = counts.get(chunk.metadata.source, 0) + 1
        return counts

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.FILENAME
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for vid, chunk in sorted(self._chunks.items()):
                handle.write(json.dumps({"id": vid, "chunk": chunk.model_dump(mode="json")}) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, directory: Path) -> MetadataStore:
        store = cls()
        path = directory / cls.FILENAME
        if not path.is_file():
            return store
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                store.add(int(record["id"]), Chunk.model_validate(record["chunk"]))
        return store
