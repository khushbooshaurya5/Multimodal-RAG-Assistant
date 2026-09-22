"""Embedding interface shared by all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise to unit length so inner product == cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@runtime_checkable
class Embedder(Protocol):
    """Maps text to unit-length ``float32`` vectors."""

    name: str
    dimension: int
    is_fallback: bool

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dimension)`` array of L2-normalised embeddings."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Return a ``(dimension,)`` L2-normalised embedding for a search query."""
        ...
