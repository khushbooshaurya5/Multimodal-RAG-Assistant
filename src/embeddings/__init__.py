"""Text embedding backends."""

from src.embeddings.base import Embedder, l2_normalise
from src.embeddings.factory import build_embedder
from src.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "build_embedder", "l2_normalise"]
