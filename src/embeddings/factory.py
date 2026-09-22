"""Select an embedding backend from settings."""

from __future__ import annotations

from src.config import Settings
from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.models.errors import ModelUnavailableError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the configured embedder.

    Raises:
        ModelUnavailableError: if a neural backend is configured but cannot load.
    """
    if settings.embedding_backend == "sentence_transformers":
        from src.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.embedding_model, settings.device)
    if settings.embedding_backend == "hashing":
        logger.warning(
            "Embedding backend 'hashing': lexical fallback, no semantic model (dim=%d)",
            settings.embedding_dim,
        )
        return HashingEmbedder(settings.embedding_dim)
    raise ModelUnavailableError(f"Unknown embedding backend: {settings.embedding_backend}")
