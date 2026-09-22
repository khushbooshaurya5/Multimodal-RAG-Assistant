"""Load sentence-transformers embedding and cross-encoder reranking models."""

from __future__ import annotations

from typing import Any

from src.models.errors import ModelUnavailableError
from src.utils.device import resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_sentence_transformer(model_name: str, device_preference: str = "auto") -> Any:
    """Load a ``SentenceTransformer`` model.

    Raises:
        ModelUnavailableError: if the library or weights are unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelUnavailableError(
            f"Cannot load embedder '{model_name}': sentence-transformers not installed ({exc})"
        ) from exc
    device = resolve_device(device_preference)
    logger.info("Loading embedding model %s on %s", model_name, device)
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Cannot load embedding model '{model_name}': {type(exc).__name__}: {exc}. "
            "Check network access to huggingface.co or set MRAG_EMBEDDING_BACKEND=hashing."
        ) from exc


def load_cross_encoder(model_name: str, device_preference: str = "auto") -> Any:
    """Load a sentence-transformers ``CrossEncoder`` for reranking."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ModelUnavailableError(
            f"Cannot load reranker '{model_name}': sentence-transformers not installed ({exc})"
        ) from exc
    device = resolve_device(device_preference)
    logger.info("Loading cross-encoder %s on %s", model_name, device)
    try:
        return CrossEncoder(model_name, device=device)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Cannot load reranker '{model_name}': {type(exc).__name__}: {exc}"
        ) from exc
