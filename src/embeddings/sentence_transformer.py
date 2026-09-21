"""Dense embeddings from ``sentence-transformers`` models."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import l2_normalise
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Wraps a ``SentenceTransformer`` (default: ``all-MiniLM-L6-v2``, 384-d)."""

    is_fallback = False

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        from src.models.embedding_loader import load_sentence_transformer

        self.model = load_sentence_transformer(model_name, device)
        self.model_name = model_name
        self.name = f"sentence_transformers:{model_name}"
        self.batch_size = batch_size
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalise(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
