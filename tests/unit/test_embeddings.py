from __future__ import annotations

import numpy as np
import pytest

from src.config import get_settings
from src.embeddings import Embedder, HashingEmbedder, build_embedder, l2_normalise


def test_hashing_embedder_shape_and_norm():
    emb = HashingEmbedder(dimension=128)
    assert isinstance(emb, Embedder)
    vectors = emb.embed_texts(["hello world", "another sentence here"])
    assert vectors.shape == (2, 128)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dimension=256).embed_query("convolutional neural networks")
    b = HashingEmbedder(dimension=256).embed_query("convolutional neural networks")
    assert np.array_equal(a, b)


def test_hashing_embedder_similarity_reflects_lexical_overlap():
    emb = HashingEmbedder(dimension=384)
    q = emb.embed_query("convolutional neural networks and pooling layers")
    related = emb.embed_query("pooling layers in convolutional neural networks")
    unrelated = emb.embed_query("the stock market closed higher on tuesday")
    assert float(q @ related) > float(q @ unrelated)
    assert float(q @ related) > 0.5


def test_hashing_embedder_empty_input():
    emb = HashingEmbedder(dimension=64)
    assert emb.embed_texts([]).shape == (0, 64)


def test_hashing_embedder_rejects_tiny_dimension():
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=4)


def test_l2_normalise_handles_zero_rows():
    out = l2_normalise(np.zeros((2, 3)))
    assert np.all(np.isfinite(out))


def test_factory_returns_hashing_in_offline_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MRAG_EMBEDDING_DIM", "200")
    from src.config import reset_settings_cache

    reset_settings_cache()
    emb = build_embedder(get_settings())
    assert emb.is_fallback and emb.dimension == 200


def test_sentence_transformer_missing_weights_fails_clearly(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("sentence_transformers")
    from src.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from src.models.errors import ModelUnavailableError

    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(ModelUnavailableError, match="hashing"):
        SentenceTransformerEmbedder("this-org/does-not-exist-embedder", "cpu")
