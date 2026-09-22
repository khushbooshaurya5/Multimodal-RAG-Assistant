from __future__ import annotations

import pytest
from src.embeddings import HashingEmbedder
from src.retrieval import RetrievalFilters, Retriever, is_near_duplicate
from src.schemas import Chunk, ChunkMetadata, ContentType, RetrievedChunk
from src.vectorstore import FaissVectorStore


def chunk(
    text: str, i: int, source: str = "doc.txt", ctype: ContentType = ContentType.TEXT
) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            source=source,
            content_type=ctype,
            chunk_id=f"{source}:{i}",
            doc_id=source,
            chunk_index=i,
        ),
    )


CORPUS = [
    chunk("Convolutional neural networks apply learned filters to images.", 0, "cnn.md"),
    chunk(
        "Convolutional neural networks apply learned filters to images!", 1, "cnn.md"
    ),  # near-duplicate
    chunk("Pooling layers downsample feature maps in a CNN.", 2, "cnn.md"),
    chunk("Transformers use self-attention over token sequences.", 0, "transformers.txt"),
    chunk("Whisper transcribes speech to text.", 0, "talk.wav", ContentType.AUDIO),
    chunk("The recipe calls for two eggs and flour.", 0, "recipe.txt"),
]


@pytest.fixture
def retriever() -> Retriever:
    emb = HashingEmbedder(dimension=256)
    store = FaissVectorStore(emb.dimension, emb.name)
    store.add(CORPUS, emb.embed_texts([c.text for c in CORPUS]))
    return Retriever(emb, store, top_k=3, similarity_threshold=0.05)


def test_near_duplicate_detection():
    assert is_near_duplicate("the cat sat on the mat", "The cat sat on the mat.")
    assert not is_near_duplicate("the cat sat on the mat", "stock prices rose sharply")


def test_top_k_and_ranks(retriever: Retriever):
    results = retriever.retrieve("convolutional neural networks filters", top_k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score >= results[1].score
    assert results[0].metadata.source == "cnn.md"


def test_deduplication_removes_near_identical_chunks(retriever: Retriever):
    results = retriever.retrieve("convolutional neural networks apply learned filters", top_k=5)
    texts = [r.text for r in results]
    assert sum("learned filters" in t for t in texts) == 1


def test_threshold_filters_weak_matches(retriever: Retriever):
    results = retriever.retrieve("convolutional neural networks", similarity_threshold=0.99)
    assert results == []
    results = retriever.retrieve(
        "convolutional neural networks", similarity_threshold=-1.0, top_k=10
    )
    assert len(results) >= 4  # everything except the deduplicated twin


def test_metadata_filter_by_content_type(retriever: Retriever):
    results = retriever.retrieve(
        "networks speech text",
        top_k=5,
        similarity_threshold=-1.0,
        filters=RetrievalFilters(content_types=[ContentType.AUDIO]),
    )
    assert [r.metadata.source for r in results] == ["talk.wav"]


def test_metadata_filter_by_source(retriever: Retriever):
    results = retriever.retrieve(
        "networks",
        top_k=5,
        similarity_threshold=-1.0,
        filters=RetrievalFilters(sources=["transformers.txt"]),
    )
    assert {r.metadata.source for r in results} == {"transformers.txt"}


def test_empty_query_or_store():
    emb = HashingEmbedder(dimension=64)
    r = Retriever(emb, FaissVectorStore(64, emb.name))
    assert r.retrieve("anything") == []
    assert r.retrieve("   ") == []


class ReverseReranker:
    name = "reverse"

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [
            c.model_copy(update={"rerank_score": float(i)})
            for i, c in enumerate(reversed(candidates))
        ]


def test_reranker_is_applied_and_can_be_disabled():
    emb = HashingEmbedder(dimension=256)
    store = FaissVectorStore(emb.dimension, emb.name)
    store.add(CORPUS, emb.embed_texts([c.text for c in CORPUS]))
    r = Retriever(emb, store, top_k=3, similarity_threshold=-1.0, reranker=ReverseReranker())
    plain = r.retrieve("pooling layers feature maps", rerank=False)
    reranked = r.retrieve("pooling layers feature maps", rerank=True)
    assert plain[0].chunk.metadata.chunk_id != reranked[0].chunk.metadata.chunk_id
    assert reranked[0].rerank_score is not None
    assert [c.rank for c in reranked] == [1, 2, 3]
