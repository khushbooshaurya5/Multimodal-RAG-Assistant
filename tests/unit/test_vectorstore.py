from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.embeddings import HashingEmbedder
from src.schemas import Chunk, ChunkMetadata, ContentType
from src.vectorstore import FaissVectorStore, VectorStoreError


def make_chunk(text: str, i: int, doc_id: str = "doc", source: str = "doc.txt", **meta) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            source=source,
            content_type=meta.pop("content_type", ContentType.TEXT),
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            chunk_index=i,
            **meta,
        ),
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=128)


@pytest.fixture
def populated(embedder: HashingEmbedder) -> FaissVectorStore:
    texts = [
        "Convolutional neural networks use convolution and pooling layers.",
        "Transformers use self-attention over token sequences.",
        "Whisper is a speech recognition model trained on audio.",
        "FAISS is a library for efficient similarity search.",
    ]
    chunks = [
        make_chunk(texts[0], 0, doc_id="a", source="cnn.md"),
        make_chunk(texts[1], 1, doc_id="a", source="cnn.md"),
        make_chunk(
            texts[2],
            0,
            doc_id="b",
            source="talk.wav",
            content_type=ContentType.AUDIO,
            start_time=1.0,
        ),
        make_chunk(
            texts[3], 0, doc_id="c", source="faiss.pdf", content_type=ContentType.PDF, page=3
        ),
    ]
    store = FaissVectorStore(embedder.dimension, embedder.name)
    store.add(chunks, embedder.embed_texts(texts))
    return store


def test_add_and_search(populated: FaissVectorStore, embedder: HashingEmbedder):
    assert len(populated) == 4
    results = populated.search(embedder.embed_query("convolution pooling layers"), k=2)
    assert len(results) == 2
    assert results[0][0].metadata.source == "cnn.md"
    assert results[0][1] >= results[1][1]
    assert -1.0 <= results[0][1] <= 1.0 + 1e-6


def test_add_is_idempotent(populated: FaissVectorStore, embedder: HashingEmbedder):
    chunk = make_chunk(
        "Convolutional neural networks use convolution and pooling layers.",
        0,
        doc_id="a",
        source="cnn.md",
    )
    assert populated.add([chunk], embedder.embed_texts([chunk.text])) == []
    assert len(populated) == 4


def test_metadata_filter(populated: FaissVectorStore, embedder: HashingEmbedder):
    q = embedder.embed_query("model")
    results = populated.search(
        q, k=4, metadata_filter=lambda m: m.content_type == ContentType.AUDIO
    )
    assert [r[0].metadata.source for r in results] == ["talk.wav"]
    assert results[0][0].metadata.start_time == 1.0


def test_remove_doc_and_source(populated: FaissVectorStore, embedder: HashingEmbedder):
    assert populated.remove_doc("a") == 2
    assert len(populated) == 2
    assert populated.remove_source("faiss.pdf") == 1
    assert populated.sources() == {"talk.wav": 1}
    results = populated.search(embedder.embed_query("convolution"), k=5)
    assert all(r[0].metadata.source != "cnn.md" for r in results)


def test_dimension_mismatch_raises(embedder: HashingEmbedder):
    store = FaissVectorStore(64)
    with pytest.raises(VectorStoreError):
        store.add([make_chunk("x", 0)], embedder.embed_texts(["x"]))
    with pytest.raises(VectorStoreError):
        store.add([make_chunk("x", 0)], np.zeros((64,), dtype=np.float32))


def test_search_empty_store(embedder: HashingEmbedder):
    assert FaissVectorStore(128).search(embedder.embed_query("x"), k=3) == []


def test_save_and_load_round_trip(
    populated: FaissVectorStore, embedder: HashingEmbedder, tmp_path: Path
):
    index_dir = tmp_path / "index"
    populated.save(index_dir)
    assert FaissVectorStore.exists(index_dir)
    assert (index_dir / "metadata.jsonl").is_file() and (index_dir / "manifest.json").is_file()

    loaded = FaissVectorStore.load(index_dir, expected_embedder=embedder.name)
    assert len(loaded) == len(populated)
    q = embedder.embed_query("speech recognition audio")
    before = populated.search(q, k=2)
    after = loaded.search(q, k=2)
    assert [c.metadata.chunk_id for c, _ in before] == [c.metadata.chunk_id for c, _ in after]
    assert np.allclose([s for _, s in before], [s for _, s in after])
    # Rich metadata survives.
    pdf_chunk = next(
        c for _, c in loaded.metadata.items() if c.metadata.content_type == ContentType.PDF
    )
    assert pdf_chunk.metadata.page == 3
    assert pdf_chunk.metadata.indexed_at is not None

    # Ids keep increasing after reload so new vectors never collide.
    new_ids = loaded.add(
        [make_chunk("new text", 9, doc_id="z")], embedder.embed_texts(["new text"])
    )
    assert new_ids[0] >= 4


def test_load_rejects_different_embedder(populated: FaissVectorStore, tmp_path: Path):
    populated.save(tmp_path)
    with pytest.raises(VectorStoreError, match="Rebuild"):
        FaissVectorStore.load(tmp_path, expected_embedder="sentence_transformers:other")


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FaissVectorStore.load(tmp_path)
    store = FaissVectorStore.load_or_create(tmp_path, 32, "hashing:32")
    assert store.is_empty


def test_load_detects_metadata_index_mismatch(populated: FaissVectorStore, tmp_path: Path):
    populated.save(tmp_path)
    lines = (tmp_path / "metadata.jsonl").read_text().splitlines()
    (tmp_path / "metadata.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(VectorStoreError, match="records"):
        FaissVectorStore.load(tmp_path)
