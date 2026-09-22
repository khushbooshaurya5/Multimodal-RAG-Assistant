from __future__ import annotations

import pytest
from src.schemas import ContentType, Document, TranscriptSegment
from src.text import TextChunker, split_sentences


def _doc(text: str, **kwargs) -> Document:
    return Document(
        doc_id="doc1", source="doc.txt", content_type=ContentType.TEXT, text=text, **kwargs
    )


def test_short_text_is_single_chunk():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    assert chunker.split_text("Hello world.") == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert TextChunker().split_text("   \n ") == []


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence {i} is about topic {i}." for i in range(40)]
    text = " ".join(sentences)
    chunker = TextChunker(chunk_size=120, chunk_overlap=40)
    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    # Every sentence must be present in at least one chunk.
    for sentence in sentences:
        assert any(sentence in c for c in chunks)
    # Overlap: the last sentence of chunk i re-appears at the start of chunk i+1.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = split_sentences(prev)[-1]
        assert nxt.startswith(last_sentence)


def test_no_overlap_when_zero():
    text = " ".join(f"Sentence {i} here." for i in range(30))
    chunks = TextChunker(chunk_size=80, chunk_overlap=0).split_text(text)
    joined = " ".join(chunks)
    assert joined == text  # no repeated content


def test_very_long_word_is_hard_split():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(500))  # no repeats across chunks
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_text(text)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, chunk_overlap=50)


def test_chunk_document_assigns_metadata_and_unique_ids():
    text = " ".join(f"Sentence {i} about networks." for i in range(30))
    chunks = TextChunker(chunk_size=100, chunk_overlap=20).chunk_document(_doc(text))
    ids = [c.metadata.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata.source == "doc.txt" for c in chunks)
    assert all(c.metadata.doc_id == "doc1" for c in chunks)


def test_paged_document_keeps_page_numbers():
    doc = Document(
        doc_id="pdf1",
        source="doc.pdf",
        content_type=ContentType.PDF,
        text="p1 p2",
        pages=["Page one text.", "", "Page three text."],
    )
    chunks = TextChunker().chunk_document(doc)
    assert [c.metadata.page for c in chunks] == [1, 3]


def test_audio_segments_keep_time_span():
    segments = [
        TranscriptSegment(text=f"segment {i} words", start=i * 2.0, end=i * 2.0 + 1.5)
        for i in range(12)
    ]
    doc = Document(
        doc_id="a1", source="talk.wav", content_type=ContentType.AUDIO, text="", segments=segments
    )
    chunks = TextChunker(chunk_size=60, chunk_overlap=0).chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0].metadata.start_time == 0.0
    assert chunks[-1].metadata.end_time == pytest.approx(23.5)
    # Time spans are monotonically increasing.
    starts = [c.metadata.start_time for c in chunks]
    assert starts == sorted(starts)
