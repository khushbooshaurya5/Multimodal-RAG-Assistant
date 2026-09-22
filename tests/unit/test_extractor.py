from __future__ import annotations

from pathlib import Path

import pytest
from src.schemas import ContentType
from src.text import TextExtractor, UnsupportedFileError, normalise_whitespace, strip_markdown


def test_extract_markdown(fixtures_dir: Path):
    doc = TextExtractor().extract(fixtures_dir / "cnn_notes.md")
    assert doc.content_type == ContentType.TEXT
    assert "convolutional neural network" in doc.text.lower()
    assert doc.source == "cnn_notes.md"
    assert len(doc.doc_id) == 16


def test_extract_pdf_keeps_pages(fixtures_dir: Path):
    doc = TextExtractor().extract(fixtures_dir / "sample.pdf")
    assert doc.content_type == ContentType.PDF
    assert doc.pages is not None and len(doc.pages) == 2
    assert "pooling" in doc.pages[0]
    assert "self-attention" in doc.pages[1]


def test_unsupported_extension(tmp_path: Path):
    path = tmp_path / "file.xyz"
    path.write_text("x")
    with pytest.raises(UnsupportedFileError):
        TextExtractor().extract(path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TextExtractor().extract(tmp_path / "missing.txt")


def test_normalise_whitespace():
    assert normalise_whitespace("a  b\r\n\r\n\r\n\nc \n d") == "a b\n\nc\nd"


def test_markdown_headings_are_stripped(fixtures_dir: Path):
    doc = TextExtractor().extract(fixtures_dir / "cnn_notes.md")
    assert "#" not in doc.text
    assert doc.text.startswith("Convolutional Neural Networks")
    assert strip_markdown("## Title\n**bold** text") == "Title\nbold text"
