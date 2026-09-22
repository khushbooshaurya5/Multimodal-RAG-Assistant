"""Text extraction from plain-text and PDF files.

Produces :class:`~src.schemas.Document` objects with per-page text for PDFs so
that page numbers survive into chunk metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.schemas import ContentType, Document
from src.utils.hashing import sha256_file
from src.utils.logging import get_logger

logger = get_logger(__name__)

TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".rst", ".csv", ".json"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})


class UnsupportedFileError(ValueError):
    """Raised when a file type has no extractor."""


_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")


def strip_markdown(text: str) -> str:
    """Remove heading markers and bold emphasis so chunks read as plain sentences.

    Heading text is kept (it carries useful keywords) but without the ``#``
    markers, which would otherwise render as headings when quoted in answers.
    """
    text = _MD_HEADING.sub("", text)
    return _MD_EMPHASIS.sub(r"\2", text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of blank lines / spaces while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TextExtractor:
    """Extract raw text from supported document files."""

    @staticmethod
    def supports(path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS | PDF_EXTENSIONS

    def extract(self, path: Path) -> Document:
        """Extract a :class:`Document` from ``path``.

        Raises:
            FileNotFoundError: if the file does not exist.
            UnsupportedFileError: for unknown extensions.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return self._extract_text(path)
        if suffix in PDF_EXTENSIONS:
            return self._extract_pdf(path)
        raise UnsupportedFileError(f"No text extractor for '{suffix}' files: {path.name}")

    def _extract_text(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in {".md", ".markdown"}:
            raw = strip_markdown(raw)
        text = normalise_whitespace(raw)
        logger.info("Extracted %d characters from %s", len(text), path.name)
        return Document(
            doc_id=sha256_file(path),
            source=path.name,
            content_type=ContentType.TEXT,
            text=text,
        )

    def _extract_pdf(self, path: Path) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency present in requirements
            raise UnsupportedFileError("pypdf is required for PDF extraction") from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pypdf can fail on malformed pages
                logger.warning("Failed to extract page %d of %s: %s", page_number, path.name, exc)
                page_text = ""
            pages.append(normalise_whitespace(page_text))
        text = "\n\n".join(p for p in pages if p)
        if not text:
            logger.warning(
                "No extractable text in %s (scanned PDF?). Consider ingesting page images.",
                path.name,
            )
        logger.info("Extracted %d pages / %d characters from %s", len(pages), len(text), path.name)
        return Document(
            doc_id=sha256_file(path),
            source=path.name,
            content_type=ContentType.PDF,
            text=text,
            pages=pages,
            extra={"num_pages": len(pages)},
        )
