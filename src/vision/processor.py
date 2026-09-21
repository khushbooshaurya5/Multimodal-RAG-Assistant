"""Turn image files into :class:`Document` objects for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.schemas import ContentType, Document, ImageAnalysis
from src.utils.hashing import sha256_file, sha256_text
from src.utils.logging import get_logger
from src.vision.analyzer import ImageAnalyzer
from src.vision.loader import load_image, load_image_bytes

logger = get_logger(__name__)


def analysis_to_document(analysis: ImageAnalysis, doc_id: str | None = None) -> Document:
    """Wrap an :class:`ImageAnalysis` as an indexable document."""
    text = analysis.to_searchable_text()
    return Document(
        doc_id=doc_id or sha256_text(f"{analysis.source}:{text}"),
        source=analysis.source,
        content_type=ContentType.IMAGE,
        text=text,
        extra={
            "image_kind": analysis.image_kind,
            "width": analysis.width,
            "height": analysis.height,
            "vision_backend": analysis.backend,
            "vision_fallback": analysis.is_fallback,
        },
    )


class ImageProcessor:
    """Load + analyse images."""

    def __init__(self, analyzer: ImageAnalyzer) -> None:
        self.analyzer = analyzer

    @property
    def is_fallback(self) -> bool:
        return self.analyzer.is_fallback

    def analyze_image(self, image: Image.Image, source: str) -> ImageAnalysis:
        return self.analyzer.analyze(image, source)

    def analyze_file(self, path: Path) -> ImageAnalysis:
        path = Path(path)
        return self.analyzer.analyze(load_image(path), path.name)

    def analyze_bytes(self, data: bytes, filename: str) -> ImageAnalysis:
        return self.analyzer.analyze(load_image_bytes(data), filename)

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return self.analyzer.answer(image, question, context)

    def process_file(self, path: Path) -> Document:
        """Analyse ``path`` and wrap the result as an indexable document."""
        path = Path(path)
        analysis = self.analyze_file(path)
        return analysis_to_document(analysis, doc_id=sha256_file(path))
