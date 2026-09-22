from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from src.rag.service import AssistantService
from src.schemas import ContentType

pytestmark = pytest.mark.integration


def test_image_to_vision_to_retrieval_to_answer(
    indexed_service: AssistantService, fixtures_dir: Path
):
    image = Image.open(fixtures_dir / "diagram.png")
    response = indexed_service.query(
        "What does this diagram show?", image=image, image_source="diagram.png"
    )

    assert response.image_analysis is not None
    assert response.image_analysis.backend == "stub_vlm"
    assert response.image_analysis.image_kind == "diagram"
    # Image content ("Conv2D", "convolutional") enriched the retrieval query → CNN notes surface.
    assert any(c.source in {"cnn_notes.md", "sample.pdf"} for c in response.citations)
    assert "Attached image analysis" in response.answer  # extractive fallback surfaces the analysis


def test_image_ingestion_makes_image_searchable(
    indexed_service: AssistantService, fixtures_dir: Path
):
    result = indexed_service.ingest_files([fixtures_dir / "diagram.png"])[0]
    assert result.ok and result.content_type == ContentType.IMAGE and result.num_chunks >= 1
    response = indexed_service.query("Input Conv2D block diagram", top_k=3)
    assert any(
        c.source == "diagram.png" and c.content_type == ContentType.IMAGE
        for c in response.citations
    )


def test_metadata_fallback_is_honest_end_to_end(service: AssistantService, fixtures_dir: Path):
    """Without a VLM the pipeline must not pretend to understand the image."""
    image = Image.open(fixtures_dir / "diagram.png")
    response = service.query(
        "What does this diagram show?", image=image, image_source="diagram.png"
    )
    assert response.image_analysis is not None and response.image_analysis.is_fallback is True
    assert "no vision model was available" in response.answer.lower()
    assert response.citations == []  # nothing indexed, nothing invented
