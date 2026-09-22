from __future__ import annotations

from pathlib import Path

import pytest
from src.rag.service import AssistantService
from src.retrieval.filters import RetrievalFilters
from src.schemas import ContentType, EvidenceLevel

pytestmark = pytest.mark.integration


def test_text_to_retrieval_to_answer(indexed_service: AssistantService):
    response = indexed_service.query("What do pooling layers do in a CNN?", top_k=3)
    assert response.citations, "expected retrieved evidence"
    assert response.citations[0].source in {"cnn_notes.md", "sample.pdf"}
    assert "pooling" in response.answer.lower()
    assert "[1]" in response.answer  # extractive answer cites evidence numbers
    assert response.evidence.level in {EvidenceLevel.MODERATE, EvidenceLevel.STRONG}
    assert response.is_fallback_generation is True  # offline profile: honest label
    assert set(response.timings_ms) >= {"retrieval", "generation"}


def test_pdf_page_numbers_reach_citations(indexed_service: AssistantService):
    response = indexed_service.query(
        "self-attention over token sequences",
        filters=RetrievalFilters(content_types=[ContentType.PDF]),
    )
    assert response.citations and all(c.content_type == ContentType.PDF for c in response.citations)
    assert response.citations[0].page == 2


def test_unanswerable_question_reports_no_evidence(indexed_service: AssistantService):
    response = indexed_service.query(
        "What is the boiling point of mercury?", similarity_threshold=0.6
    )
    assert response.evidence.level == EvidenceLevel.NONE
    assert response.citations == []
    assert "nothing to extract" in response.answer or "insufficient" in response.answer


def test_index_persists_across_service_restarts(
    indexed_service: AssistantService, fixtures_dir: Path
):
    before = indexed_service.index_stats()
    assert before["num_vectors"] > 0
    assert (indexed_service.settings.index_dir / "metadata.jsonl").is_file()

    restarted = AssistantService()  # same env → same index_dir
    assert restarted.index_stats()["num_vectors"] == before["num_vectors"]
    assert restarted.index_stats()["sources"] == before["sources"]
    again = restarted.query("pooling layers", top_k=2)
    assert again.citations and again.citations[0].source in before["sources"]

    # Re-ingesting the same file is a no-op.
    result = restarted.ingest_files([fixtures_dir / "cnn_notes.md"])[0]
    assert result.skipped is True


def test_remove_source_updates_index(indexed_service: AssistantService):
    removed = indexed_service.remove_source("cnn_notes.md")
    assert removed > 0
    response = indexed_service.query(
        "pooling layers convolution", top_k=5, similarity_threshold=-1.0
    )
    assert all(c.source != "cnn_notes.md" for c in response.citations)
