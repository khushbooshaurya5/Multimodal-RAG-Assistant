from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.runner import EvaluationRunner, render_markdown
from src.rag.service import AssistantService

pytestmark = pytest.mark.integration

DATASET = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_dataset.json"


def test_runner_produces_complete_report(service: AssistantService):
    runner = EvaluationRunner(service, DATASET, top_k=3)
    report = runner.run()
    assert report.num_questions == 16
    assert report.aggregate["recall@3"] is not None and 0.0 <= report.aggregate["recall@3"] <= 1.0
    assert report.aggregate["citation_validity"] == 1.0  # extractive answers only cite real passages
    assert report.aggregate["no_evidence_detection_rate"] is not None
    assert "retrieval" in report.latency_ms and report.latency_ms["total"]["p95"] is not None
    assert report.skipped_audio == ["a01"]  # no spoken audio shipped
    markdown = render_markdown(report)
    assert "Fallback backends active" in markdown and "| q01 |" in markdown
