from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_fallbacks(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backends"]["fallbacks"]["generator"] is True
    assert body["index"]["num_vectors"] == 0


def test_ingest_query_and_delete(client: TestClient, fixtures_dir: Path):
    with (fixtures_dir / "cnn_notes.md").open("rb") as f:
        res = client.post("/ingest", files=[("files", ("cnn_notes.md", f, "text/markdown"))])
    assert res.status_code == 200
    item = res.json()[0]
    assert item["error"] is None and item["num_chunks"] >= 1

    res = client.post(
        "/query",
        data={"question": "What do pooling layers do?", "top_k": 2, "include_prompt": "true"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["citations"] and body["citations"][0]["source"] == "cnn_notes.md"
    assert body["is_fallback_generation"] is True
    assert body["evidence"]["level"] in {"weak", "moderate", "strong"}
    assert "EVIDENCE:" in body["prompt"]

    res = client.post("/query", data={"question": "pooling", "content_types": '["audio"]'})
    assert res.json()["citations"] == []  # filtered out

    assert client.get("/sources").json() == {"cnn_notes.md": item["num_chunks"]}
    assert client.delete("/sources/cnn_notes.md").json()["removed"] == item["num_chunks"]
    assert client.get("/sources").json() == {}


def test_query_without_question_is_422(client: TestClient):
    assert client.post("/query", data={}).status_code == 422


def test_transcribe_disabled_is_503(client: TestClient, fixtures_dir: Path):
    with (fixtures_dir / "tone_16k_mono.wav").open("rb") as f:
        res = client.post("/transcribe", files={"audio": ("tone.wav", f, "audio/wav")})
    assert res.status_code == 503


def test_analyze_image_fallback(client: TestClient, fixtures_dir: Path):
    with (fixtures_dir / "diagram.png").open("rb") as f:
        res = client.post("/analyze-image", files={"image": ("diagram.png", f, "image/png")})
    assert res.status_code == 200 and res.json()["is_fallback"] is True
    res = client.post("/analyze-image", files={"image": ("x.png", b"junk", "image/png")})
    assert res.status_code == 400
