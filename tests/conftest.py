"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Give every test a clean, offline configuration and a temp data dir."""
    from src.config import reset_settings_cache

    monkeypatch.setenv("MRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MRAG_INDEX_DIR", str(tmp_path / "data" / "index"))
    monkeypatch.setenv("MRAG_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("MRAG_WHISPER_BACKEND", "none")
    monkeypatch.setenv("MRAG_VISION_BACKEND", "metadata")
    monkeypatch.setenv("MRAG_LLM_BACKEND", "extractive")
    monkeypatch.setenv("MRAG_DEVICE", "cpu")
    monkeypatch.setenv("MRAG_SIMILARITY_THRESHOLD", "0.0")
    reset_settings_cache()
    yield
    reset_settings_cache()
