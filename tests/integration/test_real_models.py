"""Tests against real model weights. They skip when weights cannot be loaded
(no network / no cache), so CI stays green offline while developers with a
Hugging Face cache get real coverage.

Run explicitly::

    pytest -m requires_models tests/integration/test_real_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.errors import ModelUnavailableError

pytestmark = [pytest.mark.integration, pytest.mark.requires_models]


def _skip_unless(loader, *args):
    try:
        return loader(*args)
    except ModelUnavailableError as exc:
        pytest.skip(f"model weights unavailable: {exc}")


def test_sentence_transformer_embeddings_are_semantic():
    from src.embeddings.sentence_transformer import SentenceTransformerEmbedder

    embedder = _skip_unless(SentenceTransformerEmbedder, "sentence-transformers/all-MiniLM-L6-v2", "cpu")
    q = embedder.embed_query("a car driving on the road")
    same = embedder.embed_query("an automobile travelling along the street")
    other = embedder.embed_query("a recipe for chocolate cake")
    assert embedder.dimension == 384
    assert float(q @ same) > float(q @ other)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-4)


def test_whisper_transcribes_fixture_tone_without_crashing(fixtures_dir: Path):
    from src.audio import AudioLoader
    from src.audio.transcriber import WhisperTranscriber

    transcriber = _skip_unless(WhisperTranscriber, "openai/whisper-tiny", "cpu")
    clip = AudioLoader().load(fixtures_dir / "tone_16k_mono.wav")
    transcript = transcriber.transcribe(clip, language="en")
    assert transcript.backend.startswith("whisper")
    assert transcript.duration_s == pytest.approx(1.0, abs=0.05)
    assert isinstance(transcript.text, str)  # a pure tone has no words; must not raise


def test_qwen_vl_describes_fixture_diagram(fixtures_dir: Path):
    from src.vision import QwenVLAnalyzer, load_image

    analyzer = _skip_unless(QwenVLAnalyzer, "Qwen/Qwen2-VL-2B-Instruct", "cpu", 256)
    analysis = analyzer.analyze(load_image(fixtures_dir / "diagram.png"), "diagram.png")
    assert analysis.is_fallback is False
    assert analysis.description
    assert "conv" in (analysis.extracted_text + analysis.description).lower()
