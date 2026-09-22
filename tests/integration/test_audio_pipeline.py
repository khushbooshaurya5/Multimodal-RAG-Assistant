from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.service import AssistantService
from src.schemas import ContentType

pytestmark = pytest.mark.integration


def test_audio_to_transcription_to_retrieval_to_answer(indexed_service: AssistantService, fixtures_dir: Path):
    clip = indexed_service.load_audio((fixtures_dir / "tone_16k_mono.wav").read_bytes(), "question.wav")
    response = indexed_service.query(None, audio=clip)

    assert response.transcript is not None and response.transcript.backend == "stub_whisper"
    assert response.query.startswith("Explain how pooling layers work")
    assert response.transcript.segments and response.transcript.segments[0].start == 0.0
    assert response.citations and response.citations[0].source in {"cnn_notes.md", "sample.pdf"}
    assert "transcription" in response.timings_ms


def test_voice_plus_text_are_merged(indexed_service: AssistantService, fixtures_dir: Path):
    clip = indexed_service.load_audio((fixtures_dir / "tone_16k_mono.wav").read_bytes(), "q.wav")
    response = indexed_service.query("Be brief.", audio=clip)
    assert response.query.startswith("Be brief. Explain how pooling layers work")


def test_audio_file_ingestion_keeps_timestamps(indexed_service: AssistantService, fixtures_dir: Path):
    result = indexed_service.ingest_files([fixtures_dir / "tone_44k_stereo.wav"])[0]
    assert result.ok and result.content_type == ContentType.AUDIO and result.num_chunks >= 1
    response = indexed_service.query(
        "pooling layers convolutional", top_k=5, similarity_threshold=-1.0
    )
    audio_cites = [c for c in response.citations if c.content_type == ContentType.AUDIO]
    assert audio_cites and audio_cites[0].start_time is not None and audio_cites[0].end_time is not None


def test_audio_disabled_is_reported_not_faked(service: AssistantService, fixtures_dir: Path):
    from src.audio import AudioDisabledError

    clip = service.load_audio((fixtures_dir / "tone_16k_mono.wav").read_bytes(), "q.wav")
    with pytest.raises(AudioDisabledError):
        service.query(None, audio=clip)
    result = service.ingest_files([fixtures_dir / "tone_16k_mono.wav"])[0]
    assert not result.ok and "AudioDisabledError" in (result.error or "")
