from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.audio import (
    AudioClip,
    AudioDecodeError,
    AudioDisabledError,
    AudioLoader,
    AudioProcessor,
    Transcriber,
    resample,
    transcript_to_document,
)
from src.schemas import ContentType, Transcript, TranscriptSegment


def test_load_mono_wav(fixtures_dir: Path):
    clip = AudioLoader().load(fixtures_dir / "tone_16k_mono.wav")
    assert clip.sample_rate == 16_000
    assert clip.samples.dtype == np.float32
    assert clip.samples.ndim == 1
    assert clip.duration_s == pytest.approx(1.0, abs=0.01)
    assert np.abs(clip.samples).max() <= 1.0


def test_load_stereo_wav_is_resampled_to_mono_16k(fixtures_dir: Path):
    clip = AudioLoader().load(fixtures_dir / "tone_44k_stereo.wav")
    assert clip.sample_rate == 16_000
    assert clip.samples.ndim == 1
    assert clip.duration_s == pytest.approx(0.5, abs=0.01)


def test_load_bytes_matches_load(fixtures_dir: Path):
    path = fixtures_dir / "tone_16k_mono.wav"
    a = AudioLoader().load(path)
    b = AudioLoader().load_bytes(path.read_bytes(), "tone.wav")
    assert np.allclose(a.samples, b.samples)


def test_resample_preserves_tone_frequency():
    rate = 44_100
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = resample(tone, rate, 16_000)
    assert len(out) == 16_000
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spectrum) * 16_000 / len(out)
    assert abs(peak_hz - 440) < 2


def test_unsupported_format(tmp_path: Path):
    bad = tmp_path / "clip.xyz"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioDecodeError):
        AudioLoader().load(bad)


def test_corrupt_wav_raises_decode_error(tmp_path: Path):
    bad = tmp_path / "clip.wav"
    bad.write_bytes(b"RIFF garbage")
    loader = AudioLoader()
    if loader.ffmpeg_available():
        pytest.skip("ffmpeg present: corrupt files are retried through ffmpeg")
    with pytest.raises(AudioDecodeError):
        loader.load(bad)


class EchoTranscriber:
    """Test double implementing the Transcriber protocol (returns clip stats)."""

    name = "echo"

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> Transcript:
        return Transcript(
            text=f"clip of {clip.duration_s:.1f} seconds",
            language=language or "en",
            segments=[TranscriptSegment(text="clip of", start=0.0, end=0.5)],
            duration_s=clip.duration_s,
            backend=self.name,
        )


def test_transcriber_protocol_and_processor(fixtures_dir: Path):
    transcriber = EchoTranscriber()
    assert isinstance(transcriber, Transcriber)
    processor = AudioProcessor(transcriber)
    doc = processor.process_file(fixtures_dir / "tone_16k_mono.wav")
    assert doc.content_type == ContentType.AUDIO
    assert doc.text.startswith("clip of 1.0")
    assert doc.segments and doc.segments[0].start == 0.0
    assert doc.extra["language"] == "en"


def test_processor_without_backend_raises(fixtures_dir: Path):
    processor = AudioProcessor(None)
    assert not processor.enabled
    with pytest.raises(AudioDisabledError):
        processor.transcribe_file(fixtures_dir / "tone_16k_mono.wav")


def test_transcript_to_document_keeps_segments():
    transcript = Transcript(
        text="hello world",
        language="en",
        segments=[TranscriptSegment(text="hello", start=0.0, end=1.0)],
        duration_s=2.0,
        backend="x",
    )
    doc = transcript_to_document(transcript, "a.wav")
    assert doc.segments is not None and len(doc.segments) == 1
    assert doc.extra["duration_s"] == 2.0


def test_whisper_missing_weights_fails_clearly(monkeypatch: pytest.MonkeyPatch):
    """With the HF cache disabled/offline, loading must raise ModelUnavailableError."""
    pytest.importorskip("transformers")
    import huggingface_hub.constants as hf_constants
    from src.models.errors import ModelUnavailableError
    from src.models.whisper_loader import load_whisper

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(ModelUnavailableError) as excinfo:
        load_whisper("this-org/does-not-exist-whisper", "cpu")
    assert "does-not-exist-whisper" in str(excinfo.value)
