"""Speech-to-text backends.

:class:`Transcriber` is the interface the rest of the system depends on.
:class:`WhisperTranscriber` implements it with Hugging Face Whisper. The
factory :func:`build_transcriber` picks a backend from :class:`Settings`.

There is deliberately **no** offline fallback that fabricates a transcript.
When Whisper is unavailable, audio input is disabled and the user is told why.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.audio.loader import TARGET_SAMPLE_RATE, AudioClip
from src.config import Settings
from src.models.errors import ModelUnavailableError
from src.schemas import Transcript, TranscriptSegment
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudioDisabledError(RuntimeError):
    """Raised when audio input is requested but no speech backend is configured."""


@runtime_checkable
class Transcriber(Protocol):
    """Interface for speech recognition backends."""

    name: str

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> Transcript:
        """Transcribe a mono 16 kHz clip."""
        ...


class WhisperTranscriber:
    """Whisper (Hugging Face ``transformers``) with timestamps and language detection.

    Long audio (> 30 s) is handled by the ``transformers`` ASR pipeline using
    chunked inference with stride overlap, which is the standard approach for
    Whisper's fixed 30 s receptive field.
    """

    name = "whisper"

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str = "auto",
        chunk_length_s: int = 30,
        default_language: str | None = None,
    ) -> None:
        from src.models.whisper_loader import load_whisper

        self.bundle = load_whisper(model_name, device)
        self.model_name = model_name
        self.default_language = default_language
        self.chunk_length_s = chunk_length_s
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Any:
        from transformers import pipeline

        return pipeline(
            task="automatic-speech-recognition",
            model=self.bundle.model,
            tokenizer=self.bundle.processor.tokenizer,
            feature_extractor=self.bundle.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            device=self.bundle.device,
            dtype=self.bundle.model.dtype,
        )

    def detect_language(self, clip: AudioClip) -> str | None:
        """Detect the spoken language from the first 30 seconds (if supported)."""
        detect = getattr(self.bundle.model, "detect_language", None)
        if detect is None:
            return None
        try:
            import torch

            window = clip.samples[: 30 * clip.sample_rate]
            features = self.bundle.processor.feature_extractor(
                window, sampling_rate=clip.sample_rate, return_tensors="pt"
            ).input_features.to(self.bundle.model.device, dtype=self.bundle.model.dtype)
            with torch.no_grad():
                lang_ids = detect(features)
            token = self.bundle.processor.tokenizer.decode(int(lang_ids[0]))
            return token.strip("<|>") or None
        except Exception as exc:  # language detection is best-effort
            logger.warning("Language detection failed: %s", exc)
            return None

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> Transcript:
        if clip.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Whisper expects {TARGET_SAMPLE_RATE} Hz audio, got {clip.sample_rate}")
        if clip.samples.size == 0:
            return Transcript(text="", language=None, segments=[], duration_s=0.0, backend=self.name)

        language = language or self.default_language or self.detect_language(clip)
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        audio = np.ascontiguousarray(clip.samples, dtype=np.float32)
        result = self._pipeline(
            {"raw": audio, "sampling_rate": clip.sample_rate},
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        segments = [
            TranscriptSegment(
                text=str(chunk.get("text", "")).strip(),
                start=_ts(chunk, 0),
                end=_ts(chunk, 1),
            )
            for chunk in result.get("chunks", [])
        ]
        text = str(result.get("text", "")).strip()
        logger.info(
            "Transcribed %s (%.1fs, lang=%s): %d segments, %d chars",
            clip.source, clip.duration_s, language, len(segments), len(text),
        )
        return Transcript(
            text=text,
            language=language,
            segments=segments,
            duration_s=clip.duration_s,
            backend=f"{self.name}:{self.model_name}",
        )


def _ts(chunk: dict[str, Any], idx: int) -> float | None:
    stamps = chunk.get("timestamp")
    if not stamps or stamps[idx] is None:
        return None
    return float(stamps[idx])


def build_transcriber(settings: Settings) -> Transcriber | None:
    """Instantiate the configured transcriber, or ``None`` if audio is disabled.

    Raises:
        ModelUnavailableError: when the configured backend cannot be loaded.
    """
    if settings.whisper_backend == "none":
        logger.warning("Audio input disabled (MRAG_WHISPER_BACKEND=none)")
        return None
    if settings.whisper_backend == "transformers":
        return WhisperTranscriber(
            model_name=settings.whisper_model,
            device=settings.device,
            default_language=settings.whisper_language,
        )
    raise ModelUnavailableError(f"Unknown whisper backend: {settings.whisper_backend}")
