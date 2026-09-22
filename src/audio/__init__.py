"""Audio decoding and speech recognition."""

from src.audio.loader import AudioClip, AudioDecodeError, AudioLoader, resample, to_mono
from src.audio.processor import AudioProcessor, transcript_to_document
from src.audio.transcriber import (
    AudioDisabledError,
    Transcriber,
    WhisperTranscriber,
    build_transcriber,
)

__all__ = [
    "AudioClip",
    "AudioDecodeError",
    "AudioDisabledError",
    "AudioLoader",
    "AudioProcessor",
    "Transcriber",
    "WhisperTranscriber",
    "build_transcriber",
    "resample",
    "to_mono",
    "transcript_to_document",
]
