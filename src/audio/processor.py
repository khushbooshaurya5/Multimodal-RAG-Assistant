"""Turn audio files into :class:`Document` objects for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from src.audio.loader import AudioLoader
from src.audio.transcriber import AudioDisabledError, Transcriber
from src.schemas import ContentType, Document, Transcript
from src.utils.hashing import sha256_file, sha256_text
from src.utils.logging import get_logger

logger = get_logger(__name__)


def transcript_to_document(
    transcript: Transcript, source: str, doc_id: str | None = None
) -> Document:
    """Build a :class:`Document` (with timed segments) from a transcript."""
    return Document(
        doc_id=doc_id or sha256_text(f"{source}:{transcript.text}"),
        source=source,
        content_type=ContentType.AUDIO,
        text=transcript.text,
        segments=transcript.segments or None,
        extra={
            "language": transcript.language,
            "duration_s": transcript.duration_s,
            "transcription_backend": transcript.backend,
        },
    )


class AudioProcessor:
    """Decode + transcribe audio files."""

    def __init__(self, transcriber: Transcriber | None, loader: AudioLoader | None = None) -> None:
        self.transcriber = transcriber
        self.loader = loader or AudioLoader()

    @property
    def enabled(self) -> bool:
        return self.transcriber is not None

    def transcribe_file(self, path: Path, *, language: str | None = None) -> Transcript:
        if self.transcriber is None:
            raise AudioDisabledError(
                "No speech-recognition backend is available (MRAG_WHISPER_BACKEND=none)."
            )
        clip = self.loader.load(Path(path))
        return self.transcriber.transcribe(clip, language=language)

    def transcribe_bytes(
        self, data: bytes, filename: str, *, language: str | None = None
    ) -> Transcript:
        if self.transcriber is None:
            raise AudioDisabledError(
                "No speech-recognition backend is available (MRAG_WHISPER_BACKEND=none)."
            )
        clip = self.loader.load_bytes(data, filename)
        return self.transcriber.transcribe(clip, language=language)

    def process_file(self, path: Path, *, language: str | None = None) -> Document:
        """Transcribe ``path`` and wrap the result as an indexable document."""
        path = Path(path)
        transcript = self.transcribe_file(path, language=language)
        return transcript_to_document(transcript, path.name, doc_id=sha256_file(path))
