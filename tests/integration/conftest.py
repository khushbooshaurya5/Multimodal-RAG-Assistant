"""Integration fixtures: a fully wired AssistantService in the offline profile.

Neural backends are replaced with protocol-conforming test doubles where the
flow needs them (vision, speech) so the *wiring* text → retrieval → answer,
image → vision → retrieval → answer and audio → whisper → retrieval → answer
is exercised end to end without model downloads. Tests marked
``requires_models`` use real weights and skip when they cannot be loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from src.audio.loader import AudioClip
from src.rag.service import AssistantService
from src.schemas import ImageAnalysis, Transcript, TranscriptSegment


class StubVLM:
    """Stands in for Qwen-VL: returns a fixed structured analysis of the CNN diagram."""

    name = "stub_vlm"
    is_fallback = False

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        return ImageAnalysis(
            source=source,
            description="A block diagram of a convolutional neural network: an Input box connected by an arrow to a Conv2D box.",
            extracted_text="Input Conv2D CNN diagram fixture",
            objects=["input box", "conv2d box", "arrow"],
            image_kind="diagram",
            width=image.width,
            height=image.height,
            backend=self.name,
        )

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return "stub VQA answer"


class StubWhisper:
    """Stands in for Whisper: returns a fixed transcript with timestamps."""

    name = "stub_whisper"

    def __init__(
        self, text: str = "Explain how pooling layers work in a convolutional neural network."
    ) -> None:
        self.text = text

    def transcribe(self, clip: AudioClip, *, language: str | None = None) -> Transcript:
        words = self.text.split()
        step = clip.duration_s / max(len(words), 1)
        segments = [
            TranscriptSegment(
                text=" ".join(words[i : i + 4]),
                start=i * step,
                end=min((i + 4) * step, clip.duration_s),
            )
            for i in range(0, len(words), 4)
        ]
        return Transcript(
            text=self.text,
            language="en",
            segments=segments,
            duration_s=clip.duration_s,
            backend=self.name,
        )


@pytest.fixture
def service(tmp_path: Path) -> AssistantService:
    """Offline service (hashing embeddings, extractive generator, metadata vision, no audio)."""
    return AssistantService()


@pytest.fixture
def multimodal_service(service: AssistantService) -> AssistantService:
    """Same service with stub vision + speech backends injected through the public seams."""
    from src.audio.processor import AudioProcessor
    from src.vision.processor import ImageProcessor

    service.image_analyzer = StubVLM()
    service.transcriber = StubWhisper()
    service.image_processor = ImageProcessor(service.image_analyzer)
    service.audio_processor = AudioProcessor(service.transcriber, service.audio_processor.loader)
    service.ingestion.image_processor = service.image_processor
    service.ingestion.audio_processor = service.audio_processor
    service.pipeline.image_analyzer = service.image_analyzer
    service.pipeline.transcriber = service.transcriber
    return service


@pytest.fixture
def indexed_service(multimodal_service: AssistantService, fixtures_dir: Path) -> AssistantService:
    results = multimodal_service.ingest_files(
        [
            fixtures_dir / "cnn_notes.md",
            fixtures_dir / "transformer_notes.txt",
            fixtures_dir / "sample.pdf",
        ]
    )
    assert all(r.ok for r in results), [r.error for r in results]
    return multimodal_service
