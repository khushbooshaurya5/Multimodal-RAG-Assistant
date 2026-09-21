from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.audio.loader import AudioClip
from src.embeddings import HashingEmbedder
from src.rag import ContextAssembler, ExtractiveGenerator, IngestionPipeline, RAGPipeline, compute_evidence
from src.rag.context import build_citations
from src.retrieval import Retriever
from src.schemas import (
    Chunk,
    ChunkMetadata,
    ContentType,
    EvidenceLevel,
    ImageAnalysis,
    RetrievedChunk,
    Transcript,
    TranscriptSegment,
)
from src.text import TextChunker
from src.vectorstore import FaissVectorStore


def rc(text: str, score: float, rank: int, **meta) -> RetrievedChunk:
    meta.setdefault("source", "doc.txt")
    meta.setdefault("content_type", ContentType.TEXT)
    return RetrievedChunk(
        chunk=Chunk(text=text, metadata=ChunkMetadata(chunk_id=f"c{rank}", doc_id="d", chunk_index=rank, **meta)),
        score=score,
        rank=rank,
    )


# --- Context assembly -------------------------------------------------------


def test_context_numbering_and_metadata():
    retrieved = [
        rc("Pooling reduces resolution.", 0.8, 1, source="a.pdf", content_type=ContentType.PDF, page=4),
        rc("Whisper transcribes audio.", 0.5, 2, source="t.wav", content_type=ContentType.AUDIO, start_time=3.0, end_time=7.5),
    ]
    ctx = ContextAssembler().build("What does pooling do?", retrieved)
    assert "[1] source=a.pdf | type=pdf | page 4 | similarity=0.80" in ctx.prompt
    assert "[2] source=t.wav | type=audio | 3.0s–7.5s" in ctx.prompt
    assert "QUESTION: What does pooling do?" in ctx.prompt
    assert [c.index for c in ctx.citations] == [1, 2]
    assert ctx.citations[0].page == 4 and ctx.citations[1].start_time == 3.0


def test_context_budget_truncates_and_marks():
    retrieved = [rc("x" * 500, 0.9 - i * 0.01, i + 1) for i in range(10)]
    ctx = ContextAssembler(max_context_chars=1200).build("q", retrieved)
    assert ctx.truncated is True
    assert len(ctx.included) < 10
    assert len(ctx.citations) == len(ctx.included)


def test_context_without_evidence_adds_note_and_multimodal_sections():
    analysis = ImageAnalysis(source="d.png", description="Two boxes.", extracted_text="Input", backend="fake")
    transcript = Transcript(text="explain this", language="en", backend="whisper")
    ctx = ContextAssembler().build("explain this", [], image_analysis=analysis, transcript=transcript)
    assert "No retrieved passages met the similarity threshold" in ctx.prompt
    assert "VOICE INPUT" in ctx.prompt and "explain this" in ctx.prompt
    assert "Visible text: Input" in ctx.prompt
    assert ctx.citations == []


def test_fallback_image_analysis_is_labelled_in_prompt():
    analysis = ImageAnalysis(source="d.png", description="", backend="metadata_only", is_fallback=True, width=10, height=5)
    ctx = ContextAssembler().build("q", [], image_analysis=analysis)
    assert "no vision model was available" in ctx.prompt


def test_citation_snippets_are_bounded():
    cites = build_citations([rc("word " * 200, 0.5, 1)], snippet_chars=50)
    assert len(cites[0].snippet) <= 50


# --- Evidence indicator -----------------------------------------------------


def test_evidence_levels():
    assert compute_evidence([], similarity_threshold=0.3).level == EvidenceLevel.NONE
    weak = compute_evidence([rc("a", 0.3, 1)], similarity_threshold=0.25)
    assert weak.level == EvidenceLevel.WEAK
    moderate = compute_evidence([rc("a", 0.5, 1)], similarity_threshold=0.25)
    assert moderate.level == EvidenceLevel.MODERATE
    strong = compute_evidence([rc("a", 0.8, 1), rc("b", 0.7, 2)], similarity_threshold=0.25)
    assert strong.level == EvidenceLevel.STRONG
    assert "not a calibrated probability" in strong.note
    lexical = compute_evidence([rc("a", 0.4, 1), rc("b", 0.3, 2)], similarity_threshold=0.05, lexical_backend=True)
    assert lexical.level == EvidenceLevel.STRONG


# --- Extractive generator ---------------------------------------------------


def test_extractive_generator_cites_relevant_sentences():
    retrieved = [
        rc("Pooling layers downsample feature maps. Dropout reduces overfitting.", 0.6, 1),
        rc("Transformers use attention.", 0.2, 2),
    ]
    gen = ExtractiveGenerator()
    out = gen.generate("prompt", retrieved=retrieved, question="What do pooling layers do?")
    assert out.startswith(ExtractiveGenerator.NOTICE)
    assert "Pooling layers downsample feature maps. [1]" in out
    assert "Transformers" not in out


def test_extractive_generator_reports_insufficient_evidence():
    gen = ExtractiveGenerator()
    assert "nothing to extract" in gen.generate("p", retrieved=[], question="q?")
    out = gen.generate("p", retrieved=[rc("Bananas are yellow.", 0.3, 1)], question="What is quantum tunnelling?")
    assert "insufficient" in out


# --- Full pipeline with test doubles ---------------------------------------


class FakeGenerator:
    name = "fake_llm"
    is_fallback = False
    supports_images = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt, *, images=None, retrieved=None, question=None, image_analysis=None):
        self.calls.append({"prompt": prompt, "images": images, "n": len(retrieved or [])})
        return f"answer citing [{len(retrieved or [])}]"


class FakeAnalyzer:
    name = "fake_vlm"
    is_fallback = False

    def analyze(self, image, source):
        return ImageAnalysis(
            source=source, description="A CNN diagram showing convolution and pooling.",
            extracted_text="Input Conv2D Pooling", objects=["box"], image_kind="diagram", backend=self.name,
        )

    def answer(self, image, question, context=None):
        return "vqa"


class FakeTranscriber:
    name = "fake_whisper"

    def transcribe(self, clip: AudioClip, *, language=None) -> Transcript:
        return Transcript(
            text="What do pooling layers do?", language="en",
            segments=[TranscriptSegment(text="What do pooling layers do?", start=0.0, end=clip.duration_s)],
            duration_s=clip.duration_s, backend=self.name,
        )


@pytest.fixture
def pipeline(fixtures_dir: Path, tmp_path: Path):
    emb = HashingEmbedder(dimension=256)
    store = FaissVectorStore(emb.dimension, emb.name)
    ingestion = IngestionPipeline(emb, store, TextChunker(200, 40), index_dir=tmp_path / "idx")
    results = ingestion.ingest_paths([fixtures_dir / "cnn_notes.md", fixtures_dir / "transformer_notes.txt"])
    assert all(r.ok for r in results)
    retriever = Retriever(emb, store, top_k=3, similarity_threshold=0.05)
    gen = FakeGenerator()
    pipe = RAGPipeline(retriever, gen, image_analyzer=FakeAnalyzer(), transcriber=FakeTranscriber(), ingestion=ingestion)
    return pipe, gen, store


def test_pipeline_text_query(pipeline):
    pipe, gen, _ = pipeline
    resp = pipe.answer("What do pooling layers do in a CNN?", include_prompt=True)
    assert resp.answer.startswith("answer citing")
    assert resp.citations and resp.citations[0].source == "cnn_notes.md"
    assert resp.evidence.level != EvidenceLevel.NONE
    assert "retrieval" in resp.timings_ms and "generation" in resp.timings_ms
    assert gen.calls[-1]["images"] is None
    assert resp.prompt and "EVIDENCE:" in resp.prompt


def test_pipeline_image_query_passes_image_and_enriches_retrieval(pipeline, fixtures_dir: Path):
    pipe, gen, store = pipeline
    img = Image.open(fixtures_dir / "diagram.png")
    resp = pipe.answer("Explain this architecture", image=img, image_source="diagram.png", index_image=True)
    assert resp.image_analysis is not None and resp.image_analysis.image_kind == "diagram"
    assert gen.calls[-1]["images"] == [img]
    assert any(c.source == "cnn_notes.md" for c in resp.citations)  # image text pulled CNN notes
    assert "diagram.png" in store.sources()  # indexed on request


def test_pipeline_audio_query_uses_transcript(pipeline, fixtures_dir: Path):
    pipe, _, _ = pipeline
    from src.audio import AudioLoader

    clip = AudioLoader().load(fixtures_dir / "tone_16k_mono.wav")
    resp = pipe.answer(None, audio=clip)
    assert resp.transcript is not None and resp.transcript.backend == "fake_whisper"
    assert resp.query == "What do pooling layers do?"
    assert "transcription" in resp.timings_ms
    assert resp.citations


def test_pipeline_requires_a_question(pipeline):
    pipe, _, _ = pipeline
    with pytest.raises(ValueError):
        pipe.answer("   ")


def test_pipeline_no_evidence_when_threshold_high(pipeline):
    pipe, _, _ = pipeline
    resp = pipe.answer("What is the capital of France?", similarity_threshold=0.99)
    assert resp.evidence.level == EvidenceLevel.NONE
    assert resp.citations == []


def test_pipeline_audio_without_transcriber_raises(pipeline, fixtures_dir: Path):
    from src.audio import AudioDisabledError, AudioLoader

    pipe, _, _ = pipeline
    pipe.transcriber = None
    clip = AudioLoader().load(fixtures_dir / "tone_16k_mono.wav")
    with pytest.raises(AudioDisabledError):
        pipe.answer("q", audio=clip)
