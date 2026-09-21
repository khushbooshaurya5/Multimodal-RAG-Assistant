"""Context assembly: turn retrieved chunks and multimodal inputs into a prompt."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.prompts import ANSWER_TEMPLATE, NO_EVIDENCE_NOTE
from src.schemas import Citation, ImageAnalysis, RetrievedChunk, Transcript


def format_location(chunk: RetrievedChunk) -> str:
    """Human-readable location string (page / timestamp) for citations."""
    meta = chunk.metadata
    if meta.page is not None:
        return f"page {meta.page}"
    if meta.start_time is not None:
        end = f"–{meta.end_time:.1f}s" if meta.end_time is not None else ""
        return f"{meta.start_time:.1f}s{end}"
    return f"chunk {meta.chunk_index}"


def build_citations(retrieved: list[RetrievedChunk], snippet_chars: int = 240) -> list[Citation]:
    """Create display citations (numbered to match the prompt's [n] labels)."""
    citations: list[Citation] = []
    for i, item in enumerate(retrieved, start=1):
        text = item.text.strip().replace("\n", " ")
        snippet = text if len(text) <= snippet_chars else text[: snippet_chars - 1].rstrip() + "…"
        citations.append(
            Citation(
                index=i,
                source=item.metadata.source,
                content_type=item.metadata.content_type,
                chunk_id=item.metadata.chunk_id,
                score=item.score,
                page=item.metadata.page,
                start_time=item.metadata.start_time,
                end_time=item.metadata.end_time,
                snippet=snippet,
            )
        )
    return citations


@dataclass(slots=True)
class AssembledContext:
    """The prompt plus bookkeeping about what went into it."""

    prompt: str
    evidence_block: str
    citations: list[Citation]
    included: list[RetrievedChunk] = field(default_factory=list)
    truncated: bool = False


class ContextAssembler:
    """Formats evidence with numbered labels and metadata under a character budget."""

    def __init__(self, max_context_chars: int = 6000, max_image_text_chars: int = 1500) -> None:
        self.max_context_chars = max_context_chars
        self.max_image_text_chars = max_image_text_chars

    def format_evidence(self, retrieved: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], bool]:
        lines: list[str] = []
        included: list[RetrievedChunk] = []
        used = 0
        truncated = False
        for i, item in enumerate(retrieved, start=1):
            header = (
                f"[{i}] source={item.metadata.source} | type={item.metadata.content_type.value} | "
                f"{format_location(item)} | similarity={item.score:.2f}"
            )
            body = item.text.strip()
            entry = f"{header}\n{body}\n"
            if used + len(entry) > self.max_context_chars and included:
                truncated = True
                break
            lines.append(entry)
            included.append(item)
            used += len(entry)
        return ("\n".join(lines).strip() if lines else "(none)"), included, truncated

    def build(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        *,
        image_analysis: ImageAnalysis | None = None,
        transcript: Transcript | None = None,
        image_attached_to_model: bool = False,
    ) -> AssembledContext:
        evidence_block, included, truncated = self.format_evidence(retrieved)
        sections: list[str] = []

        if transcript is not None and transcript.text.strip():
            sections.append(
                "VOICE INPUT (transcribed with "
                f"{transcript.backend}{', language ' + transcript.language if transcript.language else ''}):\n"
                f"{transcript.text.strip()}\n"
            )
        if image_analysis is not None:
            if image_analysis.is_fallback:
                sections.append(
                    "ATTACHED IMAGE: no vision model was available, so the image content is unknown. "
                    f"File: {image_analysis.source}, {image_analysis.width}x{image_analysis.height}px.\n"
                )
            else:
                desc = image_analysis.to_searchable_text()
                if len(desc) > self.max_image_text_chars:
                    desc = desc[: self.max_image_text_chars].rstrip() + "…"
                label = (
                    "ATTACHED IMAGE (also provided to you directly; this is the analysis text)"
                    if image_attached_to_model
                    else "ATTACHED IMAGE ANALYSIS (produced by the vision model)"
                )
                sections.append(f"{label}:\n{desc}\n")
        if not included:
            sections.append(f"NOTE: {NO_EVIDENCE_NOTE}\n")

        multimodal_section = ("\n".join(sections) + "\n") if sections else ""
        prompt = ANSWER_TEMPLATE.format(
            multimodal_section=multimodal_section, evidence=evidence_block, question=question.strip()
        )
        return AssembledContext(
            prompt=prompt,
            evidence_block=evidence_block,
            citations=build_citations(included),
            included=included,
            truncated=truncated,
        )
