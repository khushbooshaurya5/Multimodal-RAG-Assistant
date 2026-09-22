"""Render answers, evidence indicators and retrieved sources."""

from __future__ import annotations

import streamlit as st

from src.schemas import Citation, EvidenceLevel, RAGResponse

_LEVEL_STYLE = {
    EvidenceLevel.NONE: ("🔴", "No supporting evidence"),
    EvidenceLevel.WEAK: ("🟠", "Weak evidence"),
    EvidenceLevel.MODERATE: ("🟡", "Moderate evidence"),
    EvidenceLevel.STRONG: ("🟢", "Strong evidence"),
}


def _location(c: Citation) -> str:
    if c.page is not None:
        return f"page {c.page}"
    if c.start_time is not None:
        end = f"–{c.end_time:.1f}s" if c.end_time is not None else ""
        return f"{c.start_time:.1f}s{end}"
    return ""


def render_evidence(response: RAGResponse) -> None:
    icon, label = _LEVEL_STYLE[response.evidence.level]
    st.markdown(f"**Evidence indicator:** {icon} {label}")
    st.caption(response.evidence.note)


def render_sources(response: RAGResponse) -> None:
    st.markdown("#### Retrieved sources")
    if not response.citations:
        st.info("No passages above the similarity threshold were retrieved.")
        return
    for citation in response.citations:
        loc = _location(citation)
        header = (
            f"[{citation.index}] {citation.source} · {citation.content_type.value}"
            f"{' · ' + loc if loc else ''} · similarity {citation.score:.2f}"
        )
        with st.expander(header, expanded=citation.index == 1):
            full = next((r.text for r in response.retrieved if r.metadata.chunk_id == citation.chunk_id), None)
            st.write(full or citation.snippet)
            st.caption(f"chunk id `{citation.chunk_id}`")


def render_multimodal_details(response: RAGResponse) -> None:
    if response.transcript is not None:
        with st.expander("🎙️ Transcript (Whisper)", expanded=False):
            st.write(response.transcript.text or "_(empty transcript)_")
            meta = f"backend `{response.transcript.backend}`"
            if response.transcript.language:
                meta += f" · language `{response.transcript.language}`"
            if response.transcript.duration_s:
                meta += f" · {response.transcript.duration_s:.1f}s"
            st.caption(meta)
            if response.transcript.segments:
                st.table(
                    [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in response.transcript.segments[:50]
                    ]
                )
    if response.image_analysis is not None:
        analysis = response.image_analysis
        title = "🖼️ Image analysis" + (" (fallback: no vision model)" if analysis.is_fallback else " (Qwen-VL)")
        with st.expander(title, expanded=False):
            if analysis.is_fallback:
                st.warning(analysis.description)
            else:
                if analysis.image_kind:
                    st.markdown(f"**Type:** {analysis.image_kind}")
                st.markdown(f"**Description:** {analysis.description}")
                if analysis.extracted_text:
                    st.markdown("**Visible text:**")
                    st.code(analysis.extracted_text)
                if analysis.objects:
                    st.markdown("**Objects:** " + ", ".join(analysis.objects))
            st.caption(f"backend `{analysis.backend}`")


def render_response(response: RAGResponse, show_prompt: bool = False) -> None:
    st.markdown("#### Answer")
    if response.is_fallback_generation:
        st.warning(
            "This answer comes from the extractive fallback: it is evidence text, not language-model output.",
            icon="⚠️",
        )
    st.markdown(response.answer)
    render_evidence(response)
    render_multimodal_details(response)
    render_sources(response)
    timings = " · ".join(f"{k} {v:.0f} ms" for k, v in response.timings_ms.items())
    st.caption(f"Generator `{response.generator_backend}` · {timings}")
    if show_prompt and response.prompt:
        with st.expander("Assembled prompt", expanded=False):
            st.code(response.prompt)
