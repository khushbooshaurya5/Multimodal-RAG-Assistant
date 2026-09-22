"""Sidebar: backend status, retrieval controls, upload / index management."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.rag.service import AssistantService
from src.schemas import ContentType


@dataclass(slots=True)
class RetrievalControls:
    top_k: int
    similarity_threshold: float
    rerank: bool
    content_types: list[ContentType] | None
    index_uploaded_image: bool
    show_prompt: bool


def _render_status(service: AssistantService) -> None:
    status = service.status()
    st.subheader("Model configuration")
    st.caption("Configured through environment variables (`.env`). See `configs/`.")
    rows = {
        "Device": status.device,
        "Embeddings": status.embedder,
        "Generator (LLM)": status.generator,
        "Vision": status.vision or "disabled",
        "Speech": status.transcriber or "disabled",
        "Reranker": status.reranker or "off",
    }
    for label, value in rows.items():
        st.markdown(f"**{label}:** `{value}`")

    fallbacks = status.fallbacks
    if fallbacks.get("embedder"):
        st.warning("Embeddings: hashing fallback (lexical only, no semantic model).", icon="⚠️")
    if fallbacks.get("generator"):
        st.warning("Generator: extractive fallback - answers are evidence excerpts, not LLM output.", icon="⚠️")
    if fallbacks.get("vision"):
        st.warning("Vision: metadata-only fallback - images are NOT understood by a model.", icon="⚠️")
    if fallbacks.get("audio_disabled"):
        st.info("Speech recognition is disabled (MRAG_WHISPER_BACKEND=none).", icon="🔇")
    for warning in status.warnings:
        st.error(warning, icon="❌")


def _render_retrieval_controls(service: AssistantService) -> RetrievalControls:
    settings = service.settings
    st.subheader("Retrieval")
    top_k = st.slider("Top-k passages", 1, 20, settings.top_k, help="How many chunks to retrieve.")
    threshold = st.slider(
        "Similarity threshold",
        -1.0,
        1.0,
        float(settings.similarity_threshold),
        0.01,
        help="Chunks with cosine similarity below this are dropped. Lower it for the hashing embedder.",
    )
    rerank = st.toggle(
        "Cross-encoder rerank",
        value=bool(service.reranker is not None),
        disabled=service.reranker is None,
        help="Requires MRAG_RERANK=true and a downloadable cross-encoder.",
    )
    selected = st.multiselect(
        "Restrict to content types",
        options=[c.value for c in ContentType],
        default=[],
        help="Leave empty to search everything.",
    )
    index_image = st.toggle("Index attached images", value=False, help="Add the analysis of a query image to the index.")
    show_prompt = st.toggle("Show assembled prompt", value=False)
    return RetrievalControls(
        top_k=top_k,
        similarity_threshold=threshold,
        rerank=rerank,
        content_types=[ContentType(c) for c in selected] or None,
        index_uploaded_image=index_image,
        show_prompt=show_prompt,
    )


def _render_index_controls(service: AssistantService) -> None:
    st.subheader("Knowledge base")
    uploads = st.file_uploader(
        "Upload documents, images or audio to index",
        type=["txt", "md", "pdf", "png", "jpg", "jpeg", "webp", "wav", "mp3", "flac", "m4a", "ogg"],
        accept_multiple_files=True,
        key="index_uploads",
    )
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        with st.spinner("Extracting, chunking, embedding…"):
            for upload in uploads or []:
                result = service.ingest_upload(upload.name, upload.getvalue())
                if result.error:
                    st.error(f"{result.source}: {result.error}")
                elif result.skipped:
                    st.info(f"{result.source}: already indexed")
                else:
                    st.success(f"{result.source}: {result.num_chunks} chunks ({result.content_type})")

    stats = service.index_stats()
    st.caption(f"{stats['num_vectors']} vectors · {len(stats['sources'])} sources · saved in `{stats['index_dir']}`")
    if stats["sources"]:
        with st.expander("Indexed sources", expanded=False):
            for source, count in sorted(stats["sources"].items()):
                col_a, col_b = st.columns([4, 1])
                col_a.write(f"`{source}` — {count} chunks")
                if col_b.button("✕", key=f"rm_{source}", help=f"Remove {source} from the index"):
                    service.remove_source(source)
                    st.rerun()
        if st.button("Clear entire index", type="secondary", use_container_width=True):
            service.clear_index()
            st.rerun()


def render_sidebar(service: AssistantService) -> RetrievalControls:
    with st.sidebar:
        st.title("⚙️ Settings")
        _render_status(service)
        st.divider()
        controls = _render_retrieval_controls(service)
        st.divider()
        _render_index_controls(service)
    return controls
