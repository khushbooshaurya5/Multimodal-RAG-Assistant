"""Streamlit front-end for the Multimodal RAG Assistant.

Run with::

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/main.py` from the repository root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from src.audio.loader import AudioDecodeError  # noqa: E402
from src.audio.transcriber import AudioDisabledError  # noqa: E402
from src.models.errors import ModelUnavailableError  # noqa: E402
from src.retrieval.filters import RetrievalFilters  # noqa: E402
from src.vision.loader import ImageDecodeError, load_image_bytes  # noqa: E402

from app.components.chat import render_history  # noqa: E402
from app.components.sources import render_response  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402
from app.ui.state import clear_history, get_service, init_state  # noqa: E402

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠", layout="wide")
init_state()
service = get_service()
controls = render_sidebar(service)

st.title("🧠 Multimodal RAG Assistant")
st.caption(
    "Qwen-VL · Whisper · FAISS — ask with text, images or voice; answers cite retrieved evidence."
)

# --- Attachments for the next question --------------------------------------
with st.container(border=True):
    col_img, col_audio = st.columns(2)
    with col_img:
        image_upload = st.file_uploader(
            "Attach an image (optional)",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="query_image",
        )
        if image_upload is not None:
            st.image(image_upload, width=240)
    with col_audio:
        audio_source = st.radio(
            "Voice question (optional)", ["None", "Record", "Upload"], horizontal=True
        )
        audio_bytes: bytes | None = None
        audio_name = "recording.wav"
        if audio_source == "Record":
            recorded = st.audio_input("Record your question")
            if recorded is not None:
                audio_bytes, audio_name = recorded.getvalue(), "recording.wav"
        elif audio_source == "Upload":
            uploaded_audio = st.file_uploader(
                "Upload audio", type=["wav", "mp3", "flac", "m4a", "ogg"], key="query_audio"
            )
            if uploaded_audio is not None:
                audio_bytes, audio_name = uploaded_audio.getvalue(), uploaded_audio.name
        if audio_bytes and service.transcriber is None:
            st.warning(
                "Speech recognition is disabled in this configuration; the audio will not be transcribed."
            )

    cols = st.columns([1, 6])
    if cols[0].button("Clear chat"):
        clear_history()
        st.rerun()
    cols[1].caption(
        "Type a question below, or leave it empty and use only voice. Attached images are analysed by the "
        "vision backend and, when supported, passed to the generator directly."
    )

# --- History ----------------------------------------------------------------
render_history(show_prompt=controls.show_prompt)

# --- New question -----------------------------------------------------------
typed = st.chat_input("Ask about your documents, the attached image, or speak your question…")
submit_voice_only = False
if typed is None and audio_bytes and st.session_state.get("_last_audio") != audio_bytes:
    # A new recording without typed text: treat as a voice-only question.
    submit_voice_only = st.button("Ask with voice only", type="primary")

if typed is not None or submit_voice_only:
    question = (typed or "").strip()
    pil_image = None
    image_name = None
    if image_upload is not None:
        try:
            pil_image = load_image_bytes(image_upload.getvalue())
            image_name = image_upload.name
        except ImageDecodeError as exc:
            st.error(str(exc))
            st.stop()
    clip = None
    if audio_bytes:
        try:
            clip = service.load_audio(audio_bytes, audio_name)
            st.session_state["_last_audio"] = audio_bytes
        except AudioDecodeError as exc:
            st.error(str(exc))
            st.stop()

    with st.chat_message("user"):
        if pil_image is not None:
            st.image(pil_image, caption=image_name, width=280)
        if audio_bytes:
            st.audio(audio_bytes)
        st.markdown(question or "_(spoken question)_")

    with st.chat_message("assistant"):
        try:
            with st.spinner("Transcribing / analysing / retrieving / generating…"):
                response = service.query(
                    question or None,
                    image=pil_image,
                    image_source=image_name or "uploaded_image",
                    audio=clip,
                    top_k=controls.top_k,
                    similarity_threshold=controls.similarity_threshold,
                    filters=RetrievalFilters(content_types=controls.content_types),
                    rerank=controls.rerank,
                    index_image=controls.index_uploaded_image,
                    include_prompt=controls.show_prompt,
                )
        except ValueError as exc:
            st.error(f"{exc}")
            st.stop()
        except AudioDisabledError as exc:
            st.error(str(exc))
            st.stop()
        except ModelUnavailableError as exc:
            st.error(f"Model backend error: {exc}")
            st.stop()
        render_response(response, show_prompt=controls.show_prompt)

    st.session_state["history"].append(
        {
            "role": "user",
            "text": question,
            "image": pil_image,
            "image_name": image_name,
            "audio": audio_bytes,
        }
    )
    st.session_state["history"].append({"role": "assistant", "response": response})
