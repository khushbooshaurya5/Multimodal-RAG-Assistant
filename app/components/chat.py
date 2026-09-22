"""Chat history rendering."""

from __future__ import annotations

import streamlit as st
from src.schemas import RAGResponse

from app.components.sources import render_response


def render_history(show_prompt: bool) -> None:
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                if turn.get("image") is not None:
                    st.image(turn["image"], caption=turn.get("image_name"), width=280)
                if turn.get("audio") is not None:
                    st.audio(turn["audio"])
                st.markdown(turn["text"] or "_(spoken question)_")
            else:
                response: RAGResponse = turn["response"]
                render_response(response, show_prompt=show_prompt)
