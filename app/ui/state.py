"""Streamlit session-state helpers and cached service construction."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import get_settings
from src.rag.service import AssistantService
from src.utils.logging import configure_logging


@st.cache_resource(show_spinner="Loading models and index…")
def get_service() -> AssistantService:
    """Build the assistant once per Streamlit process (models are heavy)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return AssistantService(settings)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "history": [],  # list[dict]: {"role": "user"|"assistant", ...}
        "pending_image": None,
        "pending_image_name": None,
        "pending_audio": None,
        "pending_audio_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_history() -> None:
    st.session_state["history"] = []
