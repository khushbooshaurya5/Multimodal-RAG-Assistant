"""Centralised configuration.

All tunable values are read from environment variables (or a ``.env`` file)
through :class:`Settings`. Nothing else in the code base reads ``os.environ``
directly, which keeps configuration discoverable and testable.

Every variable uses the ``MRAG_`` prefix, e.g. ``MRAG_TOP_K=5``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackend = Literal["sentence_transformers", "hashing"]
WhisperBackend = Literal["transformers", "none"]
VisionBackend = Literal["transformers", "openai_compatible", "metadata"]
LLMBackend = Literal["transformers", "openai_compatible", "extractive"]
DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="MRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    log_level: str = "INFO"
    device: DeviceChoice = "auto"
    fallback_on_error: bool = Field(
        True,
        description=(
            "If a configured neural backend fails to load, switch to the documented offline "
            "fallback (and report it) instead of failing to start."
        ),
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/index")

    # --- Embeddings --------------------------------------------------------
    embedding_backend: EmbeddingBackend = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(384, ge=16, description="Dimension for the hashing backend")

    # --- Speech ------------------------------------------------------------
    whisper_backend: WhisperBackend = "transformers"
    whisper_model: str = "openai/whisper-small"
    whisper_language: str | None = None

    # --- Vision ------------------------------------------------------------
    vision_backend: VisionBackend = "transformers"
    vision_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    vision_max_new_tokens: int = Field(512, ge=16)

    # --- LLM ---------------------------------------------------------------
    llm_backend: LLMBackend = "transformers"
    llm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    llm_max_new_tokens: int = Field(512, ge=16)
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # --- OpenAI-compatible endpoint ---------------------------------------
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str | None = None
    openai_timeout_s: float = Field(120.0, gt=0)

    # --- Retrieval ---------------------------------------------------------
    chunk_size: int = Field(400, ge=50, description="Target chunk size in characters")
    chunk_overlap: int = Field(80, ge=0)
    top_k: int = Field(5, ge=1, le=50)
    similarity_threshold: float = Field(0.25, ge=-1.0, le=1.0)
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(8080, ge=1, le=65535)

    @field_validator("whisper_language", "openai_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, value: str | None) -> str | None:
        """Treat empty strings in ``.env`` as unset."""
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not exist."""
        for path in (self.raw_dir, self.processed_dir, self.index_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (useful in tests that mutate the environment)."""
    get_settings.cache_clear()
