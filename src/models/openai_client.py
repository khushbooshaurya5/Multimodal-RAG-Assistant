"""Minimal OpenAI-compatible chat-completions client.

Lets the vision and generation layers talk to any server exposing the OpenAI
``/v1/chat/completions`` contract (vLLM, Ollama, LM Studio, DashScope, ...),
which is the standard way to serve Qwen-VL models that are too large to run
inside this process.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import httpx

from src.models.errors import ModelUnavailableError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def image_to_data_url(image: Any, fmt: str = "PNG") -> str:
    """Encode a PIL image as a ``data:`` URL for the ``image_url`` content type."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


class OpenAICompatibleClient:
    """Thin synchronous client for ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s)

    @staticmethod
    def build_user_content(
        text: str, images: list[Any] | None = None
    ) -> list[dict[str, Any]] | str:
        """Build a multimodal ``content`` array (or plain string when no images)."""
        if not images:
            return text
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}} for img in images
        ]
        content.append({"type": "text", "text": text})
        return content

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat request and return the assistant message text.

        Raises:
            ModelUnavailableError: on connection errors or non-2xx responses.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelUnavailableError(
                f"{self.base_url} returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"Cannot reach OpenAI-compatible endpoint {self.base_url}: {exc}"
            ) from exc
        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"Malformed response from {self.base_url}: {data}") from exc
        content = message.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content).strip()

    def close(self) -> None:
        self._client.close()
