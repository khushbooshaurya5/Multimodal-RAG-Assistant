from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image
from src.config import get_settings
from src.schemas import ContentType, ImageAnalysis
from src.vision import (
    ImageAnalyzer,
    ImageDecodeError,
    ImageProcessor,
    MetadataImageAnalyzer,
    OpenAICompatibleAnalyzer,
    analysis_from_reply,
    build_image_analyzer,
    load_image,
    load_image_bytes,
    parse_analysis_json,
)


def test_load_image_normalises(fixtures_dir: Path):
    img = load_image(fixtures_dir / "diagram.png")
    assert img.mode == "RGB"
    assert img.size == (320, 200)


def test_load_image_bounds_size(tmp_path: Path):
    big = tmp_path / "big.png"
    Image.new("RGBA", (3000, 1500), "red").save(big)
    img = load_image(big, max_side=640)
    assert max(img.size) == 640 and img.mode == "RGB"


def test_invalid_image_bytes():
    with pytest.raises(ImageDecodeError):
        load_image_bytes(b"not an image")


def test_parse_analysis_json_handles_fences_and_prose():
    raw = 'Sure! ```json\n{"image_kind": "diagram", "objects": ["a", "b"]}\n```'
    assert parse_analysis_json(raw)["image_kind"] == "diagram"
    raw2 = 'Here you go: {"description": "x"} thanks'
    assert parse_analysis_json(raw2) == {"description": "x"}
    assert parse_analysis_json("just prose") == {"description": "just prose"}


def test_analysis_from_reply_builds_searchable_text(fixtures_dir: Path):
    img = load_image(fixtures_dir / "diagram.png")
    reply = json.dumps(
        {
            "image_kind": "Diagram",
            "description": "Two boxes connected by an arrow.",
            "visible_text": "Input Conv2D",
            "objects": ["box", "arrow", "box"],
        }
    )
    analysis = analysis_from_reply(reply, img, "diagram.png", "test")
    assert analysis.image_kind == "diagram"
    assert analysis.width == 320
    text = analysis.to_searchable_text()
    assert "Image type: diagram." in text
    assert "Visible text: Input Conv2D" in text
    assert "Objects: box, arrow." in text  # de-duplicated


def test_metadata_fallback_is_flagged(fixtures_dir: Path):
    analyzer = MetadataImageAnalyzer()
    assert isinstance(analyzer, ImageAnalyzer)
    processor = ImageProcessor(analyzer)
    doc = processor.process_file(fixtures_dir / "diagram.png")
    assert doc.content_type == ContentType.IMAGE
    assert doc.extra["vision_fallback"] is True
    assert "NOT analysed" in doc.text
    assert "diagram.png" in doc.text
    assert "No vision-language model" in processor.answer(
        load_image(fixtures_dir / "diagram.png"), "what?"
    )


def test_factory_returns_metadata_backend_in_offline_settings():
    analyzer = build_image_analyzer(get_settings())
    assert analyzer.is_fallback is True


class FakeVLMAnalyzer:
    """Test double standing in for a real VLM (implements the protocol)."""

    name = "fake_vlm"
    is_fallback = False

    def analyze(self, image: Image.Image, source: str) -> ImageAnalysis:
        return ImageAnalysis(
            source=source,
            description="A CNN diagram with an input box feeding a Conv2D box.",
            extracted_text="Input Conv2D CNN diagram fixture",
            objects=["input box", "conv2d box", "arrow"],
            image_kind="diagram",
            width=image.width,
            height=image.height,
            backend=self.name,
        )

    def answer(self, image: Image.Image, question: str, context: str | None = None) -> str:
        return f"answer to '{question}' with context={bool(context)}"


def test_processor_with_protocol_double(fixtures_dir: Path):
    processor = ImageProcessor(FakeVLMAnalyzer())
    doc = processor.process_file(fixtures_dir / "diagram.png")
    assert "Conv2D" in doc.text and doc.extra["image_kind"] == "diagram"
    assert doc.extra["vision_fallback"] is False


def test_openai_compatible_analyzer_round_trip(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the real HTTP client against an in-process mock server."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        reply = {
            "image_kind": "diagram",
            "description": "boxes",
            "visible_text": "Input",
            "objects": ["box"],
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    analyzer = OpenAICompatibleAnalyzer("http://mock/v1", model="qwen-vl", api_key="k")
    analyzer.client._client = httpx.Client(
        base_url="http://mock/v1", transport=httpx.MockTransport(handler)
    )

    analysis = analyzer.analyze(load_image(fixtures_dir / "diagram.png"), "diagram.png")
    assert analysis.image_kind == "diagram" and analysis.extracted_text == "Input"
    content = captured["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert captured["payload"]["model"] == "qwen-vl"


def test_openai_compatible_error_surfaces_as_model_unavailable(fixtures_dir: Path):
    from src.models.errors import ModelUnavailableError

    analyzer = OpenAICompatibleAnalyzer("http://mock/v1", model="m")
    analyzer.client._client = httpx.Client(
        base_url="http://mock/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down")),
    )
    with pytest.raises(ModelUnavailableError, match="503"):
        analyzer.analyze(load_image(fixtures_dir / "diagram.png"), "diagram.png")
