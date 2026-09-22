"""Generate small deterministic test fixtures (run once; outputs are committed).

python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

FIXTURES = Path(__file__).parent


def make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal valid PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    pages_id_placeholder = len(objects) + 2 * len(pages) + 1
    for text in pages:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_id = add(
            (
                f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
        page_ids.append(page_id)
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    assert pages_id == pages_id_placeholder
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def make_wav(path: Path, seconds: float = 1.0, freq: float = 440.0, rate: int = 16000) -> None:
    """Write a mono 16-bit PCM sine tone."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(
            b"".join(
                struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
                for i in range(frames)
            )
        )


def make_stereo_wav(path: Path, seconds: float = 0.5, rate: int = 44100) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(
            b"".join(
                struct.pack(
                    "<hh",
                    int(0.2 * 32767 * math.sin(2 * math.pi * 300 * i / rate)),
                    int(0.2 * 32767 * math.sin(2 * math.pi * 600 * i / rate)),
                )
                for i in range(frames)
            )
        )


def make_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 140, 90], outline="black", width=3)
    draw.rectangle([180, 20, 300, 90], outline="black", width=3)
    draw.line([140, 55, 180, 55], fill="black", width=3)
    draw.text((30, 45), "Input", fill="black")
    draw.text((190, 45), "Conv2D", fill="black")
    draw.text((20, 140), "CNN diagram fixture", fill="black")
    img.save(path)


if __name__ == "__main__":
    make_pdf(
        FIXTURES / "sample.pdf",
        [
            "Convolutional neural networks use convolution layers and pooling.",
            "Transformers rely on self-attention over token sequences.",
        ],
    )
    make_wav(FIXTURES / "tone_16k_mono.wav")
    make_stereo_wav(FIXTURES / "tone_44k_stereo.wav")
    make_image(FIXTURES / "diagram.png")
    (FIXTURES / "cnn_notes.md").write_text(
        "# Convolutional Neural Networks\n\n"
        "A convolutional neural network (CNN) applies learned filters over an input image. "
        "Each convolution layer produces feature maps that highlight local patterns such as "
        "edges and textures. Pooling layers downsample feature maps, giving translation "
        "invariance and reducing computation.\n\n"
        "## Architecture\n\n"
        "A typical CNN stacks convolution, activation (ReLU) and pooling blocks, followed by "
        "fully connected layers that map features to class scores. Batch normalisation "
        "stabilises training and dropout reduces overfitting.\n\n"
        "## Applications\n\n"
        "CNNs power image classification, object detection and semantic segmentation.\n",
        encoding="utf-8",
    )
    (FIXTURES / "transformer_notes.txt").write_text(
        "Transformers replace recurrence with self-attention. Multi-head attention lets each "
        "token attend to every other token in the sequence, and positional encodings inject "
        "order information. The encoder-decoder architecture was introduced in the paper "
        "'Attention Is All You Need' in 2017. Large language models such as GPT are decoder-only "
        "transformers trained on next-token prediction.\n",
        encoding="utf-8",
    )
    print("fixtures written to", FIXTURES)
