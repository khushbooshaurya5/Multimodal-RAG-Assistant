"""Image loading utilities."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"})
MAX_SIDE = 1280  # keeps VLM token counts and memory bounded


class ImageDecodeError(RuntimeError):
    """Raised when an image cannot be opened."""


def supports(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _normalise(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def load_image(path: Path, max_side: int = MAX_SIDE) -> Image.Image:
    """Open an image file, fix EXIF orientation, convert to RGB and bound its size."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as img:
            img.load()
            return _normalise(img, max_side)
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(f"Not a valid image: {path.name}") from exc


def load_image_bytes(data: bytes, max_side: int = MAX_SIDE) -> Image.Image:
    """Open an image from raw bytes (UI/API uploads)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return _normalise(img, max_side)
    except UnidentifiedImageError as exc:
        raise ImageDecodeError("Uploaded bytes are not a valid image") from exc
