"""Small hashing helpers used for stable IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str, length: int = 16) -> str:
    """Return a truncated hex SHA-256 of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, length: int = 16, chunk_size: int = 1 << 20) -> str:
    """Return a truncated hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()[:length]
