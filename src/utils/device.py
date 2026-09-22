"""Device selection with graceful CPU fallback.

``torch`` is imported lazily so that the pure-Python parts of the system
(chunking, FAISS store, hashing embeddings, extractive answers) work in
environments where PyTorch is not installed.
"""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


def torch_available() -> bool:
    """Return ``True`` if PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference (``auto``/``cpu``/``cuda``/``mps``) to a real device.

    Falls back to ``cpu`` with a warning when the requested accelerator is
    unavailable, so the application keeps running on machines without a GPU.
    """
    if preference == "cpu" or not torch_available():
        if preference not in ("auto", "cpu"):
            logger.warning("PyTorch not available; using CPU instead of %s", preference)
        return "cpu"

    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if preference == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if preference == "cuda" and cuda_ok:
        return "cuda"
    if preference == "mps" and mps_ok:
        return "mps"

    logger.warning("Requested device '%s' is unavailable; falling back to CPU", preference)
    return "cpu"


def preferred_dtype(device: str):
    """Pick a sensible default dtype for large model weights on ``device``."""
    import torch

    if device == "cuda":
        return torch.float16
    return torch.float32
