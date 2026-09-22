"""Pre-download the configured model weights into the Hugging Face cache.

Useful before going offline or when building a Docker image:

    python scripts/download_models.py            # everything configured in .env
    python scripts/download_models.py --only whisper embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.models.errors import ModelUnavailableError  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=["embeddings", "whisper", "vision", "llm", "reranker"])
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    wanted = set(args.only or ["embeddings", "whisper", "vision", "llm", "reranker"])
    failures = 0

    def attempt(label: str, fn) -> None:
        nonlocal failures
        try:
            fn()
            print(f"OK    {label}")
        except ModelUnavailableError as exc:
            failures += 1
            print(f"FAIL  {label}: {exc}")

    if "embeddings" in wanted and settings.embedding_backend == "sentence_transformers":
        from src.models.embedding_loader import load_sentence_transformer

        attempt(settings.embedding_model, lambda: load_sentence_transformer(settings.embedding_model, "cpu"))
    if "reranker" in wanted and settings.rerank:
        from src.models.embedding_loader import load_cross_encoder

        attempt(settings.rerank_model, lambda: load_cross_encoder(settings.rerank_model, "cpu"))
    if "whisper" in wanted and settings.whisper_backend == "transformers":
        from src.models.whisper_loader import load_whisper

        attempt(settings.whisper_model, lambda: load_whisper(settings.whisper_model, "cpu"))
    models = set()
    if "vision" in wanted and settings.vision_backend == "transformers":
        models.add(settings.vision_model)
    if "llm" in wanted and settings.llm_backend == "transformers":
        models.add(settings.llm_model)
    for name in sorted(models):
        from src.models.qwen_vl_loader import load_qwen_vl

        attempt(name, lambda name=name: load_qwen_vl(name, "cpu"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
