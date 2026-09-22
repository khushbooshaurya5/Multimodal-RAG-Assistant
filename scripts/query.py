"""Ask the assistant from the command line.

python scripts/query.py "What do pooling layers do?"
python scripts/query.py "Explain this architecture" --image diagram.png
python scripts/query.py --audio question.wav --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.rag.service import AssistantService  # noqa: E402
from src.retrieval.filters import RetrievalFilters  # noqa: E402
from src.schemas import ContentType  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.vision.loader import load_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", nargs="?", help="Question text (optional if --audio is given)")
    parser.add_argument("--image", type=Path, help="Attach an image")
    parser.add_argument("--audio", type=Path, help="Spoken question (WAV/MP3/FLAC/...)")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--threshold", type=float, help="Similarity threshold override")
    parser.add_argument(
        "--types", help="Comma-separated content types to search (text,pdf,image,audio)"
    )
    parser.add_argument(
        "--index-image", action="store_true", help="Also add the image analysis to the index"
    )
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    service = AssistantService(settings)

    image = load_image(args.image) if args.image else None
    clip = service.load_audio(args.audio.read_bytes(), args.audio.name) if args.audio else None
    filters = RetrievalFilters(
        content_types=[ContentType(t.strip()) for t in args.types.split(",")]
        if args.types
        else None
    )
    response = service.query(
        args.question,
        image=image,
        image_source=args.image.name if args.image else "uploaded_image",
        audio=clip,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        filters=filters,
        index_image=args.index_image,
        include_prompt=args.show_prompt,
    )

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0

    if response.transcript:
        print(f"Transcript ({response.transcript.backend}): {response.transcript.text}\n")
    if response.image_analysis:
        print(
            f"Image analysis ({response.image_analysis.backend}): {response.image_analysis.description}\n"
        )
    print("ANSWER\n------")
    print(response.answer)
    print(f"\nEvidence: {response.evidence.level.value.upper()} — {response.evidence.note}")
    print("\nRETRIEVED SOURCES\n-----------------")
    for c in response.citations:
        loc = (
            f" p.{c.page}"
            if c.page
            else (f" {c.start_time:.1f}s" if c.start_time is not None else "")
        )
        print(
            f"[{c.index}] {c.source}{loc} ({c.content_type.value}, sim={c.score:.2f}): {c.snippet}"
        )
    if not response.citations:
        print("(none above threshold)")
    print("\nTimings (ms): " + json.dumps({k: round(v, 1) for k, v in response.timings_ms.items()}))
    if response.prompt:
        print("\nPROMPT\n------\n" + response.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
