"""Index files or directories into the FAISS store.

python scripts/ingest.py data/raw
python scripts/ingest.py notes.md diagram.png lecture.mp3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.rag.service import AssistantService  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to index")
    parser.add_argument(
        "--no-recursive", action="store_true", help="Do not descend into sub-directories"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    service = AssistantService(settings)

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            pattern = "*" if args.no_recursive else "**/*"
            files.extend(
                p
                for p in sorted(path.glob(pattern))
                if p.is_file() and service.ingestion.supports(p)
            )
        elif path.is_file():
            files.append(path)
        else:
            print(f"skip (not found): {path}", file=sys.stderr)

    results = service.ingest_files(files)
    failures = 0
    for r in results:
        if r.error:
            failures += 1
            print(f"ERROR   {r.source}: {r.error}")
        elif r.skipped:
            print(f"SKIP    {r.source}: already indexed")
        else:
            print(f"OK      {r.source}: {r.num_chunks} chunks [{r.content_type}]")
    stats = service.index_stats()
    print(
        f"\nIndex: {stats['num_vectors']} vectors from {len(stats['sources'])} sources → {stats['index_dir']}"
    )
    return 1 if failures and failures == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
