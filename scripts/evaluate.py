"""Evaluate retrieval, groundedness, transcription and latency on the labelled dataset.

    python scripts/evaluate.py                       # uses .env configuration
    python scripts/evaluate.py --top-k 3 --judge     # LLM-as-judge when an LLM backend is configured

The corpus is indexed into a temporary directory so your working index is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.evaluation.runner import EvaluationRunner, render_markdown  # noqa: E402
from src.rag.service import AssistantService  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "eval" / "eval_dataset.json")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override the similarity threshold")
    parser.add_argument("--judge", action="store_true", help="Also score groundedness with the configured LLM")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--tag", default="", help="Suffix for the report filenames, e.g. 'offline'")
    args = parser.parse_args()

    base = get_settings()
    configure_logging(base.log_level)
    with tempfile.TemporaryDirectory(prefix="mrag-eval-") as tmp:
        overrides = {"index_dir": Path(tmp) / "index", "data_dir": Path(tmp) / "data"}
        if args.threshold is not None:
            overrides["similarity_threshold"] = args.threshold
        settings = Settings(**{**base.model_dump(), **overrides})
        service = AssistantService(settings)
        runner = EvaluationRunner(service, args.dataset, top_k=args.top_k, use_llm_judge=args.judge)
        report = runner.run()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = args.out_dir / f"eval_report{suffix}.json"
    md_path = args.out_dir / f"eval_report{suffix}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(render_markdown(report))
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
