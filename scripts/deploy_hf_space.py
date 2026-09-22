"""Deploy the app to a Hugging Face Space (Docker SDK) with the huggingface_hub library.

Prerequisites:
    pip install -U huggingface_hub
    export HF_TOKEN=hf_...        # a token with "write" permission, from https://huggingface.co/settings/tokens

Usage:
    python scripts/deploy_hf_space.py multimodal-rag-assistant          # under the logged-in account
    python scripts/deploy_hf_space.py <user>/<space> --profile configs/hf_space.env
    python scripts/deploy_hf_space.py <user>/<space> --dry-run           # assemble only, no upload

The script creates the Space if it does not exist, assembles a clean copy of the tracked
repository files with the Space card (YAML front matter, Docker SDK), the chosen .env
profile and a CPU-only PyTorch index, uploads it in one commit, and prints the Space URL.
The Space builds the repository's Dockerfile and serves Streamlit on port 8501.
Every failure is reported with the server's message; nothing is swallowed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"
EXCLUDE = {".github", "reports", "docs/screenshots", ".gitignore", ".env.example"}


def assemble(work: Path, profile: Path) -> None:
    """Export tracked files at HEAD into ``work`` and overlay the Space-specific files."""
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], check=True, capture_output=True).stdout
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp.write(archive)
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(work, filter="data")
    finally:
        tar_path.unlink(missing_ok=True)

    for rel in EXCLUDE:
        target = work / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    shutil.copy(ROOT / "deploy" / "huggingface" / "README.md", work / "README.md")
    shutil.copy(profile, work / ".env")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    (work / "requirements.txt").write_text(f"{CPU_INDEX}\n{requirements}", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space_id", help="<space-name> (created under the logged-in account) or <hf-username>/<space-name>")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs" / "hf_space.env")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--dry-run", action="store_true", help="Assemble the tree and list it; do not upload")
    args = parser.parse_args()

    if not args.profile.is_file():
        print(f"profile not found: {args.profile}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="mrag-space-"))
    try:
        assemble(work, args.profile)
        files = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file())
        print(f"Assembled {len(files)} files in {work}")
        if args.dry_run:
            print("\n".join(files))
            return 0

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError:
            print("huggingface_hub is not installed: pip install -U huggingface_hub", file=sys.stderr)
            return 1

        api = HfApi()  # token from HF_TOKEN or `hf auth login` / `huggingface-cli login`
        try:
            who = api.whoami()
        except Exception as exc:  # noqa: BLE001 - any auth problem is fatal and must be shown
            print(f"Not authenticated with Hugging Face: {exc}\nSet HF_TOKEN or run `hf auth login`.", file=sys.stderr)
            return 1
        user = who.get("name")
        print(f"Authenticated as {user}")
        space_id = args.space_id if "/" in args.space_id else f"{user}/{args.space_id}"
        namespace = space_id.split("/")[0]
        if namespace != user and namespace not in {org.get("name") for org in who.get("orgs", [])}:
            print(
                f"Warning: '{namespace}' is not your account ({user}) or one of your organisations; "
                "creating the Space there will be refused.",
                file=sys.stderr,
            )

        try:
            url = api.create_repo(
                repo_id=space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
            )
            print(f"Space ready: {url}")
        except HfHubHTTPError as exc:
            print(f"Could not create Space {space_id}: {exc}", file=sys.stderr)
            return 1

        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        try:
            api.upload_folder(
                repo_id=space_id,
                repo_type="space",
                folder_path=str(work),
                commit_message=f"Deploy from {head}",
                delete_patterns=["*"],
            )
        except HfHubHTTPError as exc:
            print(f"Upload failed: {exc}", file=sys.stderr)
            return 1
        print(f"\nDeployed: https://huggingface.co/spaces/{space_id}")
        print("Watch the build under the Space's 'Logs' tab; the first Docker build takes 5-10 minutes.")
        print("Override MRAG_* settings under Settings → Variables and secrets if needed.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
