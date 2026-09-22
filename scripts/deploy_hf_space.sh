#!/usr/bin/env bash
# Deploy the app to a Hugging Face Space (Streamlit SDK).
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"
#   huggingface-cli login            # or export HF_TOKEN=hf_...
#
# Usage:
#   scripts/deploy_hf_space.sh <hf-username>/<space-name> [configs/hf_space.env]
#
# The script creates the Space if needed, assembles a clean copy of the repository with
# the Space README (YAML front matter), apt packages and the chosen .env profile, and
# pushes it. Re-run to redeploy.
set -euo pipefail

SPACE_ID="${1:?usage: $0 <hf-username>/<space-name> [env-profile]}"
PROFILE="${2:-configs/hf_space.env}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v huggingface-cli >/dev/null || { echo "huggingface-cli not found: pip install -U 'huggingface_hub[cli]'"; exit 1; }

echo "→ ensuring Space $SPACE_ID exists"
huggingface-cli repo create "$SPACE_ID" --type space --space_sdk streamlit -y >/dev/null 2>&1 || true

echo "→ assembling deployment tree in $WORK"
git -C "$ROOT" archive HEAD | tar -x -C "$WORK"
cp "$ROOT/deploy/huggingface/README.md" "$WORK/README.md"
cp "$ROOT/deploy/huggingface/packages.txt" "$WORK/packages.txt"
cp "$ROOT/$PROFILE" "$WORK/.env"
rm -rf "$WORK/.github" "$WORK/reports" "$WORK/docs/screenshots" "$WORK/.gitignore"

echo "→ pushing to https://huggingface.co/spaces/$SPACE_ID"
huggingface-cli upload "$SPACE_ID" "$WORK" . --repo-type space --delete "*" \
  --commit-message "Deploy from $(git -C "$ROOT" rev-parse --short HEAD)"

echo "✓ deployed: https://huggingface.co/spaces/$SPACE_ID"
echo "  Set MRAG_* overrides under Settings → Variables if you want different models."
