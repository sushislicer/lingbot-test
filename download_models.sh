#!/usr/bin/env bash
set -euo pipefail

# Downloads model checkpoints required by LingBot-VA.
#
# Designed for the layout:
#   /workspace/lingbot-test   (this repo)
#   /workspace/lingbot-va     (upstream, readonly)
#   /workspace/RoboTwin       (upstream, readonly)
#
# All downloads happen into this repo's own directories by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$ROOT_DIR/checkpoints}"
mkdir -p "$CHECKPOINTS_DIR"

HF_REPO_BASE="${HF_REPO_BASE:-robbyant/lingbot-va-base}"
HF_REPO_ROBOTWIN="${HF_REPO_ROBOTWIN:-robbyant/lingbot-va-posttrain-robotwin}"

# Optional: pin revisions for reproducibility
HF_REVISION_BASE="${HF_REVISION_BASE:-}"
HF_REVISION_ROBOTWIN="${HF_REVISION_ROBOTWIN:-}"

_download_one() {
  local repo_id="$1"
  local out_dir="$2"
  local revision="$3"

  echo "\n==> Downloading ${repo_id} -> ${out_dir}"
  mkdir -p "$out_dir"

  local extra_args=()
  if [[ -n "$revision" ]]; then
    extra_args+=(--revision "$revision")
  fi

  # Use the current Hugging Face Hub CLI (`hf`) as documented:
  # https://huggingface.co/docs/huggingface_hub/en/guides/cli
  #
  # NOTE: `hf download` options vary by version. This script intentionally
  # uses only the stable options shown by `hf download --help`:
  #   --repo-type, --revision, --include/--exclude, --cache-dir, --local-dir,
  #   --force-download, --token, --max-workers
  if ! command -v hf >/dev/null 2>&1; then
    echo "Error: 'hf' command not found in PATH." >&2
    echo "Install it via:" >&2
    echo "  python3 -m pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
  fi

  local token_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    token_args+=(--token "$HF_TOKEN")
  fi

  local cache_args=()
  if [[ -n "${HF_CACHE_DIR:-}" ]]; then
    cache_args+=(--cache-dir "$HF_CACHE_DIR")
  fi

  local max_workers="${HF_MAX_WORKERS:-8}"

  hf download "$repo_id" \
    --repo-type model \
    --local-dir "$out_dir" \
    --max-workers "$max_workers" \
    "${cache_args[@]}" \
    "${token_args[@]}" \
    "${extra_args[@]}"
}

_download_one "$HF_REPO_BASE" "$CHECKPOINTS_DIR/lingbot-va-base" "$HF_REVISION_BASE"
_download_one "$HF_REPO_ROBOTWIN" "$CHECKPOINTS_DIR/lingbot-va-posttrain-robotwin" "$HF_REVISION_ROBOTWIN"

cat <<EOF

Downloads complete.

Local checkpoints:
  $CHECKPOINTS_DIR/lingbot-va-base
  $CHECKPOINTS_DIR/lingbot-va-posttrain-robotwin

For running the RoboTwin eval, you typically want the server to use:
  WAN22_MODEL_PATH=$CHECKPOINTS_DIR/lingbot-va-posttrain-robotwin
EOF
