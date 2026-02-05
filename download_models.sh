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

  # Prefer the newer `hf` CLI. Fallback to `huggingface-cli` if needed.
  # If neither exists, require `python3 -m pip install -U 'huggingface_hub[cli]'`.
  local hf_cmd=""
  if command -v hf >/dev/null 2>&1; then
    hf_cmd="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cmd="huggingface-cli"
  else
    echo "Error: neither 'hf' nor 'huggingface-cli' found in PATH." >&2
    echo "Install the Hugging Face CLI via:" >&2
    echo "  python3 -m pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
  fi

  # --local-dir-use-symlinks False makes the output self-contained (good for rsync / NFS)
  if [[ "$hf_cmd" == "hf" ]]; then
    hf download "$repo_id" \
      --local-dir "$out_dir" \
      --local-dir-use-symlinks False \
      --resume-download \
      "${extra_args[@]}"
  else
    huggingface-cli download "$repo_id" \
      --local-dir "$out_dir" \
      --local-dir-use-symlinks False \
      --resume-download \
      "${extra_args[@]}"
  fi
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
