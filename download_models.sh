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

  # --local-dir-use-symlinks False makes the output self-contained (good for rsync / NFS)
  huggingface-cli download "$repo_id" \
    --local-dir "$out_dir" \
    --local-dir-use-symlinks False \
    --resume-download \
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
