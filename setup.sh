#!/usr/bin/env bash
set -euo pipefail

# This repo is meant to live at (example):
#   /workspace/lingbot-test
# while the *actual* upstream repos live elsewhere, e.g.:
#   /workspace/lingbot-va   (readonly)
#   /workspace/RoboTwin     (readonly)
#
# We keep this repo as the "glue" layer and use env vars / symlinks to point
# at the upstream repos.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_LINGBOT_VA_ROOT="/workspace/lingbot-va"
DEFAULT_ROBOWIN_ROOT="/workspace/RoboTwin"

# Resolve LingBot-VA root
if [[ -n "${LINGBOT_VA_ROOT:-}" ]]; then
  _LINGBOT_VA_ROOT="$LINGBOT_VA_ROOT"
elif [[ -d "$ROOT_DIR/lingbot-va" ]]; then
  _LINGBOT_VA_ROOT="$ROOT_DIR/lingbot-va"
else
  _LINGBOT_VA_ROOT="$DEFAULT_LINGBOT_VA_ROOT"
fi

# Resolve RoboTwin root
if [[ -n "${ROBOWIN_ROOT:-}" ]]; then
  _ROBOWIN_ROOT="$ROBOWIN_ROOT"
elif [[ -d "$ROOT_DIR/RoboTwin" ]]; then
  _ROBOWIN_ROOT="$ROOT_DIR/RoboTwin"
elif [[ -d "$ROOT_DIR/robotwin" ]]; then
  _ROBOWIN_ROOT="$ROOT_DIR/robotwin"
else
  _ROBOWIN_ROOT="$DEFAULT_ROBOWIN_ROOT"
fi

if [[ ! -d "${_LINGBOT_VA_ROOT}" ]]; then
  echo "Error: LINGBOT_VA_ROOT not found: ${_LINGBOT_VA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${_ROBOWIN_ROOT}" ]]; then
  echo "Error: ROBOWIN_ROOT not found: ${_ROBOWIN_ROOT}" >&2
  exit 1
fi

# Create local convenience symlinks (optional but simplifies relative paths).
# These do NOT modify upstream repos.
if [[ ! -e "$ROOT_DIR/lingbot-va" ]]; then
  ln -s "${_LINGBOT_VA_ROOT}" "$ROOT_DIR/lingbot-va"
fi
if [[ ! -e "$ROOT_DIR/RoboTwin" && ! -e "$ROOT_DIR/robotwin" ]]; then
  ln -s "${_ROBOWIN_ROOT}" "$ROOT_DIR/RoboTwin"
fi

mkdir -p scripts checkpoints results logs

chmod +x scripts/launch_client.sh scripts/launch_server.sh || true

cat <<EOF
Setup complete.

Resolved paths:
  LINGBOT_VA_ROOT=${_LINGBOT_VA_ROOT}
  ROBOWIN_ROOT=${_ROBOWIN_ROOT}

Next:
  bash install.sh
  bash download_models.sh
  bash run_experiment.sh
EOF
