#!/usr/bin/env bash
set -euo pipefail

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
SERVER_NPROC=${SERVER_NPROC:-1}

SAVE_ROOT=${SAVE_ROOT:-visualization/}
mkdir -p "$SAVE_ROOT"

export LINGBOT_VA_ROOT=${LINGBOT_VA_ROOT:-/workspace/lingbot-va}
export WAN22_MODEL_PATH=${WAN22_MODEL_PATH:-}

python -m torch.distributed.run \
  --nproc_per_node "$SERVER_NPROC" \
  --master_port "$MASTER_PORT" \
  scripts/va_server_entry.py \
  --config-name robotwin \
  --port "$START_PORT" \
  --save_root "$SAVE_ROOT"

