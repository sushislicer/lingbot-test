#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_LINGBOT_VA_ROOT="/workspace/lingbot-va"
DEFAULT_ROBOWIN_ROOT="/workspace/RoboTwin"

export LINGBOT_VA_ROOT="${LINGBOT_VA_ROOT:-$DEFAULT_LINGBOT_VA_ROOT}"
export ROBOWIN_ROOT="${ROBOWIN_ROOT:-$DEFAULT_ROBOWIN_ROOT}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$ROOT_DIR/checkpoints}"
export WAN22_MODEL_PATH="${WAN22_MODEL_PATH:-$CHECKPOINTS_DIR/lingbot-va-posttrain-robotwin}"

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
mkdir -p "$RESULTS_DIR"

TASK_NAME="${TASK_NAME:-adjust_bottle}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TEST_NUM="${TEST_NUM:-100}"

PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29061}"

# GPU split default (uses both GPUs on a 2×GPU machine):
# - server on GPU0
# - simulation client on GPU1
SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-0}"
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-1}"

# If you want to run the server itself distributed across multiple GPUs, set:
#   SERVER_NPROC=2
#   SERVER_CUDA_VISIBLE_DEVICES=0,1
SERVER_NPROC="${SERVER_NPROC:-1}"

SERVER_STARTUP_SLEEP="${SERVER_STARTUP_SLEEP:-30}"

export PYTHONPATH="$ROOT_DIR/scripts:$LINGBOT_VA_ROOT:$ROBOWIN_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$LINGBOT_VA_ROOT" ]]; then
  echo "Error: LINGBOT_VA_ROOT not found: $LINGBOT_VA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ROBOWIN_ROOT" ]]; then
  echo "Error: ROBOWIN_ROOT not found: $ROBOWIN_ROOT" >&2
  exit 1
fi
if [[ ! -d "$WAN22_MODEL_PATH" ]]; then
  echo "Error: WAN22_MODEL_PATH not found: $WAN22_MODEL_PATH" >&2
  echo "Hint: run: bash download_models.sh" >&2
  exit 1
fi

echo "Resolved:"
echo "  LINGBOT_VA_ROOT=$LINGBOT_VA_ROOT"
echo "  ROBOWIN_ROOT=$ROBOWIN_ROOT"
echo "  WAN22_MODEL_PATH=$WAN22_MODEL_PATH"
echo "  RESULTS_DIR=$RESULTS_DIR"
echo "  TASK_NAME=$TASK_NAME"
echo "  TEST_NUM=$TEST_NUM"
echo "  PORT=$PORT"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  SERVER_NPROC=$SERVER_NPROC"
echo "  SERVER_CUDA_VISIBLE_DEVICES=$SERVER_CUDA_VISIBLE_DEVICES"
echo "  CLIENT_CUDA_VISIBLE_DEVICES=$CLIENT_CUDA_VISIBLE_DEVICES"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo
echo "Launching inference server ..."
CUDA_VISIBLE_DEVICES="$SERVER_CUDA_VISIBLE_DEVICES" \
  START_PORT="$PORT" \
  MASTER_PORT="$MASTER_PORT" \
  SERVER_NPROC="$SERVER_NPROC" \
  SAVE_ROOT="$ROOT_DIR/visualization" \
  bash scripts/launch_server.sh &
SERVER_PID=$!

echo "Waiting ${SERVER_STARTUP_SLEEP}s for server startup ..."
sleep "$SERVER_STARTUP_SLEEP"

echo
echo "Launching evaluation client (simulation) ..."
CUDA_VISIBLE_DEVICES="$CLIENT_CUDA_VISIBLE_DEVICES" \
  PORT="$PORT" \
  TASK_NAME="$TASK_NAME" \
  TASK_CONFIG="$TASK_CONFIG" \
  TEST_NUM="$TEST_NUM" \
  bash scripts/launch_client.sh "$RESULTS_DIR" "$TASK_NAME"

echo
echo "Client finished. Stopping server ..."
cleanup

echo "Experiment complete. Results saved in $RESULTS_DIR"
