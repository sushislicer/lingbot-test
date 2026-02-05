#!/usr/bin/env bash
set -euo pipefail

# Installs Python dependencies for running LingBot-VA evaluation on RoboTwin.
#
# IMPORTANT: This repo is intended to be a lightweight "glue" layer located at
#   /workspace/lingbot-test
# while upstream repos live elsewhere and may be mounted read-only:
#   /workspace/lingbot-va
#   /workspace/RoboTwin
#
# This script therefore MUST NOT edit files inside those upstream repos.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_LINGBOT_VA_ROOT="/workspace/lingbot-va"
DEFAULT_ROBOTWIN_SRC_ROOT="/workspace/RoboTwin"

LINGBOT_VA_ROOT="${LINGBOT_VA_ROOT:-$DEFAULT_LINGBOT_VA_ROOT}"
ROBOTWIN_SRC_ROOT="${ROBOTWIN_SRC_ROOT:-$DEFAULT_ROBOTWIN_SRC_ROOT}"

if [[ ! -d "$LINGBOT_VA_ROOT" ]]; then
  echo "Error: LINGBOT_VA_ROOT not found: $LINGBOT_VA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ROBOTWIN_SRC_ROOT" ]]; then
  echo "Error: ROBOTWIN_SRC_ROOT not found: $ROBOTWIN_SRC_ROOT" >&2
  exit 1
fi

echo "Using:"
echo "  LINGBOT_VA_ROOT=$LINGBOT_VA_ROOT"
echo "  ROBOTWIN_SRC_ROOT=$ROBOTWIN_SRC_ROOT"

echo
echo "[System deps notes]"
echo "- RoboTwin requires Vulkan runtime + tools (check: vulkaninfo)."
echo "  Ubuntu example: sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools"
echo "- ffmpeg is required for video logging (check: ffmpeg -version)."
echo

###############################################################################
# 1) Core Python deps + PyTorch
###############################################################################

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

echo "Installing PyTorch (index-url: $TORCH_INDEX_URL) ..."
pip install \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url "$TORCH_INDEX_URL"

echo "Installing common runtime deps ..."
pip install \
  websockets \
  einops \
  diffusers==0.36.0 \
  transformers \
  accelerate \
  msgpack \
  opencv-python \
  matplotlib \
  ftfy \
  easydict \
  huggingface_hub \
  imageio[ffmpeg]

# Ensure modern HF CLI is available (`hf download ...`).
pip install -U 'huggingface_hub[cli]'

# lerobot is imported by the eval client for writing metrics.
pip install lerobot || true

echo "Installing flash-attn (may compile; set MAX_JOBS=1 to reduce RAM) ..."
MAX_JOBS=1 pip install flash-attn --no-build-isolation || true

###############################################################################
# 2) LingBot-VA deps (read-only safe)
###############################################################################

echo "Installing LingBot-VA requirements (read-only safe) ..."

_tmp_req_file="$(mktemp)"
trap 'rm -f "${_tmp_req_file}"' EXIT

# Do NOT mutate the upstream requirements.txt; filter out flash_attn because we
# install it separately above (and build isolation frequently breaks remote VMs).
grep -v -E '^flash_attn\s*$' "$LINGBOT_VA_ROOT/requirements.txt" > "${_tmp_req_file}"
pip install -r "${_tmp_req_file}"

###############################################################################
# 3) RoboTwin deps (read-only safe)
###############################################################################

echo "Installing RoboTwin sim/runtime requirements (read-only safe) ..."

if [[ -f "$ROBOTWIN_SRC_ROOT/script/requirements.txt" ]]; then
  _tmp_rt_req_file="$(mktemp)"
  # RoboTwin pins torch; we already installed a newer torch for LingBot-VA.
  # Filter out torch/torchvision to avoid forced downgrades.
  grep -v -E '^torch(vision|audio)?==|^torch(vision|audio)?\b' "$ROBOTWIN_SRC_ROOT/script/requirements.txt" > "${_tmp_rt_req_file}"
  # Best-effort install of RoboTwin's full requirement set (minus torch pins).
  # Some optional packages can fail to build on certain images; we validate
  # critical ones (e.g. sapien) below.
  pip install -r "${_tmp_rt_req_file}" || true
  rm -f "${_tmp_rt_req_file}" || true
else
  echo "Warning: RoboTwin requirements not found at $ROBOTWIN_SRC_ROOT/script/requirements.txt" >&2
fi

###############################################################################
# 3b) Ensure critical RoboTwin simulation deps exist (fail fast)
###############################################################################

echo "Ensuring critical RoboTwin simulation dependencies (sapien/mplib/gymnasium/scipy) ..."

# These are the core packages required for running RoboTwin simulation/eval.
# They are pinned in RoboTwin's [`script/requirements.txt`](RoboTwin/script/requirements.txt:1).
python3 -m pip install \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "gymnasium==0.29.1" \
  "transforms3d==0.4.2" \
  "scipy==1.10.1"

python3 - <<'PY'
import importlib

missing = []
for name in ("sapien", "mplib", "gymnasium", "scipy"):
    try:
        importlib.import_module(name)
    except Exception as e:
        missing.append((name, str(e)))

if missing:
    print("\nERROR: Missing critical RoboTwin deps:")
    for name, err in missing:
        print(f"  - {name}: {err}")
    print("\nCommon causes:")
    print("  - Using a Python version other than 3.10")
    print("  - Missing system Vulkan libs / driver stack")
    print("  - pip wheel not available for your platform")
    raise SystemExit(1)

print("Critical RoboTwin deps import OK.")
PY

###############################################################################
# 4) Apply upstream-documented hotfixes in site-packages
###############################################################################

echo "Applying RoboTwin-documented hotfixes (site-packages patches) ..."

MPLIB_LOCATION="$(pip show mplib 2>/dev/null | awk -F': ' '/^Location:/{print $2}')"
if [[ -n "${MPLIB_LOCATION}" && -f "${MPLIB_LOCATION}/mplib/planner.py" ]]; then
  PLANNER_FILE="${MPLIB_LOCATION}/mplib/planner.py"
  # Remove `or collide` (see RoboTwin install doc)
  sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER_FILE" || true
  echo "  Patched: $PLANNER_FILE"
else
  echo "  Skipped mplib patch (mplib not installed or planner.py not found)"
fi

SAPIEN_LOCATION="$(pip show sapien 2>/dev/null | awk -F': ' '/^Location:/{print $2}')"
if [[ -n "${SAPIEN_LOCATION}" && -f "${SAPIEN_LOCATION}/sapien/wrapper/urdf_loader.py" ]]; then
  URDF_LOADER_FILE="${SAPIEN_LOCATION}/sapien/wrapper/urdf_loader.py"
  # Ensure UTF-8 reads and .srdf extension (see RoboTwin/script/_install.sh)
  sed -i -E 's/with open\(([^,]+), "r"\) as f:/with open(\1, "r", encoding="utf-8") as f:/g' "$URDF_LOADER_FILE" || true
  sed -i -E 's/srdf_file = urdf_file\[:-4\] \+ "srdf"/srdf_file = urdf_file[:-4] + ".srdf"/g' "$URDF_LOADER_FILE" || true
  echo "  Patched: $URDF_LOADER_FILE"
else
  echo "  Skipped sapien patch (sapien not installed or urdf_loader.py not found)"
fi

###############################################################################
# 5) Install CuRobo (required by RoboTwin; cannot clone into readonly upstream)
###############################################################################

CUROBO_DIR="${CUROBO_DIR:-$ROOT_DIR/third_party/curobo}"
mkdir -p "$(dirname "$CUROBO_DIR")"

if [[ ! -d "$CUROBO_DIR/.git" ]]; then
  echo "Cloning CuRobo into: $CUROBO_DIR"
  git clone https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
else
  echo "CuRobo repo exists: $CUROBO_DIR"
fi

echo "Installing CuRobo (editable) ..."
pip install -e "$CUROBO_DIR" --no-build-isolation || true

echo
echo "Installation complete. Next:"
echo "  bash setup.sh"
echo "  bash download_models.sh"
echo "  bash run_experiment.sh"
