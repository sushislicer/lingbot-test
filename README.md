# LingBot-VA × RoboTwin (Glue Repo) — Experiment Reproduction

This repository is a **glue layer** to reproduce LingBot-VA results on the RoboTwin benchmark.

Key constraint: **`lingbot-va` and `RoboTwin` are treated as read-only external repos** on the remote machine.

Example remote machine layout:

```text
/workspace/lingbot-test/        # this repo (editable)
/workspace/lingbot-va/          # upstream LingBot-VA (readonly)
/workspace/RoboTwin/            # upstream RoboTwin (readonly)
```

## Prerequisites

*   Linux environment (tested on Ubuntu).
*   NVIDIA GPUs (tested target: **2× RTX 5090**) with **recent NVIDIA drivers**.
*   Python 3.10.16 (or compatible).
*   Git.
*   Vulkan runtime available (`vulkaninfo` works).
*   `ffmpeg` installed (`ffmpeg -version` works).

### RTX 5090 (Blackwell) compatibility notes

RTX 5090-class GPUs generally require **newer driver stacks** than older Ampere/Ada machines.
To avoid common CUDA/Vulkan issues:

1) Confirm your driver is new enough:

```bash
nvidia-smi
```

2) Confirm Vulkan is working (RoboTwin uses SAPIEN + Vulkan):

```bash
vulkaninfo >/dev/null
```

3) If using Docker, pass graphics capability (otherwise Vulkan can segfault):

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

4) PyTorch CUDA wheel selection:

This repo defaults to installing PyTorch from `cu126`. If your machine image uses a newer CUDA runtime/driver stack and you prefer matching wheels, set:

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# or another cu12x index-url you know is supported on your image
```

## Directory Structure

```
.
├── scripts/             # Local overlays / wrappers (this repo)
├── checkpoints/         # Model checkpoints (downloaded)
├── results/             # Experiment results
├── setup.sh             # Setup script (prepares scripts)
├── install.sh           # Installation script (installs deps)
├── download_models.sh   # Model download script
├── run_experiment.sh    # Experiment runner script
└── README.md            # This file
```

`lingbot-va/` and `RoboTwin/` do **not** need to exist inside this repo. If you want convenience symlinks,
[`setup.sh`](setup.sh) will create them if missing.

## Environment variables

These scripts are written to work when the upstream repos are elsewhere:

* `LINGBOT_VA_ROOT` (default: `/workspace/lingbot-va`)
* `ROBOWIN_ROOT` (default: `/workspace/RoboTwin`)
* `CHECKPOINTS_DIR` (default: `./checkpoints`)
* `WAN22_MODEL_PATH` (default: `$CHECKPOINTS_DIR/lingbot-va-posttrain-robotwin`)

Optional GPU/port controls:

* `SERVER_CUDA_VISIBLE_DEVICES` (default: `0`)
* `CLIENT_CUDA_VISIBLE_DEVICES` (default: `1`)
* `SERVER_NPROC` (default: `1`) — set to `2` if you want to shard the server across both GPUs
* `PORT` (default: `29056`)
* `MASTER_PORT` (default: `29061`)

## Getting Started

### 1. Setup

Run the setup script to prepare the evaluation scripts.

```bash
bash setup.sh
```

### 2. Installation

Run the installation script to install all dependencies.
**Note:** This script installs packages directly into the current Python environment. It is recommended to run this in a container or virtual environment if you want isolation.

```bash
bash install.sh
```

This script follows the RoboTwin install doc and (when possible) applies the documented hotfixes:

* `mplib/planner.py` — removes the problematic `or collide` branch
* `sapien/wrapper/urdf_loader.py` — ensures utf-8 reads and `.srdf` handling

It also installs CuRobo into `./third_party/curobo` (so we don't need to clone into the upstream RoboTwin repo).

If `flash-attn` fails to build on RTX 5090 images (toolchain / CUDA arch mismatch), it is non-fatal for many workflows here; the script will continue.

### 3. Download Models

Download the required model checkpoints from Hugging Face.

```bash
bash download_models.sh
```

This will download:
*   `robbyant/lingbot-va-base`
*   `robbyant/lingbot-va-posttrain-robotwin`

### 4. Run Experiments

Run the experiment script to launch the inference server and client for the RoboTwin benchmark.

```bash
bash run_experiment.sh
```

Defaults are optimized for **2 GPUs**:

* server on GPU0
* RoboTwin simulation + evaluation client on GPU1

Override example:

```bash
SERVER_CUDA_VISIBLE_DEVICES=0 \
CLIENT_CUDA_VISIBLE_DEVICES=1 \
TASK_NAME=stack_bowls_three \
TEST_NUM=20 \
bash run_experiment.sh
```

This script will:
*   Set necessary environment variables (`PYTHONPATH`, `ROBOWIN_ROOT`, `LINGBOT_VA_ROOT`, etc.).
*   Launch the inference server in the background.
*   Run the evaluation client for the `adjust_bottle` task (default).
*   Save results to the `results/` directory.

To run other tasks, you can modify `run_experiment.sh` or run the client script manually:

```bash
# Example: Run a specific task
export LINGBOT_VA_ROOT=/workspace/lingbot-va
export ROBOWIN_ROOT=/workspace/RoboTwin
export PYTHONPATH=$(pwd)/scripts:$LINGBOT_VA_ROOT:$ROBOWIN_ROOT:$PYTHONPATH
PORT=29056 TASK_NAME=stack_bowls_three bash scripts/launch_client.sh $(pwd)/results/ "stack_bowls_three"
```

## Troubleshooting

### Flash Attention Build Crash
If your SSH connection crashes or the installation hangs while building `flash-attn`, it is likely due to running out of memory (RAM) during compilation. The `install.sh` script sets `MAX_JOBS=1` to mitigate this. If it still fails, try increasing your VM's memory or swap space.
