# LingBot-VA Experiment Reproduction

This repository contains the setup and scripts to reproduce the results of LingBot-VA on the RoboTwin benchmark. It is designed to be run on a remote VM with 4 GPUs.

## Prerequisites

*   Linux environment (tested on Ubuntu).
*   NVIDIA GPUs (4 recommended) with CUDA 12.6 drivers.
*   Conda (Miniconda or Anaconda).
*   Git.
*   `RoboTwin` repository cloned into the project root.

## Directory Structure

```
.
├── lingbot-va/          # LingBot-VA source code (submodule)
├── RoboTwin/            # RoboTwin benchmark (submodule)
├── scripts/             # Modified evaluation scripts
├── checkpoints/         # Model checkpoints (downloaded)
├── results/             # Experiment results
├── setup.sh             # Setup script (prepares scripts)
├── install.sh           # Installation script (creates conda env, installs deps)
├── download_models.sh   # Model download script
├── run_experiment.sh    # Experiment runner script
└── README.md            # This file
```

## Getting Started

### 1. Setup

Ensure `RoboTwin` is cloned into the project root.
Run the setup script to prepare the evaluation scripts.

```bash
bash setup.sh
```

### 2. Installation

Run the installation script to create a Conda environment named `lingbot` and install all dependencies.

```bash
bash install.sh
```

### 3. Activate Environment

Activate the created environment:

```bash
conda activate lingbot
```

### 4. Download Models

Download the required model checkpoints from Hugging Face.

```bash
bash download_models.sh
```

This will download:
*   `robbyant/lingbot-va-base`
*   `robbyant/lingbot-va-posttrain-robotwin`

### 5. Run Experiments

Run the experiment script to launch the inference server and client for the RoboTwin benchmark.

```bash
bash run_experiment.sh
```

This script will:
*   Set necessary environment variables (`PYTHONPATH`, `ROBOWIN_ROOT`, etc.).
*   Launch the inference server in the background.
*   Run the evaluation client for the `adjust_bottle` task (default).
*   Save results to the `results/` directory.

To run other tasks, you can modify `run_experiment.sh` or run the client script manually:

```bash
# Example: Run a specific task
export PYTHONPATH=$PYTHONPATH:$(pwd)/lingbot-va:$(pwd)/RoboTwin:$(pwd)/scripts
export ROBOWIN_ROOT=$(pwd)/RoboTwin
bash scripts/launch_client.sh results/ "stack_bowls_three"
```
