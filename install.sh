#!/bin/bash

# Install PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install websockets einops diffusers==0.36.0 transformers==5.0.0 accelerate msgpack opencv-python matplotlib ftfy easydict
# Limit MAX_JOBS to 1 to prevent OOM crashes during flash-attn compilation
MAX_JOBS=1 pip install flash-attn --no-build-isolation

# Install LingBot-VA dependencies
# Remove flash_attn from requirements.txt to avoid build isolation issues (installed manually above)
sed -i '/flash_attn/d' lingbot-va/requirements.txt
pip install -r lingbot-va/requirements.txt

# Install RoboTwin dependencies
if [ -d "RoboTwin" ]; then
    if [ -f "RoboTwin/requirements.txt" ]; then
        pip install -r RoboTwin/requirements.txt
    fi
    if [ -f "RoboTwin/setup.py" ] || [ -f "RoboTwin/pyproject.toml" ]; then
        pip install -e RoboTwin
    else
        echo "RoboTwin is not a pip package (no setup.py/pyproject.toml). Relying on PYTHONPATH."
    fi
elif [ -d "robotwin" ]; then
    if [ -f "robotwin/requirements.txt" ]; then
        pip install -r robotwin/requirements.txt
    fi
    if [ -f "robotwin/setup.py" ] || [ -f "robotwin/pyproject.toml" ]; then
        pip install -e robotwin
    else
        echo "robotwin is not a pip package. Relying on PYTHONPATH."
    fi
fi

echo "Installation complete. Please run run_experiment.sh."
