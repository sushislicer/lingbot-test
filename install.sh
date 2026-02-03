#!/bin/bash

# Create conda environment
conda create -n lingbot python=3.10.16 -y
source activate lingbot

# Install PyTorch
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install websockets einops diffusers==0.36.0 transformers==5.0.0 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation

# Install LingBot-VA dependencies
pip install -r lingbot-va/requirements.txt

# Install RoboTwin dependencies
if [ -d "RoboTwin" ]; then
    pip install -r RoboTwin/requirements.txt
    pip install -e RoboTwin
elif [ -d "robotwin" ]; then
    pip install -r robotwin/requirements.txt
    pip install -e robotwin
fi

echo "Installation complete. Please activate the environment with 'conda activate lingbot' and run run_experiment.sh."
