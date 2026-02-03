#!/bin/bash

# Create checkpoints directory
mkdir -p checkpoints

# Download LingBot-VA Base
echo "Downloading LingBot-VA Base..."
huggingface-cli download robbyant/lingbot-va-base --local-dir checkpoints/lingbot-va-base

# Download LingBot-VA Posttrain Robotwin
echo "Downloading LingBot-VA Posttrain Robotwin..."
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir checkpoints/lingbot-va-posttrain-robotwin

echo "Models downloaded to checkpoints/"
