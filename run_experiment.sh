#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/lingbot-va:$(pwd)/robotwin:$(pwd)/scripts
export ROBOWIN_ROOT=$(pwd)/robotwin

# Check if models are downloaded
if [ ! -d "checkpoints" ]; then
    echo "Warning: checkpoints directory not found. Please download models first."
    # You might want to add a download script here or instructions
fi

# Launch Server (in background)
echo "Launching Inference Server..."
bash scripts/launch_server.sh &
SERVER_PID=$!

# Wait for server to start (adjust sleep time as needed)
sleep 30

# Launch Client
echo "Launching Inference Client..."
# You can specify task name and save root
bash scripts/launch_client.sh results/ "adjust_bottle"

# Wait for client to finish
wait $!

# Kill server
kill $SERVER_PID

echo "Experiment complete. Results saved in results/"
