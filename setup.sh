#!/bin/bash

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    git init
fi

# Add submodules
# Note: If these fail, please check the URLs or your internet connection.
# You might need to manually clone them if submodule add fails.

# LingBot-VA (already present, but adding as submodule if possible)
# Since it's already cloned, we might need to convert it to a submodule or just leave it.
# Assuming it's already there as a directory.

# RoboTwin
if [ -d "RoboTwin" ]; then
    echo "RoboTwin directory found."
elif [ -d "robotwin" ]; then
    echo "robotwin directory found."
else
    echo "RoboTwin directory not found. Please clone it into the project root."
fi

# Update submodules (if any)
# git submodule update --init --recursive

# Create scripts directory and copy evaluation scripts
mkdir -p scripts
cp lingbot-va/evaluation/robotwin/*.py scripts/
cp lingbot-va/evaluation/robotwin/launch_client.sh scripts/launch_client.sh
cp lingbot-va/evaluation/robotwin/launch_server.sh scripts/launch_server.sh

# Modify scripts to use environment variables and correct paths
# Modify eval_polict_client_openpi.py
sed -i 's|robowin_root = Path("/path/to/your/robowin")|robowin_root = Path(os.environ.get("ROBOWIN_ROOT", "/path/to/your/robowin"))|' scripts/eval_polict_client_openpi.py

# Modify launch_client.sh
sed -i 's|python evaluation/robotwin/eval_polict_client_openpi.py|python scripts/eval_polict_client_openpi.py|' scripts/launch_client.sh

# Modify launch_server.sh
sed -i 's|wan_va/wan_va_server.py|lingbot-va/wan_va/wan_va_server.py|' scripts/launch_server.sh

# Make scripts executable
chmod +x scripts/launch_client.sh scripts/launch_server.sh

echo "Setup complete. Please run install.sh to install dependencies."
