#!/bin/bash

# Check if uv is installed, install if missing
if ! command -v uv &> /dev/null
then
    echo "Installing uv..."
    pip install uv
fi

# Create a virtual environment
echo "Creating virtual environment..."
uv venv

# Sync dependencies
echo "Syncing dependencies from pyproject.toml..."
uv pip sync

# Configure Git user
echo "Configuring Git..."
git config --global user.name "CooperBigFoot"
git config --global user.email "nlazaro@student.ethz.ch"

# Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Git configuration and virtual environment setup complete!"
