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

# Sync dependencies from pyproject.toml
echo "Syncing dependencies from pyproject.toml..."
if [ -f "pyproject.toml" ]; then
    uv sync
else
    echo "Error: pyproject.toml not found!"
    exit 1
fi

# Configure Git user
echo "Configuring Git..."
git config --global user.name "YourUsername"
git config --global user.email "your.email@example.com"

# Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Git configuration and virtual environment setup complete!"
