#!/bin/bash

# Install pip-tools
echo "Installing pip-tools..."
pip install pip-tools

# Create requirements.txt from pyproject.toml
echo "Creating requirements.txt from pyproject.toml..."
pip-compile

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Configure Git user
echo "Configuring Git..."
git config --global user.name "YourUsername"
git config --global user.email "your.email@example.com"

# Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Git configuration and virtual environment setup complete!"
