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
git config --global user.name "CooperBigFoot"
git config --global user.email "nlazaro@student.ethz.ch"

# Print success message
echo "RunPod setup complete. You can now train and push the code to GitHub"
