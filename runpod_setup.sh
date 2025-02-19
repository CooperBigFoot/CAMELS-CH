#!/bin/bash

echo "Installing all packages"
pip install "numpy<=3.0.0" "torch>=2.0.0,!=2.0.1,<3.0.0" "lightning>=2.0.0,<3.0.0" "scipy>=1.8,<2.0" "pandas>=1.3.0,<3.0.0" "scikit-learn>=1.2,<2.0" "ipykernel>=6.29.5" "matplotlib>=3.10.0" "seaborn>=0.13.2" "pip-tools>=7.4.1" "cdsapi>=0.7.4" "geopandas>=1.0.1" "optuna>=4.2.1"

# Configure Git user
echo "Configuring Git..."
git config --global user.name "CooperBigFoot"
git config --global user.email "nlazaro@student.ethz.ch"

# Print success message
echo "RunPod setup complete. You can now train and push the code to GitHub"
