"""
Framework for standardized hydrological forecasting experiments.

This module provides a structured approach to implementing experiments
for hydrological forecasting models, ensuring consistency across different
experimental setups while maintaining flexibility.
"""

__version__ = "0.1.0"

# Configuration
from .config import BaseExperimentConfig

# Utility functions
from .utils import (
    setup_dirs,
    train_model,
    save_experiment_results,
    setup_seeds
)

# Data utilities
from .data_utils import (
    create_datamodule,
    setup_preprocessing
)

# Model utilities
from .model_utils import (
    create_model,
    load_pretrained_model,
    load_model_configs_from_yaml
)

# Export commonly used functions and classes
__all__ = [
    # Configuration
    'BaseExperimentConfig',
    
    # Utility functions
    'setup_dirs',
    'train_model',
    'save_experiment_results',
    'setup_seeds',
    
    # Data utilities
    'create_datamodule',
    'setup_preprocessing',
    
    # Model utilities
    'create_model',
    'load_pretrained_model',
    'load_model_configs_from_yaml'
]
