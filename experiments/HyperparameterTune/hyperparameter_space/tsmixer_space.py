"""Hyperparameter search space definition for TSMixer models."""

from typing import Dict, Any


def get_tsmixer_space() -> Dict[str, Dict[str, Any]]:
    """
    Define the hyperparameter search space for TSMixer models.
    
    Returns:
        Dictionary containing common and model-specific hyperparameter ranges
    """
    return {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "hidden_size": {"type": "int", "low": 32, "high": 128},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "num_mixing_layers": {"type": "int", "low": 2, "high": 15},
            "static_embedding_size": {"type": "int", "low": 5, "high": 20},
            "fusion_method": {"type": "categorical", "choices": ["add", "concat"]},
        },
    }
