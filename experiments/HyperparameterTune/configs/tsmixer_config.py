"""TSMixer model configuration for hyperparameter tuning."""

from typing import Dict, Any, ClassVar
from .base_config import BaseHyperparamConfig
from src.models.TSMixer import TSMixerConfig


class TSMixerTuneConfig(BaseHyperparamConfig):
    """Configuration for hyperparameter tuning of TSMixer models."""

    MODEL_TYPE: str = "tsmixer"
    
    # TSMixer specific parameters
    STATIC_EMBEDDING_SIZE: int = 10
    NUM_MIXING_LAYERS: int = 5
    FUSION_METHOD: str = "add"  # Options: "add" or "concat"
    
    # Define hyperparameter search space
    HYPERPARAMETER_SPACE: ClassVar[Dict[str, Dict[str, Any]]] = {
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
        }
    }
    
    def get_model_config(self) -> TSMixerConfig:
        """Create a TSMixerConfig from the current configuration."""
        # Calculate future_input_size as the number of forcing features
        future_input_size = len(self.FORCING_FEATURES)
        
        return TSMixerConfig(
            input_len=self.INPUT_LENGTH,
            output_len=self.OUTPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            future_input_size=future_input_size,
            hidden_size=self.HIDDEN_SIZE,
            static_embedding_size=self.STATIC_EMBEDDING_SIZE,
            num_mixing_layers=self.NUM_MIXING_LAYERS,
            dropout=self.DROPOUT,
            learning_rate=self.LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            scheduler_factor=self.LR_SCHEDULER_FACTOR,
            fusion_method=self.FUSION_METHOD,
        )
