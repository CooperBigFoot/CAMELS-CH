"""TiDE model configuration for hyperparameter tuning."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, ClassVar
from .base_config import BaseHyperparamConfig
from src.models.tide import TiDEConfig


class TiDETuneConfig(BaseHyperparamConfig):
    """Configuration for hyperparameter tuning of TiDE models."""

    MODEL_TYPE: str = "tide"
    
    # TiDE specific parameters
    NUM_ENCODER_LAYERS: int = 1
    NUM_DECODER_LAYERS: int = 1
    DECODER_OUTPUT_SIZE: int = 16
    TEMPORAL_DECODER_HIDDEN_SIZE: int = 32
    PAST_FEATURE_PROJECTION_SIZE: int = 0
    FUTURE_FORCING_PROJECTION_SIZE: int = 0
    USE_LAYER_NORM: bool = True
    
    # Define hyperparameter search space
    HYPERPARAMETER_SPACE: ClassVar[Dict[str, Dict[str, Any]]] = {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "hidden_size": {"type": "int", "low": 32, "high": 128},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "num_encoder_layers": {"type": "int", "low": 1, "high": 3},
            "num_decoder_layers": {"type": "int", "low": 1, "high": 3},
            "decoder_output_size": {"type": "int", "low": 8, "high": 32},
            "temporal_decoder_hidden_size": {"type": "int", "low": 16, "high": 64},
            "use_layer_norm": {"type": "categorical", "choices": [True, False]},
        }
    }
    
    def get_model_config(self) -> TiDEConfig:
        """Create a TiDEConfig from the current configuration."""
        # Calculate future_input_size as the number of forcing features
        future_input_size = len(self.FORCING_FEATURES)
        
        return TiDEConfig(
            input_len=self.INPUT_LENGTH,
            output_len=self.OUTPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            future_input_size=future_input_size,
            hidden_size=self.HIDDEN_SIZE,
            dropout=self.DROPOUT,
            learning_rate=self.LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            num_encoder_layers=self.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.NUM_DECODER_LAYERS,
            decoder_output_size=self.DECODER_OUTPUT_SIZE,
            temporal_decoder_hidden_size=self.TEMPORAL_DECODER_HIDDEN_SIZE,
            past_feature_projection_size=self.PAST_FEATURE_PROJECTION_SIZE,
            future_forcing_projection_size=self.FUTURE_FORCING_PROJECTION_SIZE,
            use_layer_norm=self.USE_LAYER_NORM,
            scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            scheduler_factor=self.LR_SCHEDULER_FACTOR,
        )
