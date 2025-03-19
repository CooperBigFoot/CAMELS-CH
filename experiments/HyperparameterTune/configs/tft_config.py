"""TFT model configuration for hyperparameter tuning."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, ClassVar, List
from .base_config import BaseHyperparamConfig
from src.models.tft import TFTConfig


class TFTTuneConfig(BaseHyperparamConfig):
    """Configuration for hyperparameter tuning of Temporal Fusion Transformer models.
    
    This configuration class defines the search space and parameters specific
    to the TFT architecture, which combines RNNs and attention mechanisms for
    time series forecasting with interpretability features.
    """

    MODEL_TYPE: str = "tft"
    
    # TFT specific parameters
    HIDDEN_SIZE: int = 64
    NUM_ATTENTION_HEADS: int = 4
    DROPOUT: float = 0.1
    LSTM_LAYERS: int = 2
    VARIABLE_SELECTION_METHOD: str = "gating"  # Options: "gating" or "dot_product"
    
    # Define hyperparameter search space
    HYPERPARAMETER_SPACE: ClassVar[Dict[str, Dict[str, Any]]] = {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "hidden_size": {"type": "int", "low": 32, "high": 128},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "num_attention_heads": {"type": "int", "low": 1, "high": 8},
            "lstm_layers": {"type": "int", "low": 1, "high": 3},
            "variable_selection_method": {"type": "categorical", "choices": ["gating", "dot_product"]},
        }
    }
    
    # Computed properties that are needed for TFT configuration
    @property
    def static_covariates(self) -> List[str]:
        """Get list of static covariates (all static features except gauge_id)."""
        return [f for f in self.STATIC_FEATURES if f != self.GROUP_IDENTIFIER]
    
    @property
    def time_varying_known_covariates(self) -> List[str]:
        """Get list of time-varying known covariates (forcing features)."""
        return self.FORCING_FEATURES
    
    @property
    def time_varying_unknown_covariates(self) -> List[str]:
        """Get list of time-varying unknown covariates (target variable)."""
        return [self.TARGET]
    
    def get_model_config(self) -> TFTConfig:
        """Create a TFTConfig from the current configuration.
        
        Returns:
            Configuration object for TFT model
        """
        return TFTConfig(
            input_len=self.INPUT_LENGTH,
            output_len=self.OUTPUT_LENGTH,
            static_covariates=self.static_covariates,
            time_varying_known_covariates=self.time_varying_known_covariates,
            time_varying_unknown_covariates=self.time_varying_unknown_covariates,
            hidden_size=self.HIDDEN_SIZE,
            lstm_layers=self.LSTM_LAYERS,
            num_attention_heads=self.NUM_ATTENTION_HEADS,
            dropout=self.DROPOUT,
            learning_rate=self.LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            variable_selection_method=self.VARIABLE_SELECTION_METHOD,
            scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            scheduler_factor=self.LR_SCHEDULER_FACTOR,
        )
