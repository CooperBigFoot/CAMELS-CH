"""EALSTM model configuration for hyperparameter tuning."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from typing import Dict, Any, ClassVar
from .base_config import BaseHyperparamConfig
from src.models.ealstm import EALSTMConfig


class EALSTMTuneConfig(BaseHyperparamConfig):
    """Configuration for hyperparameter tuning of Entity-Aware LSTM models.

    This configuration class defines the search space and parameters specific
    to the EALSTM architecture, which is designed to effectively incorporate
    static catchment attributes alongside dynamic inputs.
    """

    MODEL_TYPE: ClassVar["str"] = "ealstm"

    # EALSTM specific parameters
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    STATIC_EMBEDDING_SIZE: int = 10
    BIDIRECTIONAL: bool = True
    FUTURE_HIDDEN_SIZE: int = 64
    FUTURE_LAYERS: int = 2
    BIDIRECTIONAL_FUSION: str = "concat"  

    # Define hyperparameter search space
    HYPERPARAMETER_SPACE: ClassVar[Dict[str, Dict[str, Any]]] = {
        "common": {
            "input_length": {"type": "int", "low": 30, "high": 365},
            "hidden_size": {"type": "int", "low": 32, "high": 256},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        },
        "model_specific": {
            "num_layers": {"type": "int", "low": 1, "high": 3},
        },
    }
    def get_model_config(self) -> EALSTMConfig:
        """Create an EALSTMConfig from the current configuration.

        Returns:
            Configuration object for EALSTM model
        """
        # Calculate future_input_size as the number of forcing features
        future_input_size = len(self.FORCING_FEATURES)

        return EALSTMConfig(
            input_len=self.INPUT_LENGTH,
            output_len=self.OUTPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            future_input_size=future_input_size,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            bidirectional=self.BIDIRECTIONAL,
            learning_rate=self.LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            scheduler_factor=self.LR_SCHEDULER_FACTOR,
            bidirectional_fusion=self.BIDIRECTIONAL_FUSION,
        )
