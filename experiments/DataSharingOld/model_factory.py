"""Factory for creating hydrological forecasting models with fixed hyperparameters."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytorch_lightning as pl
from typing import Any


class ModelFactory:
    """Factory class for creating hydrological forecasting models with fixed hyperparameters.

    This factory instantiates the appropriate model class based on the
    provided model type and configuration. It supports multiple model architectures
    including TiDE, TSMixer, EALSTM, and TFT.
    """

    @staticmethod
    def create_model(config: Any, model_type: str) -> pl.LightningModule:
        """Create a LightningModule model instance based on configuration.

        Args:
            config: Configuration object for the model
            model_type: Type of model to create

        Returns:
            An instance of a PyTorch Lightning model

        Raises:
            ValueError: If the model type is not supported
        """
        print(f"Creating model of type: {model_type}")

        # Create appropriate model based on type
        if model_type.lower() == "tide":
            from src.models.tide import LitTiDE

            return LitTiDE(config)
        elif model_type.lower() == "tsmixer":
            from src.models.tsmixer import LitTSMixer

            return LitTSMixer(config)
        elif model_type.lower() == "ealstm":
            from src.models.ealstm import LitEALSTM

            return LitEALSTM(config)
        elif model_type.lower() == "tft":
            from src.models.tft import LitTFT

            return LitTFT(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
