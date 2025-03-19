"""Factory for creating hydrological forecasting models."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Union, Dict, Any
import torch.nn as nn
import pytorch_lightning as pl
from .configs.base_config import BaseHyperparamConfig


class ModelFactory:
    """Factory class for creating hydrological forecasting models.
    
    This factory instantiates the appropriate model class based on the
    provided configuration type. It supports multiple model architectures
    including TiDE, TSMixer, EALSTM, and more.
    """
    
    @staticmethod
    def create_model(config: BaseHyperparamConfig) -> pl.LightningModule:
        """Create a LightningModule model instance based on configuration.
        
        Args:
            config: Configuration object for the model
            
        Returns:
            An instance of a PyTorch Lightning model
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = config.MODEL_TYPE.lower()
        
        # Generate model configuration
        model_config = config.get_model_config()
        
        # Create appropriate model based on type
        if model_type == "tide":
            from ..src.models.tide import LitTiDE
            return LitTiDE(model_config)
        elif model_type == "tsmixer":
            from ..src.models.tsmixer import LitTSMixer
            return LitTSMixer(model_config)
        elif model_type == "ealstm":
            # Will need implementation of EALSTM model
            from ..src.models.ealstm import LitEALSTM
            return LitEALSTM(model_config)
        elif model_type == "tft":
            # Will need implementation of TFT model
            from ..src.models.tft import LitTFT
            return LitTFT(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
