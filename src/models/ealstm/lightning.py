"""PyTorch Lightning module for EA-LSTM model."""

from typing import Dict, Any, Optional, Union
import torch
from ..base.base_lit_model import BaseLitModel
from .config import EALSTMConfig
from .model import EALSTM


class LitEALSTM(BaseLitModel):
    """PyTorch Lightning Module implementation of EA-LSTM.
    
    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the EA-LSTM model within our hydrological forecasting framework.
    """

    def __init__(
        self,
        config: Union[EALSTMConfig, Dict[str, Any]],
    ) -> None:
        """
        Initialize the LitEALSTM module.

        Args:
            config: EA-LSTM configuration as an EALSTMConfig instance or dict
        """
        # Convert dict config to EALSTMConfig if needed
        if isinstance(config, dict):
            ealstm_config = EALSTMConfig.from_dict(config)
        else:
            ealstm_config = config

        # Initialize base lightning model with the config
        super().__init__(ealstm_config)
        
        # Create the EA-LSTM model
        self.model = EALSTM(ealstm_config)

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the EA-LSTM model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)
