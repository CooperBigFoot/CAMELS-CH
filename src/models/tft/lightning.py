from typing import Dict, Any, Optional, Union
import torch
from ..base.base_lit_model import BaseLitModel
from .config import TFTConfig
from .model import TemporalFusionTransformer


class LitTFT(BaseLitModel):
    """PyTorch Lightning Module implementation of TFT.

    This class extends BaseLitModel to provide a standardized interface for training,
    validation, and testing of the Temporal Fusion Transformer within our
    hydrological forecasting framework.
    """

    def __init__(
        self,
        config: Union[TFTConfig, Dict[str, Any]],
    ) -> None:
        """
        Initialize the LitTFT module.

        Args:
            config: TFT configuration as a TFTConfig instance or dict
        """
        # Convert dict config to TFTConfig if needed
        if isinstance(config, dict):
            config = TFTConfig.from_dict(config)

        # Initialize base lightning model with the config
        super().__init__(config)

        # Create the TFT model
        self.model = TemporalFusionTransformer(config)

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TFT model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer with appropriate parameters.

        Returns:
            Dictionary with optimizer and learning rate scheduler configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        # Create scheduler dictionary
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
