"""
Based on the paper: "TSMixer: An All-MLP Architecture for Time Series Forecasting"
https://arxiv.org/abs/2303.06053

TSMixer Model Implementation. The architecture is based on Figure 6 from the paper:
"""


from typing import Dict, Optional, Tuple, Any, Union, List
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
import pytorch_lightning as pl
import torch
import torch.nn as nn


class TSMixerConfig:
    """Configuration class for TSMixer model."""

    def __init__(
        self,
        input_len: int,
        input_size: int,
        output_len: int,
        static_size: int,
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_layers: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
    ):
        """Initialize TSMixer configuration."""
        self.input_len = input_len
        self.input_size = input_size
        self.output_len = output_len
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.static_embedding_size = static_embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.group_identifier = group_identifier
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TSMixerConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()

    def update(self, **kwargs) -> "TSMixerConfig":
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self


class FeatureMixingBlock(nn.Module):
    """Feature mixing block that processes each time step across features."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float):
        super().__init__()

        self.mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mixing(x))


class TimeMixingBlock(nn.Module):
    """Time mixing block that processes each feature across time."""

    def __init__(self, input_len: int, hidden_size: int, dropout: float):
        super().__init__()

        self.mixing = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_len),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(input_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose to apply mixing along time dimension
        x_t = x.transpose(1, 2)
        mixed = x_t + self.mixing(x_t)
        # Transpose back to original shape
        return self.norm(mixed).transpose(1, 2)


class ResBlock(nn.Module):
    """Residual block combining temporal and feature mixing."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float, input_len: int):
        super().__init__()

        # Temporal mixing: mixing along the time dimension
        self.temporal = TimeMixingBlock(input_len, hidden_size, dropout)

        # Channel (feature) mixing: mixing along the feature dimension
        self.channel = FeatureMixingBlock(input_dim, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal mixing
        x = self.temporal(x)

        # Channel mixing
        x = self.channel(x)

        return x


class ConditionalFeatureMixing(nn.Module):
    """Applies conditional feature mixing using static features to modulate dynamic features."""

    def __init__(self, input_size: int, static_size: int, static_embedding_size: int, hidden_size: int):
        super().__init__()

        # Projections for static features
        self.static_proj = nn.Linear(static_size, static_embedding_size)

        # Gate projection for modulation
        self.gate_proj = nn.Linear(static_embedding_size, input_size)

        # Feature mixing after modulation
        self.feature_mixing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        self.norm = nn.LayerNorm(input_size)

    def forward(self, x_dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Project static features
        static_emb = self.static_proj(static)

        # Expand static features across time dimension
        static_expanded = static_emb.unsqueeze(
            1).expand(-1, x_dynamic.size(1), -1)

        # Create modulation gate
        gate = torch.sigmoid(self.gate_proj(static_expanded))

        # Apply modulation to dynamic features
        x_conditioned = x_dynamic * gate

        # Apply feature mixing
        mixed = self.feature_mixing(x_conditioned)

        # Apply residual connection and normalization
        return self.norm(x_conditioned + mixed)


class TSMixerBackbone(nn.Module):
    """Enhanced TSMixerBackbone with conditional feature mixing."""

    def __init__(self, config: TSMixerConfig):
        super().__init__()

        # Conditional feature mixing to integrate static features
        self.conditional_mixing = ConditionalFeatureMixing(
            input_size=config.input_size,
            static_size=config.static_size,
            static_embedding_size=config.static_embedding_size,
            hidden_size=config.hidden_size
        )

        # Main mixing layers
        self.layers = nn.ModuleList([
            ResBlock(
                input_dim=config.input_size,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
                input_len=config.input_len,
            ) for _ in range(config.num_layers)
        ])

    def forward(self, x: torch.Tensor, static: torch.Tensor, zero_static: bool = False) -> torch.Tensor:
        """
        Forward pass with conditional feature mixing.

        Args:
            x: Dynamic input features [batch_size, seq_len, input_size]
            static: Static features [batch_size, static_size]
            zero_static: If True, bypass static feature conditioning
        """
        if not zero_static:
            # Apply conditional feature mixing
            x = self.conditional_mixing(x, static)

        # Process through mixing layers
        for layer in self.layers:
            x = layer(x)

        return x


class TSMixerHead(nn.Module):
    """Prediction head for TSMixer."""

    def __init__(self, input_dim: int, input_len: int, hidden_size: int, output_len: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Flatten sequence and feature dimensions
        x = x.reshape(batch_size, -1)
        return self.projection(x).unsqueeze(-1)  # [B, output_len, 1]


class TSMixer(nn.Module):
    """Complete TSMixer model with separate backbone and head components."""

    def __init__(self, config: TSMixerConfig):
        super().__init__()

        self.backbone = TSMixerBackbone(config)
        self.head = TSMixerHead(
            input_dim=config.input_size,
            input_len=config.input_len,
            hidden_size=config.hidden_size,
            output_len=config.output_len
        )
        self.config = config

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, static)
        return self.head(features)


class LitTSMixer(pl.LightningModule):
    """PyTorch Lightning Module implementation of TSMixer."""

    def __init__(
        self,
        config: Union[TSMixerConfig, Dict[str, Any]],
    ):
        """Initialize the Lightning Module with a TSMixerConfig."""
        super().__init__()

        # Handle different config input types
        if isinstance(config, dict):
            self.config = TSMixerConfig.from_dict(config)
        else:
            self.config = config

        # Create the TSMixer model using the config
        self.model = TSMixer(self.config)

        # Save all hyperparameters from config for reproducibility
        self.save_hyperparameters(self.config.to_dict())

        # Set up criteria and tracking variables
        self.mse_criterion = MSELoss()
        self.test_outputs = []

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print("Backbone parameters frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        print("Backbone parameters unfrozen")

    def freeze_head(self):
        """Freeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = False
        print("Head parameters frozen")

    def unfreeze_head(self):
        """Unfreeze prediction head parameters."""
        for param in self.model.head.parameters():
            param.requires_grad = True
        print("Head parameters unfrozen")

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.model(x, static)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        loss = self.mse_criterion(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        loss = self.mse_criterion(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))

        return {
            "val_loss": loss,
            "preds": y_hat,
            "targets": y
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        self.test_outputs.append(output)
        return output

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Consolidate all test outputs
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        self.test_outputs = []

    def configure_optimizers(self) -> Dict:
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.lr_scheduler_patience,
                factor=self.config.lr_scheduler_factor
            ),
            "monitor": "val_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
