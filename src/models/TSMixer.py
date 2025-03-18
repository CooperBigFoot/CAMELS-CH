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
        future_input_size: Optional[int] = None,  
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_mixing_layers: int = 5,  
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 2,  
        scheduler_factor: float = 0.5,  
        fusion_method: str = "add",  
    ):
        """Initialize TSMixer configuration.

        Args:
            input_len: Length of the input sequence
            input_size: Number of input features
            output_len: Length of the output sequence (forecast horizon)
            static_size: Number of static features
            future_input_size: Number of future forcing features (defaults to input_size minus 1)
            hidden_size: Size of hidden layers
            static_embedding_size: Size of static feature embedding
            num_mixing_layers: Number of mixing layers
            dropout: Dropout rate
            learning_rate: Initial learning rate
            group_identifier: Name of the column identifying catchment groups
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            fusion_method: Method to fuse historical and future representations ("add" or "concat")
        """
        self.input_len = input_len
        self.input_size = input_size
        self.output_len = output_len
        self.static_size = static_size
        self.future_input_size = (
            future_input_size
            if future_input_size is not None
            else max(1, input_size - 1)
        )
        self.hidden_size = hidden_size
        self.static_embedding_size = static_embedding_size
        self.num_mixing_layers = num_mixing_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.group_identifier = group_identifier
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.fusion_method = fusion_method

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


class InputAlignmentModule(nn.Module):
    """Aligns historical data, future forcing, and static features into a common representation.

    This module implements the "align" stage described in Section 4.2 of the TSMixer paper,
    projecting heterogeneous inputs into a unified space for subsequent mixing layers.
    """

    def __init__(
        self,
        input_size: int,
        input_len: int,
        output_len: int,
        future_input_size: int,
        hidden_size: int,
        static_size: int,
        static_embedding_size: int,
        dropout: float = 0.1,  # Added parameter for consistent dropout
        fusion_method: str = "add",
    ):
        """Initialize the alignment module.

        Args:
            input_size: Number of input features
            input_len: Length of input sequence
            output_len: Length of output sequence (forecast horizon)
            future_input_size: Number of future forcing features
            hidden_size: Size of hidden representation
            static_size: Number of static features
            static_embedding_size: Size of static feature embedding
            dropout: Dropout rate for regularization
            fusion_method: Method to fuse representations ("add" or "concat")
        """
        super().__init__()

        self.fusion_method = fusion_method
        self.output_len = output_len

        # Historical data projection
        self.historical_projection = nn.Sequential(
            nn.Linear(input_len * input_size, hidden_size * output_len),
            nn.ReLU(),
            nn.Dropout(dropout),  # Using parameterized dropout
        )

        # Future forcing projection
        self.future_projection = nn.Sequential(
            nn.Linear(future_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),  # Using parameterized dropout
        )

        # Static feature projection (if available)
        if static_size > 0:
            self.static_projection = nn.Sequential(
                nn.Linear(static_size, static_embedding_size),
                nn.ReLU(),
                nn.Dropout(dropout),  # Using parameterized dropout
            )

            # Gate for static modulation
            self.static_gate = nn.Linear(static_embedding_size, hidden_size)
        else:
            self.static_projection = None
            self.static_gate = None

        # Final output size after fusion
        if fusion_method == "add":
            self.output_size = hidden_size
        elif fusion_method == "concat":
            self.output_size = hidden_size * 2
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # Layer normalization for aligned representations
        self.norm_historical = nn.LayerNorm(hidden_size)
        self.norm_future = nn.LayerNorm(hidden_size)
        self.norm_fused = nn.LayerNorm(self.output_size)

    def forward(
        self,
        historical: torch.Tensor,
        future: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Align and fuse historical, future, and static data.

        Args:
            historical: Historical data [batch_size, input_len, input_size]
            future: Future forcing data [batch_size, output_len, future_input_size]
            static: Static features [batch_size, static_size]

        Returns:
            Fused representation [batch_size, output_len, output_size]
        """
        batch_size = historical.size(0)

        # Project historical data to match output sequence length
        hist_flat = historical.reshape(batch_size, -1)  # [B, input_len * input_size]
        hist_proj = self.historical_projection(
            hist_flat
        )  # [B, hidden_size * output_len]
        hist_aligned = hist_proj.reshape(
            batch_size, self.output_len, -1
        )  # [B, output_len, hidden_size]
        hist_aligned = self.norm_historical(hist_aligned)

        # Project future forcing data
        future_aligned = self.norm_future(
            self.future_projection(future)
        )  # [B, output_len, hidden_size]

        # Apply static modulation if available
        if static is not None and self.static_projection is not None:
            static_emb = self.static_projection(static)  # [B, static_embedding_size]
            static_gate = torch.sigmoid(
                self.static_gate(static_emb)
            )  # [B, hidden_size]
            static_gate = static_gate.unsqueeze(1).expand(
                -1, self.output_len, -1
            )  # [B, output_len, hidden_size]

            # Apply modulation
            hist_aligned = hist_aligned * static_gate
            future_aligned = future_aligned * static_gate

        # Fuse representations
        if self.fusion_method == "add":
            fused = hist_aligned + future_aligned
        else:  # concat
            fused = torch.cat([hist_aligned, future_aligned], dim=-1)

        return self.norm_fused(fused)


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

    def __init__(
        self, input_dim: int, hidden_size: int, dropout: float, input_len: int
    ):
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

    def __init__(
        self,
        input_size: int,
        static_size: int,
        static_embedding_size: int,
        hidden_size: int,
    ):
        super().__init__()

        # Projections for static features
        self.static_proj = nn.Linear(static_size, static_embedding_size)

        # Gate projection for modulation
        self.gate_proj = nn.Linear(static_embedding_size, input_size)

        # Feature mixing after modulation
        self.feature_mixing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

        self.norm = nn.LayerNorm(input_size)

    def forward(self, x_dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Project static features
        static_emb = self.static_proj(static)

        # Expand static features across time dimension
        static_expanded = static_emb.unsqueeze(1).expand(-1, x_dynamic.size(1), -1)

        # Create modulation gate
        gate = torch.sigmoid(self.gate_proj(static_expanded))

        # Apply modulation to dynamic features
        x_conditioned = x_dynamic * gate

        # Apply feature mixing
        mixed = self.feature_mixing(x_conditioned)

        # Apply residual connection and normalization
        return self.norm(x_conditioned + mixed)


class TSMixerBackbone(nn.Module):
    """Enhanced TSMixerBackbone with input alignment and future forcing integration."""

    def __init__(self, config: TSMixerConfig):
        super().__init__()

        # Input alignment module to integrate historical, future, and static features
        self.alignment_module = InputAlignmentModule(
            input_size=config.input_size,
            input_len=config.input_len,
            output_len=config.output_len,
            future_input_size=config.future_input_size,
            hidden_size=config.hidden_size,
            static_size=config.static_size,
            static_embedding_size=config.static_embedding_size,
            dropout=config.dropout,  # Pass config dropout for consistency
            fusion_method=config.fusion_method,
        )

        # Determine the input dimension for mixing layers based on fusion method
        input_dim = (
            config.hidden_size * 2
            if config.fusion_method == "concat"
            else config.hidden_size
        )

        # Main mixing layers
        self.layers = nn.ModuleList(
            [
                ResBlock(
                    input_dim=input_dim,
                    hidden_size=config.hidden_size,
                    dropout=config.dropout,
                    input_len=config.output_len,  # Now using output_len as the sequence length
                )
                for _ in range(config.num_mixing_layers)
            ]
        )

        # Optional conditional feature mixing (kept for backward compatibility)
        self.conditional_mixing = ConditionalFeatureMixing(
            input_size=config.input_size,
            static_size=config.static_size,
            static_embedding_size=config.static_embedding_size,
            hidden_size=config.hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        zero_static: bool = False,
        use_legacy: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with input alignment and future forcing integration.

        Args:
            x: Dynamic input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Future forcing data [batch_size, output_len, future_input_size]
            zero_static: If True, bypass static feature conditioning
            use_legacy: If True, use the legacy implementation without future data
        """
        if use_legacy or future is None:
            # Legacy mode (backward compatibility)
            if not zero_static:
                # Apply conditional feature mixing
                x = self.conditional_mixing(x, static)

            # Process through mixing layers
            for layer in self.layers:
                x = layer(x)

            return x

        # Modern mode with future forcing
        # Align and fuse historical data with future forcing
        fused = self.alignment_module(
            historical=x, future=future, static=None if zero_static else static
        )

        # Process through mixing layers
        for layer in self.layers:
            fused = layer(fused)

        return fused


class TSMixerHead(nn.Module):
    """Prediction head for TSMixer."""

    def __init__(
        self, input_dim: int, input_len: int, hidden_size: int, output_len: int
    ):
        super().__init__()

        # Adjust the head to work with aligned data
        self.output_len = output_len
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to predictions.

        Args:
            x: Input features [batch_size, seq_len, input_dim]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        # Apply projection to each time step independently
        return self.projection(x)  # [B, output_len, 1]


class TSMixer(nn.Module):
    """Complete TSMixer model with future forcing integration."""

    def __init__(self, config: TSMixerConfig):
        super().__init__()

        self.backbone = TSMixerBackbone(config)

        # Determine input dimension for head based on fusion method
        head_input_dim = (
            config.hidden_size * 2
            if config.fusion_method == "concat"
            else config.hidden_size
        )

        self.head = TSMixerHead(
            input_dim=head_input_dim,
            input_len=config.output_len,  # Now using output_len as sequence length
            hidden_size=config.hidden_size,
            output_len=config.output_len,
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with support for future forcing data.

        Args:
            x: Historical input features of shape [B, input_len, input_size]
            static: Static features of shape [B, static_size]
            future: Optional future forcing data of shape [B, output_len, future_input_size]

        Returns:
            Predictions of shape [B, output_len, 1]
        """
        # Validate input dimensions
        assert x.ndim == 3, "Input tensor x must be of shape [B, input_len, input_size]"
        if static is not None:
            assert static.ndim == 2, "Static tensor must be of shape [B, static_size]"
        if future is not None:
            assert future.ndim == 3, (
                "Future tensor must be of shape [B, output_len, future_input_size]"
            )
        features = self.backbone(x, static, future)
        return self.head(features)


class LitTSMixer(pl.LightningModule):
    """PyTorch Lightning Module implementation of TSMixer."""

    def __init__(
        self,
        config: Union[TSMixerConfig, Dict[str, Any]],
    ) -> None:
        """
        Initialize the Lightning Module with a TSMixerConfig.

        Args:
            config: TSMixer configuration as a TSMixerConfig instance or dict.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the TSMixer model.

        Args:
            x: Historical input features [B, input_len, input_size]
            static: Static features [B, static_size]
            future: Optional future forcing data [B, output_len, future_input_size]

        Returns:
            Predictions [B, output_len, 1]
        """
        return self.model(x, static, future)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            batch: Dictionary with keys:
                "X" -> [B, input_len, input_size],
                "y" -> [B, output_len],
                "static" -> [B, static_size],
                "future" (optional) -> [B, output_len, future_input_size]
            batch_idx: Index of the current batch.

        Returns:
            Computed training loss (MSE) as a tensor.
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")
        y_hat = self(x, static, future)
        loss = self.mse_criterion(y_hat, y)
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute a single validation step.

        Args:
            batch: Dictionary with keys similar to training_step.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with validation loss, predictions, and targets.
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")
        y_hat = self(x, static, future)
        loss = self.mse_criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))
        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute a single test step.

        Args:
            batch: Dictionary with keys:
                "X" -> [B, input_len, input_size],
                "y" -> [B, output_len],
                "static" -> [B, static_size],
                "future" (optional) -> [B, output_len, future_input_size],
                plus optional "input_end_date" and "slice_idx" metadata.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with test predictions, observations, and metadata.
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")
        y_hat = self(x, static, future)

        # Calculate loss metrics
        loss = self.mse_criterion(y_hat, y)

        # Log test metrics
        self.log("test_loss", loss, batch_size=x.size(0))
        self.log("test_mse", loss, batch_size=x.size(0))

        # Create output dictionary for evaluation
        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        # Add optional metadata if available in the batch
        if "input_end_date" in batch:
            output["input_end_date"] = batch["input_end_date"]
        if "slice_idx" in batch:
            output["slice_idx"] = batch["slice_idx"]

        # Collect outputs for evaluation
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
            "input_end_date": [
                date for x in self.test_outputs for date in x["input_end_date"]
            ],
            "slice_idx": [idx for x in self.test_outputs for idx in x["slice_idx"]],
        }

        self.test_outputs = []

    def configure_optimizers(self) -> Dict:
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            ),
            "monitor": "val_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
