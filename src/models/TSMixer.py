import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from typing import Dict, Optional, Tuple


class TSMixerConfig:
    """Configuration class for TSMixer model.

    This class handles all hyperparameters needed to define the model architecture.
    Having a separate config class makes it easier to experiment with different 
    architectures and save/load model configurations.
    """

    def __init__(
        self,
        input_len: int,
        input_size: int,
        output_len: int,
        static_size: int,
        hidden_size: int = 64,
        static_embedding_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        self.input_len = input_len
        self.input_size = input_size
        self.output_len = output_len
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.static_embedding_size = static_embedding_size
        self.num_layers = num_layers
        self.dropout = dropout


class ResBlock(nn.Module):
    """Residual block combining temporal and feature mixing.

    This block is the core component of TSMixer, performing two types of mixing:
    1. Temporal mixing: Processes each feature across time steps
    2. Channel mixing: Processes each time step across features

    Both mixing operations use residual connections to help with gradient flow
    and maintain the original signal strength.
    """

    def __init__(self, input_dim: int, hidden_size: int, dropout: float, input_len: int):
        super().__init__()

        # Temporal mixing - operates on sequence length
        self.temporal = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_len),
            nn.Dropout(dropout),
        )

        # Channel mixing - operates on feature dimension
        self.channel = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal mixing with residual connection
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing with residual connection
        x = x + self.channel(x)
        return x


class TSMixerBackbone(nn.Module):
    def __init__(self, configs: TSMixerConfig):
        super().__init__()
        self.static_proj = nn.Linear(
            configs.static_size, configs.static_embedding_size)
        self.input_dim = configs.input_size + configs.static_embedding_size
        self.layers = nn.ModuleList([
            ResBlock(
                input_dim=self.input_dim,
                hidden_size=configs.hidden_size,
                dropout=configs.dropout,
                input_len=configs.input_len,
            ) for _ in range(configs.num_layers)
        ])

        # Cache tensor for efficiency
        self._zero_static_cache = None
        self._last_batch_size = None
        self._last_device = None

    def _get_zero_static(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get cached zero tensor for static embeddings or create new one if needed."""
        if (self._zero_static_cache is None or
            batch_size != self._last_batch_size or
                device != self._last_device):
            self._zero_static_cache = torch.zeros(
                batch_size,
                self.static_proj.out_features,
                device=device
            )
            self._last_batch_size = batch_size
            self._last_device = device
        return self._zero_static_cache

    def forward(self, x: torch.Tensor, static: torch.Tensor, zero_static: bool = False) -> torch.Tensor:
        """
        Forward pass through the backbone network.

        Args:
            x: Dynamic input features [batch_size, seq_len, input_size]
            static: Static features [batch_size, static_size] 
            zero_static: If True, zero out static embeddings for domain adaptation
        """
        if zero_static:
            # Use cached zero tensor for efficiency
            static_emb = self._get_zero_static(x.size(0), x.device)
        else:
            static_emb = self.static_proj(static)

        # Expand static features across time dimension
        static_emb = static_emb.unsqueeze(1).expand(-1, x.size(1), -1)

        # Combine features and process through mixing layers
        x = torch.cat([x, static_emb], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


class TSMixerHead(nn.Module):
    """Prediction head for TSMixer.

    Takes processed features from the backbone and produces final predictions.
    """

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

    def __init__(self, configs: TSMixerConfig):
        super().__init__()

        self.backbone = TSMixerBackbone(configs)
        self.head = TSMixerHead(
            input_dim=self.backbone.input_dim,
            input_len=configs.input_len,
            hidden_size=configs.hidden_size,
            output_len=configs.output_len
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x, static)
        return self.head(features)


class LitTSMixer(pl.LightningModule):
    """PyTorch Lightning Module implementation of TSMixer.

    This class handles:
    1. Training, validation, and testing loops
    2. Optimization configuration
    3. Fine-tuning controls
    4. Metric logging
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int,
        hidden_size: int = 64,
        static_embedding_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # Create config and model
        self.config = TSMixerConfig(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            hidden_size=hidden_size,
            static_embedding_size=static_embedding_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model = TSMixer(self.config)

        # Save parameters and setup
        self.save_hyperparameters()
        self.learning_rate = learning_rate
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
            "basin_ids": batch["gauge_id"],
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
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=2,
                factor=0.5
            ),
            "monitor": "val_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
