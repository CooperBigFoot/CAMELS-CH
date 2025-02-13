import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np


class TSMixerConfig:
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
    ):
        self.seq_len = input_len
        self.pred_len = output_len
        self.enc_in = input_size
        self.static_size = static_size
        self.d_model = hidden_size
        self.static_embedding_size = static_embedding_size
        self.e_layers = num_layers
        self.dropout = dropout


class ResBlock(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float, seq_len: int):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class TSMixer(nn.Module):
    def __init__(self, configs: TSMixerConfig):
        super(TSMixer, self).__init__()

        # Static feature projection
        self.static_proj = nn.Linear(configs.static_size, configs.static_embedding_size)

        # Combined dimension for time series and static features
        self.input_dim = configs.enc_in + configs.static_embedding_size

        # Mixer layers
        self.layers = nn.ModuleList(
            [
                ResBlock(
                    input_dim=self.input_dim,
                    d_model=configs.d_model,
                    dropout=configs.dropout,
                    seq_len=configs.seq_len,
                )
                for _ in range(configs.e_layers)
            ]
        )

        # Final projection layer for prediction
        self.projection = nn.Sequential(
            nn.Linear(configs.seq_len * self.input_dim, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len),
        )

    def forward(self, x, static):
        batch_size = x.size(0)

        # Project static features and expand to match sequence length
        static_emb = self.static_proj(static)  # [B, static_dim]
        static_emb = static_emb.unsqueeze(1).expand(-1, x.size(1), -1)

        # Combine features
        x = torch.cat([x, static_emb], dim=-1)  # [B, seq_len, input_dim]

        # Process through mixer layers
        for layer in self.layers:
            x = layer(x)

        # Flatten and project to output length
        x = x.reshape(batch_size, -1)
        out = self.projection(x)

        return out.unsqueeze(-1)  # [B, pred_len, 1]


class LitTSMixer(pl.LightningModule):
    """TSMixer PyTorch Lightning Module.

    During testing, returns:
    - predictions: torch.Tensor of shape [batch_size, pred_len]
    - observations: torch.Tensor of shape [batch_size, pred_len]
    - basin_ids: torch.Tensor of shape [batch_size]

    Postprocessing (inverse transforms, metrics, etc.) should be handled externally
    using the returned tensors and the data module's transformation methods.
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

        # Save parameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Loss function
        self.mse_criterion = MSELoss()

        # Storage for test outputs
        self.test_outputs = []

    def forward(self, x, static):
        return self.model(x, static)

    def training_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        loss = self.mse_criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        y_hat = self(x, static)

        loss = self.mse_criterion(y_hat, y)

        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(self, batch, batch_idx):
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

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Collect all outputs
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=2, factor=0.5
            ),
            "monitor": "val_loss",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
