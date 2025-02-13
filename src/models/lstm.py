import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from utils.loss_functions import NSELoss
from torch.optim import Adam
import pandas as pd
import numpy as np
from utils.metrics import nash_sutcliffe_efficiency
from typing import Dict


# TODO: Double check lit implementation
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LitLSTM(pl.LightningModule):
    """Simple LSTM PyTorch Lightning Module.

    During testing, returns:
    - predictions: torch.Tensor of shape [batch_size, pred_len]
    - observations: torch.Tensor of shape [batch_size, pred_len]
    - basin_ids: torch.Tensor of shape [batch_size]

    Postprocessing (inverse transforms, metrics, etc.) should be handled externally
    using the returned tensors and the data module's transformation methods.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # LSTM and output layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, output_size)

        # Save parameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Loss function
        self.mse_criterion = MSELoss()

        # Storage for test outputs
        self.test_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Apply dropout to final hidden state
        lstm_out = self.dropout(lstm_out)

        # Project final hidden state to output size
        output = self.projection(lstm_out[:, -1, :])

        return output.unsqueeze(-1)  # [B, pred_len, 1]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

        loss = self.mse_criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

        loss = self.mse_criterion(y_hat, y)

        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

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

        # Collect all outputs
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        self.test_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)
