import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd
import numpy as np


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
    def __init__(self, input_size, hidden_size, num_layers, output_size, target):
        super().__init__()
        self.model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)
        self.criterion = MSELoss()
        self.save_hyperparameters()
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, batch_size=32)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, batch_size=32)
        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)
        gauge_id = batch["gauge_id"]
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, batch_size=32)

        output = {
            "preds": y_hat.detach().cpu().numpy(),
            "targets": y.detach().cpu().numpy(),
            "gauge_id": (
                gauge_id.cpu().numpy()
                if isinstance(gauge_id, torch.Tensor)
                else gauge_id
            ),
        }
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        self.test_results = {
            "predictions": np.concatenate(
                [out["preds"] for out in self.test_outputs], axis=0
            ),
            "targets": np.concatenate(
                [out["targets"] for out in self.test_outputs], axis=0
            ),
            "basin_ids": np.concatenate(
                [out["gauge_id"] for out in self.test_outputs], axis=0
            ),
        }
        self.test_outputs = []

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
