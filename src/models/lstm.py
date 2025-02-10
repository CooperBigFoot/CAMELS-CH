import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd
import numpy as np
from utils.metrics import nash_sutcliffe_efficiency


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

        # Get predictions and targets for each horizon
        horizons = []
        predictions = []
        observations = []
        basin_ids = []

        # Convert tensors to numpy arrays
        y_pred = y_hat.detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        gauge_ids = (
            gauge_id.cpu().numpy() if isinstance(gauge_id, torch.Tensor) else gauge_id
        )

        # For each sample in the batch
        for i in range(len(y_pred)):
            # For each forecast horizon
            for h in range(y_pred.shape[1]):
                horizons.append(h + 1)
                predictions.append(y_pred[i, h])
                observations.append(y_true[i, h])
                basin_ids.append(gauge_ids[i])

        output = {
            "horizons": horizons,
            "predictions": predictions,
            "targets": observations,
            "basin_ids": basin_ids,
        }

        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # Concatenate all batches
        all_horizons = np.concatenate([out["horizons"] for out in self.test_outputs])
        all_predictions = np.concatenate(
            [out["predictions"] for out in self.test_outputs]
        )
        all_targets = np.concatenate([out["targets"] for out in self.test_outputs])
        all_basin_ids = np.concatenate([out["basin_ids"] for out in self.test_outputs])

        # Create DataFrame with all results
        results_df = pd.DataFrame(
            {
                "horizon": all_horizons,
                "prediction": all_predictions,
                "observed": all_targets,
                "basin_id": all_basin_ids,
            }
        )

        # Calculate metrics per horizon
        horizon_metrics = {}
        for horizon in range(1, max(all_horizons) + 1):
            horizon_data = results_df[results_df["horizon"] == horizon]

            # Calculate metrics for this horizon
            mse = np.mean((horizon_data["observed"] - horizon_data["prediction"]) ** 2)
            mae = np.mean(np.abs(horizon_data["observed"] - horizon_data["prediction"]))
            nse = nash_sutcliffe_efficiency(
                horizon_data["observed"].values, horizon_data["prediction"].values
            )

            horizon_metrics[horizon] = {"MSE": mse, "MAE": mae, "NSE": nse}

        # Store results
        self.test_results = {
            "forecast_df": results_df,
            "horizon_metrics": horizon_metrics,
        }

        # Log metrics
        for horizon, metrics in horizon_metrics.items():
            for metric_name, value in metrics.items():
                self.log(f"test_{metric_name}_h{horizon}", value)

        # Clear outputs
        self.test_outputs = []

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
