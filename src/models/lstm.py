import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from utils.loss_functions import NSELoss
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
        self.mse_criterion = MSELoss(reduction="mean")
        self.nse_criterion = NSELoss()
        self.save_hyperparameters()
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        batch_size = x.size(0)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_mse", mse_loss, batch_size=batch_size)
        self.log("train_nse", nse_loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        batch_size = x.size(0)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_mse", mse_loss, batch_size=batch_size)
        self.log("val_nse", nse_loss, batch_size=batch_size)

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"]
        y_hat = self(x)
        gauge_id = batch["gauge_id"]

        horizons = []
        predictions = []
        observations = []
        basin_ids = []

        y_pred = y_hat.detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        gauge_ids = (
            gauge_id.cpu().numpy() if isinstance(gauge_id, torch.Tensor) else gauge_id
        )

        for i in range(len(y_pred)):
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

        self.test_outputs.append(output)  # Store the output
        return output

    def on_test_epoch_start(self):
        self.test_outputs = []  # Reset test outputs at the start of test epoch

    def on_test_epoch_end(self):
        # Check if we have any test outputs
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Concatenate all outputs
        all_horizons = np.concatenate([out["horizons"] for out in self.test_outputs])
        all_predictions = np.concatenate(
            [out["predictions"] for out in self.test_outputs]
        )
        all_targets = np.concatenate([out["targets"] for out in self.test_outputs])
        all_basin_ids = np.concatenate([out["basin_ids"] for out in self.test_outputs])

        # Create DataFrame
        df = pd.DataFrame(
            {
                "horizon": all_horizons,
                "prediction": all_predictions,
                "observed": all_targets,
                "basin_id": all_basin_ids,
            }
        )

        # Get inverse transformed values using the datamodule
        if hasattr(self.trainer.datamodule, "inverse_transform_predictions"):
            df["prediction"] = self.trainer.datamodule.inverse_transform_predictions(
                df["prediction"].values, df["basin_id"].values
            )
            df["observed"] = self.trainer.datamodule.inverse_transform_predictions(
                df["observed"].values, df["basin_id"].values
            )

        # Calculate metrics per horizon
        horizon_metrics = {}
        for horizon in range(1, int(max(all_horizons)) + 1):
            horizon_data = df[df["horizon"] == horizon]

            pred_tensor = torch.tensor(
                horizon_data["prediction"].values, dtype=torch.float32
            )
            obs_tensor = torch.tensor(
                horizon_data["observed"].values, dtype=torch.float32
            )

            mse = self.mse_criterion(pred_tensor, obs_tensor).item()
            mae = torch.mean(torch.abs(pred_tensor - obs_tensor)).item()
            nse = 1 - self.nse_criterion(pred_tensor, obs_tensor).item()

            horizon_metrics[horizon] = {"MSE": mse, "MAE": mae, "NSE": nse}

            # Log metrics for each horizon
            self.log(f"test_MSE_h{horizon}", mse)
            self.log(f"test_MAE_h{horizon}", mae)
            self.log(f"test_NSE_h{horizon}", nse)

        # Store results
        self.test_results = {"forecast_df": df, "horizon_metrics": horizon_metrics}

        # Clear test outputs
        self.test_outputs = []

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
