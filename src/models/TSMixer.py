import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from utils.loss_functions import NSELoss
from torch.optim import Adam
import pandas as pd
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

        # Final projection layer for single target prediction
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
        target: str = None,
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
        self.target = target

        # Loss functions
        self.mse_criterion = MSELoss()
        self.nse_criterion = NSELoss()

        # Test outputs
        self.test_outputs = []

    def forward(self, x, static):
        return self.model(x, static)

    # TODO: WTF is this loss metric?
    def training_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"].unsqueeze(-1)  # Add channel dimension
        static = batch["static"]
        y_hat = self(x, static)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        batch_size = x.size(0)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_mse", mse_loss, batch_size=batch_size)
        self.log("train_nse", nse_loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"].unsqueeze(-1)  # Add channel dimension
        static = batch["static"]
        y_hat = self(x, static)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        batch_size = x.size(0)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_mse", mse_loss, batch_size=batch_size)
        self.log("val_nse", nse_loss, batch_size=batch_size)

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(self, batch, batch_idx):
        x, y = batch["X"], batch["y"].unsqueeze(-1)  # Add channel dimension
        static = batch["static"]
        y_hat = self(x, static)
        gauge_id = batch["gauge_id"]

        horizons = []
        predictions = []
        observations = []
        basin_ids = []

        y_pred = y_hat.squeeze(-1).detach().cpu().numpy()  # Remove channel dimension
        y_true = y.squeeze(-1).detach().cpu().numpy()  # Remove channel dimension
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

        self.test_outputs.append(output)
        return output

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        all_horizons = np.concatenate([out["horizons"] for out in self.test_outputs])
        all_predictions = np.concatenate(
            [out["predictions"] for out in self.test_outputs]
        )
        all_targets = np.concatenate([out["targets"] for out in self.test_outputs])
        all_basin_ids = np.concatenate([out["basin_ids"] for out in self.test_outputs])

        df = pd.DataFrame(
            {
                "horizon": all_horizons,
                "prediction": all_predictions,
                "observed": all_targets,
                "basin_id": all_basin_ids,
            }
        )

        if hasattr(self.trainer.datamodule, "inverse_transform_predictions"):
            df["prediction"] = self.trainer.datamodule.inverse_transform_predictions(
                df["prediction"].values, df["basin_id"].values
            )
            df["observed"] = self.trainer.datamodule.inverse_transform_predictions(
                df["observed"].values, df["basin_id"].values
            )

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
            self.log(f"test_MSE_h{horizon}", mse)
            self.log(f"test_MAE_h{horizon}", mae)
            self.log(f"test_NSE_h{horizon}", nse)

        self.test_results = {"forecast_df": df, "horizon_metrics": horizon_metrics}
        self.test_outputs = []

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
