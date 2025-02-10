"""Entity-Aware LSTM (EA-LSTM) Implementation.

Code adapted from: https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/ealstm.py

The EA-LSTM extends the standard LSTM by incorporating static (entity-aware) features
that modulate the input gate, allowing the model to learn how static attributes 
influence the importance of dynamic inputs.
"""

from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd
import numpy as np
from utils.loss_functions import NSELoss


class EALSTM(nn.Module):
    """Entity-Aware LSTM implementation.

    The EA-LSTM extends traditional LSTM architecture by using static features to
    modulate the input gate, enabling the model to learn how static catchment
    attributes influence the processing of dynamic inputs.

    Args:
        input_size_dyn: Number of dynamic features passed to the LSTM at each time step.
        input_size_stat: Number of static features used to modulate the input gate.
        hidden_size: Number of hidden/memory cells.
        batch_first: If True, batch dimension is first in input tensor shape. If False,
            sequence dimension is first. Defaults to True.
        initial_forget_bias: Initial value for forget gate bias. Defaults to 0.

    Attributes:
        input_size_dyn: Size of dynamic input features.
        input_size_stat: Size of static input features.
        hidden_size: Number of hidden units.
        batch_first: Whether batch dimension comes first in input tensors.
        initial_forget_bias: Initial forget gate bias value.
        weight_ih: Input-to-hidden weights for dynamic features.
        weight_hh: Hidden-to-hidden weights.
        weight_sh: Static-to-hidden weights.
        bias: Bias terms for dynamic features.
        bias_s: Bias terms for static features.
    """

    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        batch_first: bool = True,
        initial_forget_bias: int = 0,
    ):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # Create tensors of learnable parameters
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size_dyn, 3 * hidden_size)
        )
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_stat, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM.

        Initializes weights using orthogonal initialization for input-to-hidden
        and static-to-hidden weights. Hidden-to-hidden weights are initialized as
        block diagonal matrices. Biases are initialized to zero, with optional
        initial forget gate bias.
        """
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[: self.hidden_size] = self.initial_forget_bias

    def forward(
        self, x_d: torch.Tensor, x_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the EA-LSTM.

        Args:
            x_d: Tensor containing batch of sequences of dynamic features.
                Shape: [batch_size, seq_len, input_size_dyn] if batch_first=True,
                else [seq_len, batch_size, input_size_dyn].
            x_s: Tensor containing batch of static features.
                Shape: [batch_size, input_size_stat].

        Returns:
            tuple: Contains:
                - h_n: Hidden states for each time step.
                    Shape: [batch_size, seq_len, hidden_size] if batch_first=True,
                    else [seq_len, batch_size, hidden_size].
                - c_n: Cell states for each time step.
                    Shape: Same as h_n.

        Note:
            The input gate is calculated once using static features and then
            applied at each time step. This allows the model to learn how static
            catchment attributes modulate the influence of dynamic inputs.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # Empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # Expand bias vectors to batch size
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())

        # Calculate input gate only once because inputs are static
        bias_s_batch = self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size())
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # Perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # Calculate gates
            gates = torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(
                x_d[t], self.weight_ih
            )
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # Store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

    """PyTorch Lightning wrapper for the Entity-Aware LSTM (EA-LSTM).

This module implements a PyTorch Lightning wrapper around the EA-LSTM model, adding
support for training, validation, and testing loops with built-in logging and metrics
calculation. The wrapper handles both dynamic and static features, making it suitable
for hydrological forecasting tasks where catchment attributes need to be considered.
"""


class LitEALSTM(pl.LightningModule):
    """PyTorch Lightning wrapper for EA-LSTM model.

    This class wraps the EA-LSTM model with PyTorch Lightning functionality,
    providing structured training, validation, and testing loops. It includes
    built-in logging for multiple metrics (MSE, NSE) and handles both dynamic
    and static input features.

    Args:
        input_size_dyn: Number of dynamic input features.
        input_size_stat: Number of static input features.
        hidden_size: Number of hidden units in the LSTM.
        output_size: Number of output features (prediction horizons).
        target: Name of the target variable being predicted.

    Attributes:
        model: The underlying EA-LSTM model.
        fc: Fully connected layer for final predictions.
        mse_criterion: Mean squared error loss function.
        nse_criterion: Nash-Sutcliffe efficiency loss function.
        test_outputs: List to store test predictions.
        test_results: Dictionary containing test results and metrics.
    """

    def __init__(
        self, input_size_dyn, input_size_stat, hidden_size, output_size, target
    ):
        super().__init__()
        self.model = EALSTM(
            input_size_dyn, input_size_stat, hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.mse_criterion = MSELoss(reduction="mean")
        self.nse_criterion = NSELoss()
        self.save_hyperparameters()
        self.test_outputs = []

    def forward(self, x_d, x_s):
        """Forward pass of the model.

        Args:
            x_d: Dynamic input features of shape [batch_size, seq_len, input_size_dyn].
            x_s: Static input features of shape [batch_size, input_size_stat].

        Returns:
            torch.Tensor: Model predictions of shape [batch_size, output_size].
        """
        h_n, _ = self.model(x_d, x_s)
        out = self.fc(h_n[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        """Training step logic.

        Args:
            batch: Dictionary containing input features and targets.
            batch_idx: Index of current batch.

        Returns:
            torch.Tensor: Computed loss value for optimization.
        """
        x, y = batch["X"], batch["y"]
        static = batch["static"]
        batch_size = x.size(0)

        y_hat = self(x, static)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        # Explicitly specify batch_size in logging
        self.log("train_loss", loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(
            "train_mse", mse_loss, batch_size=batch_size, on_step=True, on_epoch=True
        )
        self.log(
            "train_nse", nse_loss, batch_size=batch_size, on_step=True, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic.

        Args:
            batch: Dictionary containing input features and targets.
            batch_idx: Index of current batch.

        Returns:
            dict: Dictionary containing validation loss and predictions.
        """
        x, y = batch["X"], batch["y"]
        static = batch["static"]
        batch_size = x.size(0)

        y_hat = self(x, static)

        mse_loss = self.mse_criterion(y_hat, y)
        nse_loss = self.nse_criterion(y_hat, y)
        loss = mse_loss + nse_loss

        # Explicitly specify batch_size in logging
        self.log("val_loss", loss, batch_size=batch_size, on_epoch=True)
        self.log("val_mse", mse_loss, batch_size=batch_size, on_epoch=True)
        self.log("val_nse", nse_loss, batch_size=batch_size, on_epoch=True)

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(self, batch, batch_idx):
        """Test step logic.

        Processes a test batch and stores predictions for later analysis.

        Args:
            batch: Dictionary containing input features and targets.
            batch_idx: Index of current batch.

        Returns:
            dict: Dictionary containing test predictions and related information.
        """
        x, y = batch["X"], batch["y"]
        static = batch["static"]
        gauge_id = batch["gauge_id"]

        y_hat = self(x, static)

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

        self.test_outputs.append(output)
        return output

    def on_test_epoch_start(self):
        """Called at the beginning of testing.

        Initializes storage for test outputs.
        """
        self.test_outputs = []

    def on_test_epoch_end(self):
        """Called at the end of testing.

        Processes all test outputs and computes metrics for each prediction horizon.
        Creates a DataFrame with predictions and calculates MSE, MAE, and NSE metrics.
        """
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
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with learning rate 0.001.
        """
        return Adam(self.parameters(), lr=0.001)
