"""Entity-Aware LSTM (EA-LSTM) Implementation.

Code adapted from: https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/ealstm.py

The EA-LSTM extends the standard LSTM by incorporating static (entity-aware) features
that modulate the input gate, allowing the model to learn how static attributes
influence the importance of dynamic inputs.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import MSELoss
from torch.optim import Adam
import pandas as pd
import numpy as np
from utils.loss_functions import NSELoss


class EALSTMConfig:
    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        output_size: int,  # This will be mapped to pred_len
        batch_first: bool = True,
        initial_forget_bias: int = 0,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
    ):
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.pred_len = output_size  # Critical for evaluator compatibility
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.dropout = dropout
        self.learning_rate = learning_rate


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
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        output_size: int,
        batch_first: bool = True,
        initial_forget_bias: int = 0,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # Create config
        self.config = EALSTMConfig(
            input_size_dyn=input_size_dyn,
            input_size_stat=input_size_stat,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_first=batch_first,
            initial_forget_bias=initial_forget_bias,
            dropout=dropout,
            learning_rate=learning_rate,
        )

        # Initialize EALSTM model using config parameters
        self.model = EALSTM(
            input_size_dyn=self.config.input_size_dyn,
            input_size_stat=self.config.input_size_stat,
            hidden_size=self.config.hidden_size,
            batch_first=self.config.batch_first,
            initial_forget_bias=self.config.initial_forget_bias,
        )

        # Add dropout and output projection
        self.dropout = nn.Dropout(self.config.dropout)
        self.projection = nn.Linear(self.config.hidden_size, self.config.pred_len)

        # Save hyperparameters and configure loss
        self.save_hyperparameters()
        self.learning_rate = self.config.learning_rate
        self.mse_criterion = MSELoss()
        self.test_outputs = []

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        # Forward pass through EALSTM
        hidden_states, _ = self.model(x_d, x_s)

        # Apply dropout to final hidden state
        hidden_states = self.dropout(hidden_states)

        # Project to output size using only the last hidden state
        if self.model.batch_first:
            output = self.projection(hidden_states[:, -1, :])
        else:
            output = self.projection(hidden_states[-1])

        return output.unsqueeze(-1)  # [B, pred_len, 1]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x_d, x_s = batch["X"], batch["static"]
        y = batch["y"].unsqueeze(-1)
        y_hat = self(x_d, x_s)

        loss = self.mse_criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=x_d.size(0))
        self.log("train_mse", loss, batch_size=x_d.size(0))

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x_d, x_s = batch["X"], batch["static"]
        y = batch["y"].unsqueeze(-1)
        y_hat = self(x_d, x_s)

        loss = self.mse_criterion(y_hat, y)

        self.log("val_loss", loss, batch_size=x_d.size(0))
        self.log("val_mse", loss, batch_size=x_d.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x_d, x_s = batch["X"], batch["static"]
        y = batch["y"].unsqueeze(-1)
        y_hat = self(x_d, x_s)

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
