"""
Implementation of Entity-Aware LSTM (EA-LSTM) for hydrological forecasting.

Based on the paper: "Kratzert et al. (2019) - Towards learning universal, regional, and
local hydrological behaviors via machine learning applied to large-sample datasets"
https://hess.copernicus.org/articles/23/5089/2019/

This implementation follows the model conventions defined in the project guidelines.
"""

from typing import Dict, Optional, Tuple, Any, Union, List, Type
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EALSTMConfig:
    """Configuration class for EA-LSTM model."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
        future_input_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        num_layers: int = 1,
        bias: bool = True,
    ):
        """
        Initialize EA-LSTM configuration.

        Args:
            input_len: Length of the input sequence (lookback window)
            output_len: Length of the forecast horizon
            input_size: Dimensionality of input features per time step
            static_size: Dimensionality of static/time-invariant features
            hidden_size: Size of the LSTM hidden state
            dropout: Dropout rate for regularization
            future_input_size: Dimensionality of future forcing features
            learning_rate: Learning rate for optimization
            group_identifier: Column name identifying the grouping variable
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            num_layers: Number of stacked LSTM layers
            bias: Whether to use bias in LSTM layers
        """
        self.input_len = input_len
        self.output_len = output_len
        self.input_size = input_size
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.future_input_size = (
            future_input_size
            if future_input_size is not None
            else max(1, input_size - 1)
        )
        self.learning_rate = learning_rate
        self.group_identifier = group_identifier
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.num_layers = num_layers
        self.bias = bias

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EALSTMConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()

    def update(self, **kwargs) -> "EALSTMConfig":
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self


class EALSTMCell(nn.Module):
    """
    Entity-Aware LSTM cell that modulates its input gate using static features.

    The EA-LSTM differs from standard LSTM by conditioning the input gate on static
    features, making the network able to learn entity-specific behaviors. Other gates
    (forget, output) operate only on dynamic inputs as in standard LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        static_size: int,
        bias: bool = True,
    ):
        """
        Initialize an EA-LSTM cell.

        Args:
            input_size: Size of dynamic features
            hidden_size: Size of hidden state
            static_size: Size of static features
            bias: Whether to use bias parameters
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.static_size = static_size
        self.bias = bias

        # Input gate (i_t) uses static features
        self.static_i2i = nn.Linear(static_size, hidden_size, bias=bias)

        # Dynamic input for cell state update
        self.i2c = nn.Linear(input_size, hidden_size, bias=bias)

        # Forget gate (f_t)
        self.i2f = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output gate (o_t)
        self.i2o = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Cell state update
        self.h2c = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single EA-LSTM cell.

        Args:
            dynamic_x: Dynamic input tensor [batch_size, input_size]
            static_x: Static input tensor [batch_size, static_size]
            hidden_state: Previous hidden state (h, c) or None for initial state

        Returns:
            Tuple of new hidden state (h_t, c_t)
        """
        if hidden_state is None:
            batch_size = dynamic_x.size(0)
            h_t = torch.zeros(batch_size, self.hidden_size, device=dynamic_x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=dynamic_x.device)
        else:
            h_t, c_t = hidden_state

        # Input gate: uses static features only
        i_t = torch.sigmoid(self.static_i2i(static_x))

        # Forget gate: uses dynamic inputs and previous hidden state
        f_t = torch.sigmoid(self.i2f(dynamic_x) + self.h2f(h_t))

        # Output gate: uses dynamic inputs and previous hidden state
        o_t = torch.sigmoid(self.i2o(dynamic_x) + self.h2o(h_t))

        # Cell state update: traditional input modulation with input gate
        c_tilde = torch.tanh(self.i2c(dynamic_x) + self.h2c(h_t))
        c_t = f_t * c_t + i_t * c_tilde

        # Hidden state update
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class EALSTM(nn.Module):
    """
    Entity-Aware LSTM model for hydrological forecasting.

    This model uses static catchment attributes to modulate the input gate
    of the LSTM, enabling better transfer learning between different catchments.
    """

    def __init__(self, config: EALSTMConfig):
        """
        Initialize the EA-LSTM model.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config

        # Create stacked EA-LSTM layers
        self.ealstm_cells = nn.ModuleList(
            [
                EALSTMCell(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    static_size=config.static_size,
                    bias=config.bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Add dropout between layers
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Projection from hidden state to output (for multi-step forecasting)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_len),
        )

        # Future forcing integration (if provided)
        if config.future_input_size > 0:
            self.future_forcing_layer = nn.Sequential(
                nn.Linear(
                    config.future_input_size * config.output_len, config.hidden_size
                ),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.output_len),
            )
        else:
            self.future_forcing_layer = None

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, input_len, input_size]
        static: torch.Tensor,  # [batch_size, static_size]
        future: Optional[
            torch.Tensor
        ] = None,  # [batch_size, output_len, future_input_size]
    ) -> torch.Tensor:  # [batch_size, output_len, 1]
        """
        Forward pass of the EA-LSTM model.

        Args:
            x: Dynamic input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Optional future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Forecast tensor [batch_size, output_len, 1]
        """
        batch_size = x.size(0)

        # Process each time step through EA-LSTM cells
        hidden_states = [None] * self.config.num_layers

        # Process the sequence through EA-LSTM
        for t in range(self.config.input_len):
            x_t = x[:, t, :]  # [batch_size, input_size]

            # Pass through all LSTM layers
            for layer in range(self.config.num_layers):
                # Get input for this layer
                if layer == 0:
                    layer_input = x_t
                else:
                    # Get the h_t from the previous layer's output
                    h_t, _ = hidden_states[
                        layer - 1
                    ]  # This line uses the previous layer's h_t
                    if self.dropout is not None:
                        layer_input = self.dropout(h_t)
                    else:
                        layer_input = h_t

                # Process through EA-LSTM cell
                h_t, c_t = self.ealstm_cells[layer](
                    dynamic_x=layer_input,
                    static_x=static,
                    hidden_state=hidden_states[layer],
                )
                hidden_states[layer] = (h_t, c_t)

        # Use final hidden state for projection
        final_h = hidden_states[-1][0]  # [batch_size, hidden_size]

        # Project hidden state to output sequence
        output = self.projection(final_h)  # [batch_size, output_len]

        # Integrate future forcing if available
        if future is not None and self.future_forcing_layer is not None:
            # Flatten future features
            future_flat = future.reshape(
                batch_size, -1
            )  # [batch_size, output_len * future_input_size]

            # Project future features
            future_effect = self.future_forcing_layer(
                future_flat
            )  # [batch_size, output_len]

            # Combine with LSTM output
            output = output + future_effect

        # Reshape to [batch_size, output_len, 1]
        return output.unsqueeze(-1)


class LitEALSTM(pl.LightningModule):
    """PyTorch Lightning Module implementation of EA-LSTM."""

    def __init__(
        self,
        config: Union[EALSTMConfig, Dict[str, Any]],
    ) -> None:
        """
        Initialize the Lightning Module with an EALSTMConfig.

        Args:
            config: EA-LSTM configuration as an EALSTMConfig instance or dict
        """
        super().__init__()

        # Handle different config input types
        if isinstance(config, dict):
            self.config = EALSTMConfig.from_dict(config)
        else:
            self.config = config

        # Create the EA-LSTM model using the config
        self.model = EALSTM(self.config)

        # Save all hyperparameters from config for reproducibility
        self.save_hyperparameters(self.config.to_dict())

        # Set up criteria and tracking variables
        self.mse_criterion = MSELoss()
        self.test_outputs = []
        self.test_results = None

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that delegates to the EA-LSTM model.

        Args:
            x: Historical input features [batch_size, input_len, input_size]
            static: Static features [batch_size, static_size]
            future: Optional future forcing data [batch_size, output_len, future_input_size]

        Returns:
            Predictions [batch_size, output_len, 1]
        """
        return self.model(x, static, future)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            batch: Dictionary with keys:
                "X" -> [batch_size, input_len, input_size],
                "y" -> [batch_size, output_len],
                "static" -> [batch_size, static_size],
                "future" (optional) -> [batch_size, output_len, future_input_size]
            batch_idx: Index of the current batch

        Returns:
            Computed training loss (MSE) as a tensor
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
            batch: Dictionary with keys similar to training_step
            batch_idx: Index of the current batch

        Returns:
            Dictionary with validation loss, predictions, and targets
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
                "X" -> [batch_size, input_len, input_size],
                "y" -> [batch_size, output_len],
                "static" -> [batch_size, static_size],
                "future" (optional) -> [batch_size, output_len, future_input_size],
                plus optional "input_end_date" and "slice_idx" metadata.
            batch_idx: Index of the current batch

        Returns:
            Dictionary with test predictions, observations, and metadata
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
        """Reset test outputs collector at the beginning of test epoch."""
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        """Consolidate all test outputs at the end of test epoch."""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Consolidate all test outputs into a single dictionary
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        # Add optional metadata if available
        if "input_end_date" in self.test_outputs[0]:
            self.test_results["input_end_date"] = [
                date for x in self.test_outputs for date in x["input_end_date"]
            ]

        if "slice_idx" in self.test_outputs[0]:
            self.test_results["slice_idx"] = [
                idx for x in self.test_outputs for idx in x["slice_idx"]
            ]

        # Clear temporary storage
        self.test_outputs = []

    def configure_optimizers(self) -> Dict:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
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
