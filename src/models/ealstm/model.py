"""
Implementation of Entity-Aware LSTM (EA-LSTM) for hydrological forecasting.

Based on the paper: "Kratzert et al. (2019) - Towards learning universal, regional, and
local hydrological behaviors via machine learning applied to large-sample datasets"
https://hess.copernicus.org/articles/23/5089/2019/
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .config import EALSTMConfig


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
               (contains target as the first feature, followed by optional past features)
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
                    h_t, _ = hidden_states[layer - 1]
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
            future_effect = self.future_forcing_layer(future_flat)  # [batch_size, output_len]

            # Combine with LSTM output
            output = output + future_effect

        # Reshape to [batch_size, output_len, 1]
        return output.unsqueeze(-1)
