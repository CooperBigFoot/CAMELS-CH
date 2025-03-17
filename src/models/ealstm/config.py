from typing import ClassVar, List, Optional
from ..base.base_config import BaseConfig


class EALSTMConfig(BaseConfig):
    """Configuration class for Entity-Aware LSTM model.

    EA-LSTM is a model architecture that uses static catchment attributes to modulate
    the LSTM input gate, enabling better transfer learning between different catchments.

    Reference: "Kratzert et al. (2019) - Towards learning universal, regional, and
    local hydrological behaviors via machine learning applied to large-sample datasets"
    https://hess.copernicus.org/articles/23/5089/2019/
    """

    # Define model-specific parameters
    MODEL_PARAMS: ClassVar[List[str]] = [
        "num_layers",
        "bias",
        "scheduler_patience",
        "scheduler_factor",
    ]

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_size: int,
        static_size: int = 0,
        future_input_size: Optional[int] = None,
        hidden_size: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        num_layers: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize EA-LSTM configuration.

        Args:
            input_len: Length of the input sequence (lookback window)
            output_len: Length of the forecast horizon
            input_size: Number of input features per timestep
            static_size: Number of static/time-invariant features
            future_input_size: Number of future forcing features (defaults to input_size minus 1)
            hidden_size: Size of the LSTM hidden state
            dropout: Dropout rate for regularization
            learning_rate: Initial learning rate for optimization
            group_identifier: Column name identifying the grouping variable (e.g., "gauge_id")
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate reduction
            num_layers: Number of stacked LSTM layers
            bias: Whether to use bias in LSTM layers
            **kwargs: Additional parameters
        """
        # Initialize base config with standard parameters
        super().__init__(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size,
            learning_rate=learning_rate,
            group_identifier=group_identifier,
            **kwargs,
        )

        # Set model-specific parameters
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.future_input_size = future_input_size or input_size - 1
        self.group_identifier = group_identifier
        self.static_size = static_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bias = bias
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # Validate parameters
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
