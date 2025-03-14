import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Union, List


class RepeatLastValuesConfig:
    """Configuration for RepeatLastValues model."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        group_identifier: str = "gauge_id",
        learning_rate: float = 1e-10  # Very small since this model doesn't really learn
    ):
        """Initialize RepeatLastValues configuration.
        
        Args:
            input_size: Number of input features
            output_size: Length of output sequence (forecast horizon)
            group_identifier: Column name for basin ID
            learning_rate: Learning rate for optimizer (minimal for this model)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.group_identifier = group_identifier
        self.learning_rate = learning_rate
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RepeatLastValuesConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    def update(self, **kwargs) -> "RepeatLastValuesConfig":
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self


class RepeatLastValuesCore(nn.Module):
    """
    Core implementation of the RepeatLastValues model.
    
    This model predicts future values by repeating past observations.
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the RepeatLastValues model core.
        
        Args:
            input_size: Number of input features
            output_size: Length of output sequence (forecast horizon)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: repeat the last N values from the input sequence.

        Args:
            x: Input time series, shape [batch_size, input_len, input_size]
            static: Static catchment attributes (not used in this model)
            future: Future forcing data (not used in this model)

        Returns:
            Predictions tensor of shape [batch_size, output_size, 1]
        """
        batch_size = x.size(0)

        # Extract the target variable column (assume last column or single column)
        target_idx = 0 if self.input_size == 1 else -1
        x_target = x[:, :, target_idx : target_idx + 1]  # Keep dimension

        # Take the last output_size values from the input sequence
        # If input sequence is shorter than output_size, use what's available and cycle
        if x.size(1) >= self.output_size:
            # Simple case: take the last output_size values
            last_values = x_target[:, -self.output_size:, :]
        else:
            # Input sequence is shorter than output_size, need to repeat values
            n_repeats = self.output_size // x.size(1) + 1
            repeated_input = x_target.repeat(1, n_repeats, 1)
            last_values = repeated_input[:, -self.output_size:, :]

        # Ensure output shape is [batch_size, output_size, 1]
        return last_values


class LitRepeatLastValues(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the RepeatLastValues model.
    
    This model serves as a baseline for more sophisticated forecasting models by
    using a simple strategy: the forecast for the next N days is the same as the
    last N days of the input sequence.
    """
    
    def __init__(
        self,
        config: Union[RepeatLastValuesConfig, Dict[str, Any]],
    ):
        """
        Initialize the Lightning module.
        
        Args:
            config: Either a RepeatLastValuesConfig object or a dictionary of config parameters
        """
        super().__init__()
        
        # Handle different config input types
        if isinstance(config, dict):
            self.config = RepeatLastValuesConfig.from_dict(config)
        else:
            self.config = config
        
        # Create the core model
        self.model = RepeatLastValuesCore(
            input_size=self.config.input_size,
            output_size=self.config.output_size
        )
        
        # Save hyperparameters
        self.save_hyperparameters(self.config.to_dict())
        
        # Storage for test results
        self.test_outputs = []
        self.test_results = None
        
    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: delegate to core model.

        Args:
            x: Input time series, shape [batch_size, input_len, input_size]
            static: Static catchment attributes (not used in this model)
            future: Future forcing data (not used in this model)

        Returns:
            Predictions tensor of shape [batch_size, output_size, 1]
        """
        return self.model(x, static, future)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute training step (mostly a placeholder since this model doesn't train).

        Args:
            batch: Dictionary containing input data
            batch_idx: Index of batch

        Returns:
            Loss value (MSE between predictions and actual values)
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute validation step.

        Args:
            batch: Dictionary containing input data
            batch_idx: Index of batch

        Returns:
            Dictionary with validation metrics
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Execute test step.

        Args:
            batch: Dictionary containing input data
            batch_idx: Index of batch

        Returns:
            Dictionary with test outputs
        """
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        y_hat = self(x)

        # Store outputs in the same format as other models
        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        # Store additional metadata if available
        if "input_end_date" in batch:
            output["input_end_date"] = batch["input_end_date"]
        if "slice_idx" in batch:
            output["slice_idx"] = batch["slice_idx"]

        self.test_outputs.append(output)
        return output

    def on_test_epoch_start(self) -> None:
        """Reset test outputs at the beginning of the test epoch."""
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        """Consolidate test outputs at the end of the test epoch."""
        if not self.test_outputs:
            print("Warning: No test outputs collected")
            return

        # Aggregate all test outputs to match the format expected by the evaluator
        self.test_results = {
            "predictions": torch.cat([x["predictions"] for x in self.test_outputs]),
            "observations": torch.cat([x["observations"] for x in self.test_outputs]),
            "basin_ids": [bid for x in self.test_outputs for bid in x["basin_ids"]],
        }

        # Add additional metadata if available
        if "input_end_date" in self.test_outputs[0]:
            self.test_results["input_end_date"] = [
                date for x in self.test_outputs for date in x["input_end_date"]
            ]
        if "slice_idx" in self.test_outputs[0]:
            self.test_results["slice_idx"] = [
                idx for x in self.test_outputs for idx in x["slice_idx"]
            ]

        self.test_outputs = []

    def configure_optimizers(self):
        """
        Configure optimizer (not actually used since this model doesn't learn).

        Returns:
            A simple Adam optimizer to maintain compatibility with training loops
        """
        # Use a simple Adam optimizer with a very small learning rate
        # This won't actually be used for learning but maintains API compatibility
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


# For backwards compatibility
RepeatLastValuesModel = LitRepeatLastValues
