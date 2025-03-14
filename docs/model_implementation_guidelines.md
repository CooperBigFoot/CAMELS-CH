# Model Implementation Conventions

This document defines the conventions for implementing and extending models in our hydrological time series analysis project. These guidelines ensure consistency, maintainability, and compatibility when refactoring code or introducing new models. All models in this project should adhere to these standards, including architectures like TSMixer, TiDE, LSTM, RepeatLastValues, and others.

## 1. Overall Structure

Every model implementation should consist of three key components:

1. **Configuration Class**: Encapsulates all hyperparameters and model-specific settings as the single source of truth for configuration values, with methods for conversion and safe updates.

2. **Core Model (`nn.Module`)**: Implements the main computational logic using PyTorch's `nn.Module`, potentially including multiple submodules for different architectural components.

3. **PyTorch Lightning Module**: Handles training, validation, testing, and logging, serving as the interface between the model and the training pipeline.

> **Note**: All configuration details must be managed exclusively through the configuration class, never hardcoded in the model implementation.

## 2. Model Configuration

### Purpose

The configuration class centralizes all hyperparameters required by the model, ensuring that model-specific details (dimensions, layer counts, learning rates, etc.) are defined in one place.

### Implementation Guidelines

- Use a dedicated configuration class for each model (e.g., `TSMixerConfig`, `TiDEConfig`).
- Provide methods to:
  - Convert configuration to dictionary (`to_dict`)
  - Create configuration from dictionary (`from_dict`)
  - Update parameters safely (`update`)
- Use appropriate type hints for all attributes.
- Implement validation logic for interdependent parameters.
- Provide clear default values that work reasonably well.

### Standard Hyperparameters

All model configurations should include these standard parameters (with consistent naming):

- `input_len`: Length of the historical input sequence (lookback window)
- `output_len`: Length of the forecast horizon (prediction steps)
- `input_size`: Dimensionality of input features per time step (dynamic features)
- `static_size`: Dimensionality of static/time-invariant features
- `future_input_size`: Dimensionality of future forcing features (when applicable)
- `hidden_size`: Size of hidden layers in the model
- `learning_rate`: Initial learning rate for optimization
- `dropout`: Dropout rate for regularization
- `group_identifier`: Column name identifying the grouping variable (e.g., "gauge_id")

### Example Configuration Class

```python
class TSMixerConfig:
    """Configuration for TSMixer model."""

    def __init__(
        self,
        input_len: int,
        input_size: int,
        output_len: int,
        static_size: int,
        hidden_size: int = 64,
        static_embedding_size: int = 10,
        num_layers: int = 5,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        group_identifier: str = "gauge_id",
    ):
        """Initialize TSMixer configuration."""
        self.input_len = input_len
        self.input_size = input_size
        self.output_len = output_len
        self.static_size = static_size
        self.hidden_size = hidden_size
        self.static_embedding_size = static_embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.group_identifier = group_identifier

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TSMixerConfig":
        """Create a config object from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()

    def update(self, **kwargs) -> "TSMixerConfig":
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
```

## 3. Core Model (`nn.Module`) Implementation

### Purpose

The core model implements the model's architecture and computational logic, encapsulating all layers, operations, and forward pass behavior.

### Implementation Guidelines

- Break down complex models into clear, modular sub-components using separate classes (e.g., encoder, decoder, mixing blocks).
- Use clear, descriptive names for layers and operations.
- Ensure the forward method accepts inputs for:
  - Historical data (dynamic features)
  - Static features (when applicable)
  - Future forcing features (when applicable)
- Document tensor shapes and dimensional assumptions in docstrings.
- Implement reusable components that can be shared across models when appropriate.

### Standard Forward Method Signature

All core models should implement a forward method with this signature:

```python
def forward(
    self, 
    x: torch.Tensor,               # [batch_size, input_len, input_size]
    static: Optional[torch.Tensor] = None,  # [batch_size, static_size]
    future: Optional[torch.Tensor] = None,  # [batch_size, output_len, future_input_size]
) -> torch.Tensor:                 # [batch_size, output_len, 1]
    """Forward pass implementation."""
    pass
```

Here's the new section focusing on what the model should expect from the data modules:

## 3a. Data Interface Expectations

Models in our framework interact with standardized data provided by the `HydroDataModule` and `HydroDataset` classes. Understanding this interface is essential for implementing compatible models.

### Batch Structure

All models should expect batches with the following structure:

```python
{
    "X": torch.Tensor,              # [batch_size, input_len, input_size] - Historical time series data
    "y": torch.Tensor,              # [batch_size, output_len] - Target values to predict
    "static": torch.Tensor,         # [batch_size, static_size] - Static catchment attributes
    "future": torch.Tensor,         # [batch_size, output_len, future_input_size] - Future forcing data (optional)
    "gauge_id": List[str],          # Basin identifiers
    "slice_idx": List[List[int]],   # Original indices in the dataset
    "input_end_date": List[str],    # End dates of input windows
    "domain_id": torch.Tensor,      # [batch_size, 1] - Domain identifier (for transfer learning)
    "domain_name": str              # Domain name (for transfer learning)
}
```

Note that not all fields will be present in every batch. Models should handle cases where optional elements (particularly `future`) are missing.

### Data Characteristics

- **Historical Data (`X`)**: Contains the target variable and potentially other dynamic features for the input window. The last feature is always the target variable.
- **Target Values (`y`)**: Contains the target values for the forecast horizon. During training, these are the ground truth values to predict.
- **Static Features (`static`)**: Contains time-invariant catchment attributes preprocessed as tensors.
- **Future Forcing (`future`)**: When available, contains known or forecasted external variables for the prediction period. Not all datasets will provide this.

### Handling Missing Components

Models should implement graceful fallbacks when optional components are missing:

```python
def forward(
    self, 
    x: torch.Tensor, 
    static: torch.Tensor, 
    future: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Forward pass with robust handling of optional components."""
    # Handle case where future forcing is not available
    if future is None:
        # Implement fallback behavior
        pass
    
    # Rest of implementation
    ...
```

### Preprocessing Considerations

- All features are preprocessed by the `HydroDataModule` according to the configured pipeline.
- Models receive standardized data (typically z-score normalized) and should output predictions in the same scale.
- The `HydroDataModule` handles inverse transformations when evaluating performance metrics.

### Transfer Learning Support

For models supporting transfer learning:

- The `domain_id` tensor indicates whether each sample is from the source (0.0) or target (1.0) domain.
- The `domain_name` provides a string identifier for the specific domain (e.g., "CH" for Switzerland).
- Models can use this information to implement domain-specific processing or domain adaptation techniques.

By designing models to work with this standardized interface, we ensure compatibility with the training pipeline and facilitate easier comparison between different architectures.

## 4. PyTorch Lightning Module Wrapper

### Purpose

The Lightning module wraps the core model and handles training logic, loss computation, metric logging, and optimization.

### Implementation Guidelines

#### Class Structure

- Accept either a configuration object or a dictionary in the constructor.
- Save hyperparameters using `self.save_hyperparameters()`.
- Store test outputs in a standardized format for evaluation.

#### Training, Validation, and Testing

- Implement consistent methods for each phase:
  - `training_step`
  - `validation_step`
  - `test_step`
  - `configure_optimizers`
- Each method should handle input extraction, forward pass, loss computation, and metric logging.
- `test_step` should store predictions, observations, and basin IDs for later evaluation.

#### Logging

Every model must log the following metrics consistently:

- `train_loss`: Main loss during training
- `train_mse`: Mean squared error during training (if different from main loss)
- `val_loss`: Validation loss
- `val_mse`: Mean squared error during validation
- Additional metrics can be added with clear, consistent naming.

#### Standardized Test Output

Test results should be stored in a dictionary with these keys:

- `predictions`: Model outputs
- `observations`: Ground truth values
- `basin_ids`: Identifiers for each sample
- `input_end_date`: (if available) End dates of input window
- `slice_idx`: (if available) Original indices in the dataset

### Example Lightning Module

```python
class LitTSMixer(pl.LightningModule):
    """PyTorch Lightning Module implementation of TSMixer."""

    def __init__(
        self,
        config: Union[TSMixerConfig, Dict[str, Any]],
    ):
        """Initialize the Lightning Module with a TSMixerConfig."""
        super().__init__()

        # Handle different config input types
        if isinstance(config, dict):
            self.config = TSMixerConfig.from_dict(config)
        else:
            self.config = config

        # Create the TSMixer model using the config
        self.model = TSMixer(self.config)

        # Save all hyperparameters from config
        self.save_hyperparameters(self.config.to_dict())

        # Set up criteria and tracking variables
        self.mse_criterion = MSELoss()
        self.test_outputs = []

    def forward(
        self, 
        x: torch.Tensor, 
        static: torch.Tensor, 
        future: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        return self.model(x, static, future)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute training step."""
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")  # Get future forcing if available
        
        # Forward pass
        y_hat = self(x, static, future)

        # Calculate loss
        loss = self.mse_criterion(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_mse", loss, batch_size=x.size(0))

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute validation step."""
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")
        
        # Forward pass
        y_hat = self(x, static, future)

        # Calculate loss
        loss = self.mse_criterion(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, batch_size=x.size(0))
        self.log("val_mse", loss, batch_size=x.size(0))

        return {"val_loss": loss, "preds": y_hat, "targets": y}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute test step."""
        x, y = batch["X"], batch["y"].unsqueeze(-1)
        static = batch["static"]
        future = batch.get("future")
        
        # Forward pass
        y_hat = self(x, static, future)

        # Store outputs
        output = {
            "predictions": y_hat.squeeze(-1),
            "observations": y.squeeze(-1),
            "basin_ids": batch[self.config.group_identifier],
        }

        self.test_outputs.append(output)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.config.learning_rate)
```

## 5. Hyperparameter Naming Conventions

To maintain clarity and consistency, use these naming conventions for hyperparameters:

### Lengths and Dimensions

- `input_len`: Length of the historical input sequence (time steps).
- `output_len`: Forecast horizon or number of future steps to predict.
- `input_size`: Dimensionality of input features per time step.
- `static_size`: Dimensionality of static (time-invariant) features.
- `future_input_size`: Dimensionality of future forcing features (when applicable).
- `hidden_size`: Size of hidden layers in the model.

### Other Parameters

- Use the suffix `_size` for dimensions of vector spaces.
- Use the suffix `_len` for sequence lengths or time steps.
- Use the suffix `_dim` only when referring to specific dimensions in a tensor.
- Use descriptive prefixes to differentiate sizes for different components (e.g., `encoder_hidden_size`, `decoder_hidden_size`).

## 6. Feature Terminology

For clarity and consistency, use these definitions throughout the codebase:

- **Target**: The primary time series variable being forecast (typically streamflow).
- **Dynamic Features**: Time-varying features that include the target and potentially other time series.
- **Static Features**: Time-invariant features that describe fixed properties (e.g., catchment attributes).
- **Future Forcing Features**: Known or forecasted external variables for the prediction period.
- **Forcing Features**: External inputs (not the target) used to improve the forecast, either historical or future.

## 7. Logging Conventions

All models must consistently log these metrics:

### Required Metrics

- `train_loss`: Loss value during training.
- `val_loss`: Loss value during validation.
- `train_mse`: Mean squared error during training.
- `val_mse`: Mean squared error during validation.

### Additional Metrics

Models may log additional metrics like:

- `train_rmse`: Root mean squared error during training.
- `val_rmse`: Root mean squared error during validation.
- `learning_rate`: Current learning rate (if using a scheduler).

### Metric Naming Convention

- Use the format `{phase}_{metric}` where:
  - `phase` is one of: `train`, `val`, `test`
  - `metric` describes the measurement: `loss`, `mse`, `rmse`, etc.

## 8. Model-Agnostic Training Script

A model-agnostic training script should:

1. Take a model configuration file as input.
2. Load and prepare data using standard data modules.
3. Instantiate the appropriate model class based on the configuration.
4. Set up training parameters, callbacks, and loggers.
5. Train the model and evaluate performance.
6. Save model checkpoints and evaluation results.

Specific implementation details will be added in a future update.

## 9. Code Style and Documentation

### Code Style

- Follow PEP 8 standards for Python code formatting.
- Use consistent indentation (4 spaces).
- Keep line length under 88 characters (compatible with Black formatter).
- Use meaningful variable names that reflect hydrological domain concepts.

### Documentation

- Include comprehensive docstrings for all classes and methods.
- Follow Google docstring style:

  ```python
  def function(arg1, arg2):
      """Short description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1.
          arg2: Description of arg2.
          
      Returns:
          Description of return value.
          
      Raises:
          ExceptionType: When and why this exception is raised.
      """
  ```

- Document expected tensor shapes in the forward method.
- Include examples for complex methods or classes.

### Type Hints

- Use type hints for all function and method arguments.
- Use generic types (e.g., `List`, `Dict`, `Optional`) from the `typing` module.
- Use `Union` for parameters that can accept multiple types.

## 10. Testing and Error Handling

### Error Handling

- Validate input shapes and types in forward methods.
- Provide clear error messages that help identify the source of problems.
- Add assertions for critical assumptions and invariants.

## 11. Extensions and Special Cases

This section will be expanded as new requirements or special cases are identified. It may include guidelines for:

- Transfer learning capabilities
- Handling missing data
- Model interpretability features
- Uncertainty quantification
