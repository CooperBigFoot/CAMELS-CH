# Refined Guidelines for Hydrological Forecasting Experiments

This document outlines a practical approach to implementing hydrological forecasting experiments. The goal is to maintain a balance between standardization and flexibility, making experiments easy to develop and understand without unnecessary complexity.

## Core Principles

1. **Standalone experiments**: Each experiment should be completely self-contained, with no dependencies on other experiments
2. **Src dependencies only**: Experiments should only depend on code in the `src` directory
3. **Consistent structure**: Follow a common directory structure and file naming pattern
4. **Clear interfaces**: Use standardized function signatures for key operations
5. **Configuration over code**: Use configuration files to define experiment parameters

## Directory Structure

Each experiment should follow this directory structure:

```
experiments/
  ├── ExperimentName/
  │   ├── experiment.py          # Main entry point
  │   ├── data_loader.py         # Experiment-specific data loading
  │   ├── config.py              # Experiment configuration
  │   ├── utils.py               # Experiment-specific utilities
  │   ├── yaml_files/            # Model hyperparameter YAML files
  │   │   ├── tide.yaml
  │   │   ├── tsmixer.yaml
  │   │   ├── ealstm.yaml
  │   │   └── tft.yaml
  │   └── README.md              # Experiment documentation
```

## Standard Components

### 1. Experiment Configuration (`config.py`)

Define experiment parameters in a dataclass:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ExperimentConfig:
    """Configuration for ExperimentName experiment."""
    # Experiment metadata
    experiment_name: str = "experiment_name"
    
    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    forcing_features: List[str] = field(default_factory=lambda: [
        "temperature_2m_mean",
        "total_precipitation_sum"
    ])
    static_features: List[str] = field(default_factory=lambda: [
        "gauge_id", "area", "elevation"
    ])
    
    # Training parameters
    batch_size: int = 2048
    num_workers: int = 4
    max_epochs: int = 100
    early_stopping_patience: int = 5
    
    # Data splitting parameters
    train_prop: float = 0.6
    val_prop: float = 0.2
    test_prop: float = 0.2
    use_proportional_split: bool = True
    
    # Experiment-specific parameters
    # Add parameters unique to this experiment
    
    def validate(self):
        """Validate configuration parameters."""
        # Example validation
        if self.train_prop + self.val_prop + self.test_prop != 1.0:
            raise ValueError("Data proportions must sum to 1.0")
        
        # Add more validation as needed
```

### 2. Data Loading (`data_loader.py`)

Define a standard `load_data` function:

```python
def load_data(config: Any, **kwargs) -> Dict[str, Any]:
    """
    Load data for the experiment.
    
    Args:
        config: Experiment configuration
        **kwargs: Additional arguments from command line
        
    Returns:
        Dictionary containing at minimum:
        - 'time_series': pd.DataFrame - Time series data
        - 'static': pd.DataFrame - Static catchment attributes
        - Any other experiment-specific data
    """
    # Implement experiment-specific data loading
    pass
```

### 3. Utility Functions for Model Creation

Use the central model factory in `src/utils/model_factory.py`:

```python
def create_model(model_type: str, yaml_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a model instance from a YAML file.
    
    Args:
        model_type: Type of model to create ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML file
        
    Returns:
        Tuple containing:
        - Model instance
        - Dictionary of model hyperparameters
    """
    # Load hyperparameters from YAML
    model_hp = hp_from_yaml(model_type, yaml_path)
    
    # Create appropriate model configuration
    if model_type == "tide":
        from src.models.tide import TiDEConfig, LitTiDE
        model_config = TiDEConfig(**model_hp)
        model = LitTiDE(config=model_config)
    elif model_type == "tsmixer":
        from src.models.tsmixer import TSMixerConfig, LitTSMixer
        model_config = TSMixerConfig(**model_hp)
        model = LitTSMixer(config=model_config)
    elif model_type == "ealstm":
        from src.models.ealstm import EALSTMConfig, LitEALSTM
        model_config = EALSTMConfig(**model_hp)
        model = LitEALSTM(config=model_config)
    elif model_type == "tft":
        from src.models.tft import TFTConfig, LitTFT
        model_config = TFTConfig(**model_hp)
        model = LitTFT(config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return model, model_hp

def load_pretrained_model(model_type: str, yaml_path: str, checkpoint_path: str, 
                          finetune: bool = False, lr_factor: float = 10.0) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a pretrained model from a checkpoint.
    
    Args:
        model_type: Type of model to load
        yaml_path: Path to model hyperparameter YAML file
        checkpoint_path: Path to model checkpoint
        finetune: Whether to prepare model for fine-tuning
        lr_factor: Factor to reduce learning rate by for fine-tuning
        
    Returns:
        Tuple containing:
        - Loaded model instance
        - Dictionary of model hyperparameters
    """
    # Create model config
    model_hp = hp_from_yaml(model_type, yaml_path)
    
    if model_type == "tide":
        from src.models.tide import TiDEConfig, LitTiDE
        model_config = TiDEConfig(**model_hp)
        model = LitTiDE.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "tsmixer":
        from src.models.tsmixer import TSMixerConfig, LitTSMixer
        model_config = TSMixerConfig(**model_hp)
        model = LitTSMixer.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "ealstm":
        from src.models.ealstm import EALSTMConfig, LitEALSTM
        model_config = EALSTMConfig(**model_hp)
        model = LitEALSTM.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "tft":
        from src.models.tft import TFTConfig, LitTFT
        model_config = TFTConfig(**model_hp)
        model = LitTFT.load_from_checkpoint(checkpoint_path, config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Adjust learning rate for fine-tuning if needed
    if finetune:
        original_lr = model.hparams.learning_rate
        model.hparams.learning_rate = original_lr / lr_factor
        
        # Store original learning rate for reference
        model.original_lr = original_lr
        model.fine_tuned_lr = original_lr / lr_factor
    
    return model, model_hp
```

### 4. Main Experiment Script (`experiment.py`)

Structure the main script with these sections:

```python
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.model_factory import create_model, load_pretrained_model
from src.data_models.datamodule import HydroDataModule

from config import ExperimentConfig
from data_loader import load_data

def get_preprocessing_config():
    """Create standard preprocessing configuration."""
    from sklearn.pipeline import Pipeline
    from src.preprocessing.log_scale import LogTransformer
    from src.preprocessing.standard_scale import StandardScaleTransformer
    
    return {
        "features": {"pipeline": Pipeline([("scaler", StandardScaleTransformer())])},
        "target": {"pipeline": Pipeline([
            ("log", LogTransformer()),
            ("scaler", StandardScaleTransformer())
        ])},
        "static_features": {"pipeline": Pipeline([("scaler", StandardScaleTransformer())])},
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Experiment description")
    
    # Common arguments
    parser.add_argument("--output-dir", type=str, default="output", 
                      help="Output directory")
    parser.add_argument("--models", type=str, nargs="+", 
                      default=["tide", "tsmixer", "ealstm", "tft"], 
                      help="Models to evaluate")
    parser.add_argument("--yaml-dir", type=str, default="yaml_files",
                      help="Directory containing YAML files")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of runs per model")
    
    # Experiment-specific arguments
    # Add arguments unique to this experiment
    
    return parser.parse_args()

def setup_dirs(output_dir):
    """Create necessary output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    return {
        "checkpoints": f"{output_dir}/checkpoints",
        "logs": f"{output_dir}/logs",
    }

def train_model(model, data_module, model_type, run_idx, output_dirs, config):
    """Train a model and save checkpoints."""
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=config.early_stopping_patience,
            mode="min"
        ),
        ModelCheckpoint(
            dirpath=f"{output_dirs['checkpoints']}/{model_type}/run_{run_idx}",
            filename=f"{model_type}_{{epoch}}_{{val_loss:.4f}}",
            monitor="val_loss",
            save_top_k=1,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=output_dirs["logs"],
        name=model_type,
        version=f"run_{run_idx}"
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Return best validation loss
    return trainer.callback_metrics.get("val_loss", float("inf"))

def main():
    """Run the experiment."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load experiment configuration
    config = ExperimentConfig()
    
    # Load data
    data = load_data(config, **vars(args))
    
    # Create output directories
    output_dirs = setup_dirs(args.output_dir)
    
    # Get preprocessing configuration
    preprocessing_config = get_preprocessing_config()
    
    # Process each model
    for model_type in args.models:
        # Create model-specific directories
        os.makedirs(f"{output_dirs['checkpoints']}/{model_type}", exist_ok=True)
        
        # Define YAML path
        yaml_path = f"{args.yaml_dir}/{model_type}.yaml"
        
        for run_idx in range(args.num_runs):
            # Set seed for this run
            run_seed = args.seed + run_idx
            pl.seed_everything(run_seed)
            
            # Create model and get hyperparameters
            model, model_hp = create_model(model_type, yaml_path)
            
            # Create data module
            data_module = HydroDataModule(
                time_series_df=data["time_series"],
                static_df=data["static"],
                group_identifier=config.group_identifier,
                preprocessing_config=preprocessing_config,
                batch_size=config.batch_size,
                input_length=model_hp["input_len"],
                output_length=model_hp["output_len"],
                num_workers=config.num_workers,
                features=config.forcing_features,
                static_features=config.static_features,
                target=config.target,
                use_proportional_split=config.use_proportional_split,
                train_prop=config.train_prop,
                val_prop=config.val_prop,
                test_prop=config.test_prop,
            )
            
            # Prepare data
            data_module.prepare_data()
            data_module.setup()
            
            # Train model
            val_loss = train_model(
                model=model,
                data_module=data_module,
                model_type=model_type,
                run_idx=run_idx,
                output_dirs=output_dirs,
                config=config
            )
            
            print(f"Completed run {run_idx+1}/{args.num_runs} for {model_type}: val_loss={val_loss:.4f}")
    
    print("Experiment complete")

if __name__ == "__main__":
    main()
```

## Implementation Guidelines

### 1. Data Loading

- The `load_data` function is responsible for:
  - Loading raw data from disk
  - Applying experiment-specific filtering or transformations
  - Returning a dictionary with standard keys

- Keep all data-specific logic in `data_loader.py`, not in the main script

### 2. Model Creation

- Use the central `model_factory.py` utility for all model creation
- Keep model hyperparameters in YAML files, not hardcoded in the experiment
- For pretrained models, always use the `load_pretrained_model` function

### 3. Configuration Management

- Define default configuration in the `ExperimentConfig` class
- Allow command line arguments to override configuration values
- Use explicit names for configuration parameters (avoid abbreviations)
- Include validation logic in the config class

### 4. Training & Logging

- Focus on generating checkpoints and TensorBoard logs
- Use PyTorch Lightning's built-in callbacks for:
  - Early stopping
  - Model checkpointing
  - Learning rate monitoring
- Handle multiple runs with different seeds

### 5. Output Organization

- Organize output directories consistently:

  ```
  output_dir/
  ├── checkpoints/
  │   ├── tide/
  │   │   ├── run_0/
  │   │   └── run_1/
  │   ├── tsmixer/
  │   │   └── ...
  │   └── ...
  └── logs/
      ├── tide/
      │   ├── run_0/
      │   └── run_1/
      ├── tsmixer/
      │   └── ...
      └── ...
  ```

- Each model should have its own directory
- Multiple runs should be organized in numbered subdirectories
- Maintain consistent naming patterns across all experiments
- Ensure file paths are platform-independent by using `os.path.join()`
- Checkpoints should follow a naming pattern that includes:
  - Model type
  - Epoch number
  - Validation loss
- TensorBoard logs should be organized hierarchically:
  - By model type
  - By run index
  - With consistent naming

This organization makes it easy to locate specific runs, compare models, and visualize results in TensorBoard.
