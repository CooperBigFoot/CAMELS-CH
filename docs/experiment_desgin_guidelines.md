# Opinionated Guidelines for Hydrological Forecasting Experiments

This document outlines a standardized approach to implementing experiments for hydrological forecasting models. The goal is to provide a flexible yet structured framework that ensures consistency across different experimental setups.

## Directory Structure

Every experiment should follow this standard directory structure:

```
experiments/
  ├── ExperimentName/
  │   ├── experiment.py          # Main entry point with argparse CLI
  │   ├── data_loader.py         # Experiment-specific data loading function
  │   ├── configs/               # Configuration classes
  │   │   ├── __init__.py 
  │   │   └── experiment_config.py
  │   ├── utils.py               # Experiment-specific utilities
  │   ├── yaml_files/            # YAML files for model hyperparameters
  │   │   ├── tide.yaml
  │   │   ├── tsmixer.yaml
  │   │   ├── ealstm.yaml
  │   │   └── tft.yaml
  │   └── README.md              # Experiment documentation
```

## Core Principles

1. **Functional approach** over class-based inheritance
2. **Consistent interfaces** with flexible implementations
3. **Configuration through dataclasses**
4. **Standardized CLI entry points**
5. **Uniform output structure** for logs and checkpoints

## Framework Components

### 1. Data Loading

Each experiment must implement a `load_data` function in `data_loader.py` with a consistent signature:

```python
def load_data(config: Any, **kwargs) -> Dict[str, Any]:
    """
    Load data for an experiment.
    
    Args:
        config: Experiment configuration
        **kwargs: Additional keyword arguments from CLI
        
    Returns:
        Dictionary containing at minimum:
        - 'time_series': pd.DataFrame - Time series data
        - 'static': pd.DataFrame - Static catchment attributes
        - 'basin_count': int - Number of basins
    """
    # Experiment-specific data loading logic
    pass
```

This function is responsible for loading all necessary data for the experiment, with complete flexibility in implementation as long as it returns the expected data structure.

### 2. Configuration

Configurations should use dataclasses, with a hierarchy from base to experiment-specific:

```python
@dataclass
class BaseExperimentConfig:
    """Base configuration for all experiments."""
    # Common parameters across all experiments
    exp_name: str  # Mandatory experiment name
    model_types: List[str] = field(default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"])
    num_runs: int = 3
    max_epochs: int = 100
    batch_size: int = 2048
    
    # Output structure
    output_dir: str = "experiments/results"
    
    # Common training parameters
    early_stopping_patience: int = 5
    save_top_k: int = 1
    save_last: bool = True
    
    # Checkpoint parameters
    checkpoint_path: Optional[str] = None
    finetune: bool = False
    lr_factor: float = 10.0
    reset_optimizer: bool = False

@dataclass
class ExperimentNameConfig(BaseExperimentConfig):
    """Experiment-specific configuration."""
    # Experiment-specific parameters
    pass
```

### 3. Command Line Interface

Every experiment should define its CLI in `experiment.py` with a standard structure:

```python
def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Experiment description")
    
    # Common arguments across all experiments
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (used for logging and checkpoints)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to evaluate"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for each model"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Base output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility"
    )
    
    # Checkpoint loading arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint"
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Reset optimizer when loading from checkpoint"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune pre-trained model with reduced learning rate"
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=10.0,
        help="Factor to reduce learning rate by when fine-tuning"
    )
    
    # YAML paths arguments
    parser.add_argument(
        "--tide-yaml",
        type=str,
        default="experiments/ExperimentName/yaml_files/tide.yaml",
        help="Path to TiDE hyperparameter YAML"
    )
    parser.add_argument(
        "--tsmixer-yaml",
        type=str,
        default="experiments/ExperimentName/yaml_files/tsmixer.yaml",
        help="Path to TSMixer hyperparameter YAML"
    )
    parser.add_argument(
        "--ealstm-yaml",
        type=str,
        default="experiments/ExperimentName/yaml_files/ealstm.yaml",
        help="Path to EALSTM hyperparameter YAML"
    )
    parser.add_argument(
        "--tft-yaml",
        type=str,
        default="experiments/ExperimentName/yaml_files/tft.yaml",
        help="Path to TFT hyperparameter YAML"
    )
    
    # Experiment-specific arguments
    # Add any experiment-specific arguments here
    
    return parser.parse_args()
```

### 4. Model Configuration via YAML

All model hyperparameters should be defined in YAML files and loaded using the `hp_from_yaml.py` function:

```python
from src.model_evaluation.hp_from_yaml import load_model_config

# Load model configurations
model_params = load_model_config(model_type, yaml_path)
```

### 5. Model Creation and Loading

Two key functions are needed for model handling:

```python
def create_model(model_type: str, model_config: Any) -> pl.LightningModule:
    """
    Create a new model instance of the specified type with given configuration.
    
    Args:
        model_type: Type of model to create
        model_config: Configuration for the model
        
    Returns:
        New PyTorch Lightning model instance
    """
    # Create appropriate model type
    pass

def load_pretrained_model(
    checkpoint_path: str, 
    model_type: str, 
    model_config: Any,
    finetune: bool = False,
    lr_factor: float = 10.0,
) -> pl.LightningModule:
    """
    Load a pre-trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model to create
        model_config: Configuration for the model
        finetune: Whether to prepare the model for fine-tuning
        lr_factor: Factor to reduce learning rate by when fine-tuning
        
    Returns:
        Pre-trained PyTorch Lightning model
    """
    # Create new model
    model = create_model(model_type, model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Adjust learning rate for fine-tuning if needed
    if finetune:
        original_lr = model.hparams.learning_rate
        new_lr = original_lr / lr_factor
        
        # Update learning rate in all relevant places
        model.hparams.learning_rate = new_lr
        if hasattr(model, "config"):
            model.config.learning_rate = new_lr
        
        # Mark as fine-tuned
        model.is_fine_tuned = True
        model.original_lr = original_lr
        model.fine_tuned_lr = new_lr
    
    return model
```

### 6. Standardized Training Function

A common training function should be used across experiments, supporting both new training and fine-tuning:

```python
def train_model(
    model_type: str,
    model_config: Any,
    data_module: Any,
    exp_name: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    finetune: bool = False,
    lr_factor: float = 10.0,
    reset_optimizer: bool = False,
    num_runs: int = 3,
    early_stopping_patience: int = 5,
    save_top_k: int = 1,
    save_last: bool = True,
) -> Dict[str, Any]:
    """
    Train a model and save checkpoints.
    
    Args:
        model_type: Type of model to train
        model_config: Configuration for the model
        data_module: DataModule for the model
        exp_name: Experiment name for output
        output_dir: Base output directory
        checkpoint_path: Optional path to pre-trained model checkpoint
        finetune: Whether to fine-tune the model with reduced learning rate
        lr_factor: Factor to reduce learning rate by when fine-tuning
        reset_optimizer: Whether to reset optimizer when loading from checkpoint
        num_runs: Number of training runs with different seeds
        early_stopping_patience: Patience for early stopping
        save_top_k: Number of best models to save
        save_last: Whether to save the last model checkpoint
        
    Returns:
        Dictionary with training results
    """
    # Create model(s) and train
    pass
```

### 7. Output Structure

Experiments must always maintain this output structure:

```
output_dir/
  ├── checkpoints/
  │   ├── tide/
  │   │   └── {exp-name}/
  │   ├── tsmixer/
  │   │   └── {exp-name}/
  │   ├── ealstm/
  │   │   └── {exp-name}/
  │   └── tft/
  │       └── {exp-name}/
  └── logs/
      ├── tide/
      │   └── {exp-name}/
      ├── tsmixer/
      │   └── {exp-name}/
      ├── ealstm/
      │   └── {exp-name}/
      └── tft/
          └── {exp-name}/
```

### 8. Directory Setup Utility

```python
def setup_dirs(output_dir: str, exp_name: str) -> Dict[str, Path]:
    """
    Create and return necessary directories for experiment outputs.
    
    Args:
        output_dir: Base output directory path
        exp_name: Experiment name for subdirectories
        
    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path(output_dir)
    model_types = ["tide", "tsmixer", "ealstm", "tft"]
    
    # Define directory structure
    dirs = {
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
    }
    
    # Create directories for each model type and experiment
    for model_type in model_types:
        (dirs["checkpoints"] / model_type / exp_name).mkdir(parents=True, exist_ok=True)
        (dirs["logs"] / model_type / exp_name).mkdir(parents=True, exist_ok=True)
    
    return dirs
```

### 9. Experiment Main Function Template

Every experiment's `main()` function should follow this pattern:

```python
def main():
    """Run the experiment."""
    # Parse CLI arguments
    args = parse_args()
    
    # Set global seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Create experiment configuration
    config = ExperimentNameConfig(
        exp_name=args.exp_name,
        model_types=args.models,
        num_runs=args.num_runs,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        finetune=args.finetune,
        lr_factor=args.lr_factor,
        reset_optimizer=args.reset_optimizer,
        # Experiment-specific parameters
        ...
    )
    
    # Setup directories
    setup_dirs(config.output_dir, config.exp_name)
    
    # Load data
    data = load_data(config, **vars(args))
    
    # Create model configurations and datamodules
    model_configs, data_modules = setup_models_and_data(
        data=data,
        config=config,
        yaml_paths={
            "tide": args.tide_yaml,
            "tsmixer": args.tsmixer_yaml,
            "ealstm": args.ealstm_yaml,
            "tft": args.tft_yaml,
        }
    )
    
    # Run experiment for each model type
    results = {}
    for model_type in args.models:
        if model_type not in model_configs:
            print(f"Skipping {model_type} (no configuration found)")
            continue
            
        print(f"\nTraining {model_type.upper()} model")
        
        model_config = model_configs[model_type]
        data_module = data_modules[model_type]
        
        # Train model
        model_results = train_model(
            model_type=model_type,
            model_config=model_config,
            data_module=data_module,
            exp_name=args.exp_name,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            finetune=args.finetune,
            lr_factor=args.lr_factor,
            reset_optimizer=args.reset_optimizer,
            num_runs=args.num_runs,
            early_stopping_patience=config.early_stopping_patience,
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        )
        
        results[model_type] = model_results
    
    # Save experiment results
    save_experiment_results(results, args.output_dir, args.exp_name)
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()
```

## Framework Implementation Structure

The core framework utilities should be organized as follows:

```
src/experiment_framework/
  ├── __init__.py                # Exports key functions and version information
  ├── config.py                  # Base dataclass configuration with validation methods
  │   └── BaseExperimentConfig   # Base configuration class that all experiments extend
  ├── utils.py                   # Common utility functions
  │   ├── setup_dirs()           # Create output directories 
  │   ├── setup_seeds()          # Set random seeds for reproducibility
  │   ├── setup_logging()        # Configure logging for experiments
  │   ├── train_model()          # Standard training function with multi-run support
  │   ├── save_experiment_results() # Save experiment results in CSV and JSON formats
  │   └── create_experiment_parser() # Create standard CLI parser
  ├── data_utils.py              # Data handling utilities
  │   ├── create_datamodule()    # Create HydroDataModule from data
  │   ├── setup_preprocessing()  # Create preprocessing pipelines
  │   ├── validate_data()        # Validate and clean data before using in experiments
  │   └── load_country_data()    # Load country-specific data (optional utility)
  └── model_utils.py             # Model handling utilities
      ├── create_model()         # Create new model from config
      ├── load_pretrained_model() # Load model from checkpoint with fine-tuning support
      ├── load_model_configs_from_yaml() # Load model configs from YAML files
      └── load_model_datamodules() # Create DataModules for each model type
```

Each module has a clear responsibility:

1. **config.py**: Defines the base configuration dataclass with validation, saving, and loading methods
2. **utils.py**: Provides experiment workflow functions and utilities for directory setup, training, and results handling
3. **data_utils.py**: Contains utilities for data processing, validation, and DataModule creation
4. **model_utils.py**: Handles model creation, checkpoint loading, and configuration loading from YAML files

When implementing a new experiment, you should:

1. Create a new experiment directory following the standardized structure
2. Implement a custom configuration class extending `BaseExperimentConfig`
3. Create a data loading function that returns data in the expected format
4. Use the framework's standard utilities for training and evaluation

## Framework Requirements

1. The `exp-name` CLI argument should be mandatory
2. YAML files must be used for model hyperparameters
3. All experiments must have consistent output structures
4. The data loader implementation is fully flexible but must return a standard data structure
5. Support for both new training and continuing from checkpoints (with fine-tuning option)

## Best Practices

1. Keep experiment-specific logic in `utils.py`
2. Document experiment parameters and workflow in `README.md`
3. Add comments to clarify experiment-specific adaptations
4. Use clear, descriptive naming for experiment parameters
5. Handle errors gracefully and provide informative messages
6. Validate configuration parameters before running experiments
7. When designing fine-tuning experiments, ensure learning rate adjustment is appropriate
8. Always include seeds for reproducibility
