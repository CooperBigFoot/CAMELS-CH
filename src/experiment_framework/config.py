"""Base configuration for hydrological forecasting experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml
import numpy as np
import os


@dataclass
class BaseExperimentConfig:
    """Base configuration for all hydrological forecasting experiments.
    
    This class provides common parameters and validation methods that
    all experiment configurations should inherit from.
    
    Attributes:
        exp_name: Name of the experiment
        model_types: List of model types to use
        num_runs: Number of training runs to perform
        max_epochs: Maximum number of epochs for training
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        accelerator: Accelerator to use (auto, cpu, gpu, etc.)
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum change for early stopping
        save_top_k: Number of best models to save
        save_last: Whether to save the last model
        group_identifier: Column name for grouping data (e.g., gauge_id)
        target: Target variable to predict
        forcing_features: List of forcing features
        static_features: List of static features
        use_proportional_split: Whether to use proportional split
        train_prop: Proportion of data for training
        val_prop: Proportion of data for validation
        test_prop: Proportion of data for testing
        min_train_years: Minimum years of data for training
        checkpoint_path: Path to checkpoint for loading
        finetune: Whether to finetune a pretrained model
        lr_factor: Factor to reduce learning rate by when finetuning
        reset_optimizer: Whether to reset optimizer when loading
        output_dir: Directory for outputs
        save_predictions: Whether to save predictions
        max_workers: Maximum number of workers
        yaml_paths: Dictionary mapping model types to YAML paths
        
    Note:
        This class provides backward compatibility for both lowercase (snake_case)
        and uppercase attribute access patterns. The preferred convention is
        lowercase snake_case following Python standards.
    """
    # Mandatory experiment parameters
    exp_name: str = field(default="unnamed_experiment")
    
    # Model parameters
    model_types: List[str] = field(default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"])
    
    # Training parameters
    num_runs: int = 3
    max_epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 0.001
    accelerator: str = "auto"
    
    # Early stopping parameters
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    save_top_k: int = 1
    save_last: bool = True
    
    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    forcing_features: List[str] = field(default_factory=list)
    static_features: List[str] = field(default_factory=list)
    
    # Data splitting parameters
    use_proportional_split: bool = True
    train_prop: float = 0.6
    val_prop: float = 0.2
    test_prop: float = 0.2
    min_train_years: int = 5
    
    # Checkpoint parameters
    checkpoint_path: Optional[str] = None
    finetune: bool = False
    lr_factor: float = 10.0
    reset_optimizer: bool = False
    
    # Output structure
    output_dir: str = "experiments/results"
    save_predictions: bool = True
    
    # Performance parameters
    max_workers: int = field(default_factory=lambda: min(6, os.cpu_count() or 4))
    
    # YAML paths for model hyperparameters
    yaml_paths: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived attributes and set default values if needed."""
        # Initialize default feature lists if not provided
        if not self.forcing_features:
            self.forcing_features = [
                "temperature_2m_mean",
                "total_precipitation_sum",
                "potential_evaporation_sum",
                "snow_depth_water_equivalent_mean"
            ]
            
        if not self.static_features:
            self.static_features = [
                self.group_identifier,
                "area",
                "elevation",
                "slope",
                "p_mean",
                "pet_mean",
                "aridity"
            ]
            
        # Generate YAML paths dictionary if not provided
        if not self.yaml_paths and self.model_types:
            base_dir = "experiments"
            self.yaml_paths = {
                model_type: f"{base_dir}/{self.exp_name}/yaml_files/{model_type}.yaml"
                for model_type in self.model_types
            }
    
    # Property getters for uppercase attribute access (backward compatibility)
    @property
    def GROUP_IDENTIFIER(self) -> str:
        """Uppercase getter for group_identifier."""
        return self.group_identifier
    
    @property
    def TARGET(self) -> str:
        """Uppercase getter for target."""
        return self.target
    
    @property
    def FORCING_FEATURES(self) -> List[str]:
        """Uppercase getter for forcing_features."""
        return self.forcing_features
    
    @property
    def STATIC_FEATURES(self) -> List[str]:
        """Uppercase getter for static_features."""
        return self.static_features
    
    @property
    def USE_PROPORTIONAL_SPLIT(self) -> bool:
        """Uppercase getter for use_proportional_split."""
        return self.use_proportional_split
    
    @property
    def TRAIN_PROP(self) -> float:
        """Uppercase getter for train_prop."""
        return self.train_prop
    
    @property
    def VAL_PROP(self) -> float:
        """Uppercase getter for val_prop."""
        return self.val_prop
    
    @property
    def TEST_PROP(self) -> float:
        """Uppercase getter for test_prop."""
        return self.test_prop
    
    @property
    def MAX_WORKERS(self) -> int:
        """Uppercase getter for max_workers."""
        return self.max_workers
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate experiment name
        if not self.exp_name:
            raise ValueError("Experiment name cannot be empty")
            
        # Validate model types
        valid_models = {"tide", "tsmixer", "ealstm", "tft"}
        invalid_models = set(self.model_types) - valid_models
        if invalid_models:
            raise ValueError(f"Invalid model types: {invalid_models}. Must be one of: {valid_models}")
            
        # Validate numeric parameters
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"Max epochs must be positive, got {self.max_epochs}")
        if self.num_runs <= 0:
            raise ValueError(f"Number of runs must be positive, got {self.num_runs}")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.max_workers <= 0:
            raise ValueError(f"Max workers must be positive, got {self.max_workers}")
            
        # Validate split proportions
        if self.use_proportional_split:
            total_prop = self.train_prop + self.val_prop + self.test_prop
            if not 0.999 <= total_prop <= 1.001:  # Allow for floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
            if any(p <= 0 for p in [self.train_prop, self.val_prop, self.test_prop]):
                raise ValueError("All split proportions must be positive")
                
        # Validate checkpoint parameters
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            raise ValueError(f"Checkpoint file not found: {self.checkpoint_path}")
        if self.finetune and not self.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for fine-tuning")
            
        # Validate YAML paths if provided
        for model_type, yaml_path in self.yaml_paths.items():
            if yaml_path and not Path(yaml_path).exists():
                raise ValueError(f"YAML file for {model_type} not found: {yaml_path}")
                
    def save(self, filepath: str) -> None:
        """Save configuration to a JSON or YAML file.
        
        Args:
            filepath: Path to save the configuration (must end with .json or .yaml)
        """
        # Make sure directory exists
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dictionary
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        # Save based on file extension
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Configuration must be saved as .json or .yaml file")
            
    @classmethod
    def load(cls, filepath: str) -> 'BaseExperimentConfig':
        """Load configuration from a JSON or YAML file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Loaded configuration object
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
        # Load based on file extension
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Configuration must be loaded from .json or .yaml file")
            
        # Create instance from dictionary
        return cls(**config_dict)
    
    def get_preprocessing_config(self) -> Dict[str, Dict[str, Any]]:
        """Create default preprocessing configuration.
        
        Returns:
            Dictionary containing preprocessing pipelines for features,
            target and static features.
        """
        from sklearn.pipeline import Pipeline
        
        # Import custom transformers
        try:
            from src.preprocessing.log_scale import LogTransformer
            from src.preprocessing.grouped import GroupedTransformer
            from src.preprocessing.standard_scale import StandardScaleTransformer
        except ImportError:
            # Fallback to default transformers if custom ones not available
            from sklearn.preprocessing import StandardScaler

            class LogTransformer:
                def __init__(self, epsilon=1e-8):
                    self.epsilon = epsilon
                
                def fit(self, X, y=None):
                    return self
                
                def transform(self, X):
                    return np.log1p(X + self.epsilon)
                
                def inverse_transform(self, X):
                    return np.expm1(X) - self.epsilon
            
            class StandardScaleTransformer(StandardScaler):
                pass
            
            class GroupedTransformer:
                def __init__(self, pipeline, columns, group_identifier, n_jobs=1):
                    self.pipeline = pipeline
                    self.columns = columns
                    self.group_identifier = group_identifier
                    self.n_jobs = n_jobs
                    self.transformers = {}
                
                def fit(self, X, y=None):
                    # This is a simplified implementation
                    return self
                
                def transform(self, X):
                    # This is a simplified implementation
                    return self.pipeline.fit_transform(X)
                
                def inverse_transform(self, X):
                    # This is a simplified implementation
                    return self.pipeline.inverse_transform(X)
        
        # Create feature pipeline
        feature_pipeline = Pipeline([("scaler", StandardScaleTransformer())])
        
        # Create target pipeline with GroupedTransformer if available
        try:
            target_pipeline = GroupedTransformer(
                Pipeline([
                    ("log", LogTransformer()),
                    ("scaler", StandardScaleTransformer())
                ]),
                columns=[self.target],
                group_identifier=self.group_identifier,
                n_jobs=self.max_workers,
            )
        except NameError:
            # Fallback if GroupedTransformer not available
            target_pipeline = Pipeline([
                ("log", LogTransformer()),
                ("scaler", StandardScaleTransformer())
            ])
        
        # Create static feature pipeline
        static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])
        
        return {
            "features": {"pipeline": feature_pipeline},
            "target": {"pipeline": target_pipeline},
            "static_features": {"pipeline": static_pipeline},
        }
    
    def get_output_dirs(self) -> Dict[str, Path]:
        """Get dictionary of output directories for the experiment.
        
        Returns:
            Dictionary containing paths for checkpoints, logs, and results
        """
        base_dir = Path(self.output_dir)
        
        dirs = {
            "base": base_dir,
            "checkpoints": base_dir / "checkpoints",
            "logs": base_dir / "logs",
            "results": base_dir / "results",
        }
        
        return dirs
    
    def get_model_dir(self, model_type: str) -> Dict[str, Path]:
        """Get model-specific directories.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of model-specific paths
        """
        dirs = self.get_output_dirs()
        
        model_dirs = {
            "checkpoints": dirs["checkpoints"] / model_type / self.exp_name,
            "logs": dirs["logs"] / model_type / self.exp_name,
            "results": dirs["results"] / model_type / self.exp_name,
        }
        
        return model_dirs
