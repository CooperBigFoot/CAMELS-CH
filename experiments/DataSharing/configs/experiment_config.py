"""Configuration for Central Asian data sharing experiment."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_config import BaseDataSharingConfig


@dataclass
class ExperimentConfig(BaseDataSharingConfig):
    """Configuration for the Central Asian data sharing experiment.
    
    This class extends the base configuration with experiment-specific
    parameters for evaluating the impact of data sharing between
    Tajikistan and Kyrgyzstan on hydrological model performance.
    """
    
    # Experimental output settings
    OUTPUT_DIR: str = "experiments/DataSharing"
    
    # Checkpoint settings
    SAVE_TOP_K: int = 1  # Number of best models to save
    SAVE_LAST: bool = True  # Whether to save the last model
    
    # Results settings
    SAVE_PREDICTIONS: bool = True  # Whether to save model predictions
    
    def get_checkpoint_dir(self, country: str, model_type: str) -> Path:
        """Get checkpoint directory for a specific country and model type."""
        return Path(self.OUTPUT_DIR) / "checkpoints" / country.lower() / model_type
    
    def get_logs_dir(self, country: str, model_type: str) -> Path:
        """Get logs directory for a specific country and model type."""
        return Path(self.OUTPUT_DIR) / "logs" / country.lower() / model_type
    
    def get_results_dir(self, country: str, model_type: str) -> Path:
        """Get results directory for a specific country and model type."""
        return Path(self.OUTPUT_DIR) / "results" / country.lower() / model_type
