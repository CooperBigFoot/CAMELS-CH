"""Configuration for quantile mapping experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from experiments.DataSharing.configs.experiment_config import ExperimentConfig


@dataclass
class QuantileMappingConfig(ExperimentConfig):
    """Configuration for quantile mapping experiment.
    
    This class extends the experiment configuration with parameters specific to
    the quantile mapping experiment, which compares hydrological model performance
    using original versus quantile-mapped meteorological forcing data.
    """
    
    # Quantile mapping specific parameters
    DATA_SOURCE: str = "original"  # Either "original" or "quantile_mapped"
    QUANTILE_MAPPED_FOLDER: Optional[str] = None
    
    # Use reduced forcing features for both sources
    FORCING_FEATURES: List[str] = field(default_factory=lambda: [
        "temperature_2m_mean",
        "total_precipitation_sum"
    ])
    
    # Override output directory
    OUTPUT_DIR: str = "experiments/QuantileMapping/results"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # First run base validation
        super()._validate_config()
        
        # Then validate quantile mapping specific parameters
        if self.DATA_SOURCE.lower() not in ["original", "quantile_mapped"]:
            raise ValueError(f"Unsupported data source: {self.DATA_SOURCE}")
            
        if self.DATA_SOURCE.lower() == "quantile_mapped" and not self.QUANTILE_MAPPED_FOLDER:
            raise ValueError(
                "Quantile mapped folder path must be provided when using quantile_mapped data source"
            )
            
        if not self.FORCING_FEATURES:
            raise ValueError("Forcing features list cannot be empty")
            
    def get_checkpoint_dir(self, data_source: str, model_type: str) -> Path:
        """Get checkpoint directory for a specific data source and model type."""
        return Path(self.OUTPUT_DIR) / "checkpoints" / data_source.lower() / model_type
    
    def get_logs_dir(self, data_source: str, model_type: str) -> Path:
        """Get logs directory for a specific data source and model type."""
        return Path(self.OUTPUT_DIR) / "logs" / data_source.lower() / model_type
    
    def get_results_dir(self, data_source: str, model_type: str) -> Path:
        """Get results directory for a specific data source and model type."""
        return Path(self.OUTPUT_DIR) / "results" / data_source.lower() / model_type
