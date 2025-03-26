"""Configuration for quantile mapping experiment."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import numpy as np
import torch


@dataclass
class QuantileMappingConfig:
    """Configuration for quantile mapping experiment.

    This class provides parameters for evaluating the impact of using quantile-mapped
    meteorological forcing data versus original data on hydrological model performance.
    """

    # Experiment metadata
    experiment_name: str = "quantile_mapping"

    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    
    # Static features used for both data sources
    static_features: List[str] = field(
        default_factory=lambda: [
            "gauge_id",
            "p_mean",
            "area",
            "ele_mt_sav",
            "high_prec_dur",
            "frac_snow",
            "high_prec_freq",
            "slp_dg_sav",
            "cly_pc_sav",
            "aridity_ERA5_LAND",
            "aridity_FAO_PM",
        ]
    )

    # Use reduced forcing features for both sources
    forcing_features: List[str] = field(
        default_factory=lambda: [
            "temperature_2m_mean", 
            "total_precipitation_sum"
        ]
    )

    # Caravan dataset configuration
    ca_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CA/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CA",
            "HUMAN_INFLUENCE_PATH": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv"
        }
    )

    # Data source parameters
    data_sources: List[str] = field(
        default_factory=lambda: ["original", "quantile_mapped"]
    )
    quantile_mapped_folder: Optional[str] = None
    max_workers: int = min(6, os.cpu_count())

    # Training parameters
    batch_size: int = 2048
    max_epochs: int = 100
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model evaluation
    num_runs: int = 1
    model_types: List[str] = field(
        default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"]
    )
    
    # Data splitting configuration
    use_proportional_split: bool = True
    train_prop: float = 0.6
    val_prop: float = 0.2
    test_prop: float = 0.2
    
    # Output settings
    output_dir: str = "experiments/QuantileMapping/output"
    save_top_k: int = 1
    save_last: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate data source parameters
        if any(source not in ["original", "quantile_mapped"] for source in self.data_sources):
            raise ValueError("Data sources must be 'original' or 'quantile_mapped'")
            
        if "quantile_mapped" in self.data_sources and not self.quantile_mapped_folder:
            raise ValueError("quantile_mapped_folder must be provided when using 'quantile_mapped' data source")
            
        # Validate split proportions
        if self.use_proportional_split:
            total_prop = self.train_prop + self.val_prop + self.test_prop
            if not 0.999 <= total_prop <= 1.001:  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
                
        # Validate training parameters
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
            
        # Validate forcing features
        if not self.forcing_features:
            raise ValueError("Forcing features list cannot be empty")

    def set_seed(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_preprocessing_config(self) -> Dict:
        """Create preprocessing configuration."""
        from sklearn.pipeline import Pipeline
        from src.preprocessing.log_scale import LogTransformer
        from src.preprocessing.grouped import GroupedTransformer
        from src.preprocessing.standard_scale import StandardScaleTransformer

        # Use standard scaler for features
        feature_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

        # Use log transform + standard scaler for target with grouped transformer
        target_pipeline = GroupedTransformer(
            Pipeline(
                [("log", LogTransformer()), ("scaler", StandardScaleTransformer())]
            ),
            columns=[self.target],
            group_identifier=self.group_identifier,
            n_jobs=self.max_workers,
        )

        # Use standard scaler for static features
        static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

        return {
            "features": {"pipeline": feature_pipeline},
            "target": {"pipeline": target_pipeline},
            "static_features": {"pipeline": static_pipeline},
        }
        
    def get_checkpoint_dir(self, data_source: str, model_type: str) -> Path:
        """Get checkpoint directory for a specific data source and model type."""
        return Path(self.output_dir) / "checkpoints" / data_source.lower() / model_type
        
    def get_logs_dir(self, data_source: str, model_type: str) -> Path:
        """Get logs directory for a specific data source and model type."""
        return Path(self.output_dir) / "logs" / data_source.lower() / model_type
        
    def get_results_dir(self, data_source: str, model_type: str) -> Path:
        """Get results directory for a specific data source and model type."""
        return Path(self.output_dir) / "results" / data_source.lower() / model_type
