"""Configuration for hyperparameter tuning experiment."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import random
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Configuration for the hyperparameter tuning experiment.
    
    This class provides parameters for optimizing hyperparameters of 
    hydrological forecasting models with support for country-specific tuning.
    """

    # Experiment metadata
    experiment_name: str = "hyperparameter_tuning"

    # Model types to evaluate
    model_types: List[str] = field(
        default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"]
    )

    # Country scenarios
    countries: List[str] = field(
        default_factory=lambda: ["Tajikistan", "Kyrgyzstan", "Combined"]
    )

    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    static_features: List[str] = field(
        default_factory=lambda: [
            "gauge_id",
            "country",  # Added country for filtering
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

    forcing_features: List[str] = field(
        default_factory=lambda: [
            "snow_depth_water_equivalent_mean",
            "surface_net_solar_radiation_mean",
            "surface_net_thermal_radiation_mean",
            "potential_evaporation_sum_ERA5_LAND",
            "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "total_precipitation_sum",
        ]
    )

    # Central Asian dataset paths
    ca_attribute_dir: str = "/workspace/CARAVANIFY/CA/post_processed/attributes"
    ca_timeseries_dir: str = "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv"
    ca_gauge_id_prefix: str = "CA"
    ca_min_train_years: int = 5
    ca_human_influence_path: str = "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv"

    # Common training parameters
    batch_size: int = 2048
    max_epochs: int = 50
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_workers: int = min(6, os.cpu_count())

    # Input/output parameters
    input_length: int = 100  # Default, will be overridden by tuning
    output_length: int = 10  # Forecast horizon

    # Data splitting configuration
    use_proportional_split: bool = True
    train_prop: float = 0.5
    val_prop: float = 0.25
    test_prop: float = 0.25

    # Optuna tuning configuration
    n_trials: int = 20
    timeout: Optional[int] = None  # No timeout by default
    early_stopping_patience: int = 5  # For model training early stopping
    early_stopping_min_delta: float = 0.0001
    
    # Output settings
    output_dir: str = "experiments/HyperparameterTune/output"
    save_top_k: int = 1
    save_last: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        if self.input_length <= 0:
            raise ValueError("Input length must be positive")
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")

        # Validate split proportions
        if self.use_proportional_split:
            total_prop = self.train_prop + self.val_prop + self.test_prop
            if (
                not 0.999 <= total_prop <= 1.001
            ):  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
            if any(p <= 0 for p in [self.train_prop, self.val_prop, self.test_prop]):
                raise ValueError("All split proportions must be positive")

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

    def get_checkpoint_dir(self, country: str, model_type: str) -> Path:
        """Get checkpoint directory for a specific country and model type."""
        return Path(self.output_dir) / "checkpoints" / country.lower() / model_type

    def get_logs_dir(self, country: str, model_type: str) -> Path:
        """Get logs directory for a specific country and model type."""
        return Path(self.output_dir) / "logs" / country.lower() / model_type

    def get_results_dir(self, country: str, model_type: str) -> Path:
        """Get results directory for a specific country and model type."""
        return Path(self.output_dir) / "results" / country.lower() / model_type
