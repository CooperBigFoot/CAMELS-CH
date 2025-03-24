"""Base configuration class for Central Asian data sharing experiment."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
import random
import numpy as np
import torch


@dataclass
class BaseDataSharingConfig:
    """Base configuration for the Central Asian data sharing experiment.

    This class provides common parameters and functionality for experiment configuration
    across different model types and country scenarios.
    """

    # Data parameters
    GROUP_IDENTIFIER: str = "gauge_id"
    TARGET: str = "streamflow"
    STATIC_FEATURES: List[str] = field(default_factory=list)
    FORCING_FEATURES: List[str] = field(default_factory=list)

    # Common training parameters
    BATCH_SIZE: int = 2048
    MAX_EPOCHS: int = 50
    ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_RUNS: int = 3  # Multiple runs for statistical significance
    MAX_WORKERS: int = min(6, os.cpu_count())

    # Early stopping configuration
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_MIN_DELTA: float = 0.0001

    # Data splitting configuration
    USE_PROPORTIONAL_SPLIT: bool = True
    TRAIN_PROP: float = 0.5
    VAL_PROP: float = 0.25
    TEST_PROP: float = 0.25

    # Country scenarios
    COUNTRIES: List[str] = field(
        default_factory=lambda: ["Tajikistan", "Kyrgyzstan", "Combined"]
    )

    # Dataset paths
    CA_CONFIG: Dict[str, Any] = field(default_factory=dict)

    # Model types to evaluate
    MODEL_TYPES: List[str] = field(
        default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"]
    )

    # YAML paths for hyperparameters - to be set by experiment script
    YAML_PATHS: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived attributes and validate configuration."""
        # Initialize feature lists if not provided
        if not self.STATIC_FEATURES:
            self.STATIC_FEATURES = [
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

        if not self.FORCING_FEATURES:
            self.FORCING_FEATURES = [
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

        # Initialize CA configuration if not provided
        if not self.CA_CONFIG:
            self.CA_CONFIG = {
                "ATTRIBUTE_DIR": "workspace/CARAVANIFY/CA/post_processed/attributes",
                "TIMESERIES_DIR": "workspace/CARAVANIFY/CA/post_processed/timeseries/csv",
                "GAUGE_ID_PREFIX": "CA",
                "MIN_TRAIN_YEARS": 5,
                "HUMAN_INFLUENCE_PATH": "workspace/src/human_influence_index/results/human_influence_classification.csv",
            }

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.BATCH_SIZE <= 0:
            raise ValueError("Batch size must be positive")
        if self.MAX_WORKERS <= 0:
            raise ValueError("Max workers must be positive")

        # Validate split proportions
        if self.USE_PROPORTIONAL_SPLIT:
            total_prop = self.TRAIN_PROP + self.VAL_PROP + self.TEST_PROP
            if (
                not 0.999 <= total_prop <= 1.001
            ):  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
            if any(p <= 0 for p in [self.TRAIN_PROP, self.VAL_PROP, self.TEST_PROP]):
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

        # Use GroupedTransformer for both features and target
        feature_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

        target_pipeline = GroupedTransformer(
            Pipeline(
                [("log", LogTransformer()), ("scaler", StandardScaleTransformer())]
            ),
            columns=[self.TARGET],
            group_identifier=self.GROUP_IDENTIFIER,
            n_jobs=self.MAX_WORKERS,
        )

        static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])

        return {
            "features": {"pipeline": feature_pipeline},
            "target": {"pipeline": target_pipeline},
            "static_features": {"pipeline": static_pipeline},
        }
