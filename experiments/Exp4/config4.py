import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ExperimentConfig:
    # Base configuration
    GROUP_IDENTIFIER: str = "gauge_id"
    BATCH_SIZE: int = 128
    INPUT_LENGTH: int = 64
    OUTPUT_LENGTH: int = 10
    MAX_EPOCHS: int = 15
    ACCELERATOR: str = "cuda"
    NUM_RUNS: int = 5
    MAX_WORKERS: int = 4  # Limit worker count

    # Learning rates with scheduling
    PRETRAIN_LR: float = 1e-3
    FINETUNE_LR: float = 1e-4
    LR_SCHEDULER_PATIENCE: int = 2
    LR_SCHEDULER_FACTOR: float = 0.5

    # Model configuration
    HIDDEN_SIZE: int = 32

    # Dataset configuration
    TARGET: str = "streamflow"
    STATIC_FEATURES: list = None
    FORCING_FEATURES: list = None

    # Domain specific configs
    CA_CONFIG: Dict[str, Any] = None
    CH_CONFIG: Dict[str, Any] = None

    def __post_init__(self):
        # Initialize feature lists
        self.STATIC_FEATURES = [
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

        # Central Asia configuration
        self.CA_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CA/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CA",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 3,
            "MAX_MISSING_PCT": 10,
        }

        # Switzerland configuration
        self.CH_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CH/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CH/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CH",
            "MIN_TRAIN_YEARS": 20,
            "VAL_YEARS": 10,
            "TEST_YEARS": 0,
            "MAX_MISSING_PCT": 10,
        }

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.BATCH_SIZE <= 0:
            raise ValueError("Batch size must be positive")
        if self.MAX_WORKERS <= 0:
            raise ValueError("Max workers must be positive")
        if self.INPUT_LENGTH <= 0:
            raise ValueError("Input length must be positive")

    def get_run_seed(self, run_index: int) -> int:
        """Generate a unique seed for each experimental run."""
        base_seed = 42  # Fixed base seed for reproducibility
        return base_seed + run_index

    def set_seed(self, run_index: int) -> None:
        """Set all random seeds for reproducibility."""
        seed = self.get_run_seed(run_index)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_preprocessing_config(self, domain: str) -> Dict:
        """Get domain-specific preprocessing configuration."""
        if domain not in ["CH", "CA"]:
            raise ValueError("Domain must be either 'CH' or 'CA'")

        # Create preprocessing pipelines with domain-specific parameters
        from sklearn.pipeline import Pipeline
        from src.preprocessing.transformers import (
            LogTransformer,
            GroupedTransformer,
        )
        from sklearn.preprocessing import StandardScaler

        # Use GroupedTransformer for both features and target
        feature_pipeline = (Pipeline([("scaler", StandardScaler())]),)

        target_pipeline = GroupedTransformer(
            Pipeline([("log", LogTransformer()), ("scaler", StandardScaler())]),
            columns=[self.TARGET],
            group_identifier=self.GROUP_IDENTIFIER,
        )

        static_pipeline = Pipeline([("scaler", StandardScaler())])

        return {
            "features": {"pipeline": feature_pipeline},
            "target": {"pipeline": target_pipeline},
            "static_features": {"pipeline": static_pipeline},
        }


# Create configuration instance
config = ExperimentConfig()
