import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from src.models.TSMixer import TSMixerConfig
from src.models.TSMixerDomainAdaptation import TSMixerDomainAdaptationConfig


@dataclass
class ExperimentConfig:
    """Configuration for domain adaptation experiments."""

    EXPERIMENT_NAME: str = "v3"
    # Base configuration
    GROUP_IDENTIFIER: str = "gauge_id"
    BATCH_SIZE: int = 1024
    INPUT_LENGTH: int = 128
    OUTPUT_LENGTH: int = 10
    MAX_EPOCHS: int = 40
    ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_RUNS: int = 5
    MAX_WORKERS: int = os.cpu_count()

    # Learning rates with scheduling
    FINETUNE_LR: float = 1e-5
    PRETRAIN_LR: float = 3e-4
    LR_SCHEDULER_PATIENCE: int = 3
    LR_SCHEDULER_FACTOR: float = 0.5

    # Model configuration
    HIDDEN_SIZE: int = 32
    DROPOUT: float = 0.3
    NUM_LAYERS: int = 10
    STATIC_EMBEDDING_SIZE: int = 10

    # Dataset configuration
    TARGET: str = "streamflow"
    STATIC_FEATURES: list = None
    FORCING_FEATURES: list = None

    # Domain specific configs
    CA_CONFIG: Dict[str, Any] = None
    CH_CONFIG: Dict[str, Any] = None

    # Adversarial configs
    LAMBDA_ADV: float = 1.0
    DOMAIN_LOSS_WEIGHT: float = 0.3
    DISCRIMINATOR_HIDDEN_DIM: int = 16

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
            "MIN_TRAIN_YEARS": 23,
            "VAL_YEARS": 7,
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

    def get_tsmixer_config(self) -> TSMixerConfig:
        """Generate a TSMixerConfig from experiment parameters."""
        return TSMixerConfig(
            input_len=self.INPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            output_len=self.OUTPUT_LENGTH,
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            hidden_size=self.HIDDEN_SIZE,
            static_embedding_size=self.STATIC_EMBEDDING_SIZE,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            learning_rate=self.PRETRAIN_LR,
            group_identifier=self.GROUP_IDENTIFIER,
            lr_scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=self.LR_SCHEDULER_FACTOR,
        )

    def get_domain_adaptation_config(self) -> TSMixerDomainAdaptationConfig:
        """Generate a TSMixerDomainAdaptationConfig from experiment parameters."""
        return TSMixerDomainAdaptationConfig(
            input_len=self.INPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            output_len=self.OUTPUT_LENGTH,
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            hidden_size=self.HIDDEN_SIZE,
            static_embedding_size=self.STATIC_EMBEDDING_SIZE,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            learning_rate=self.PRETRAIN_LR / 5,  # Reduced learning rate for adaptation
            group_identifier=self.GROUP_IDENTIFIER,
            lr_scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=self.LR_SCHEDULER_FACTOR,
            lambda_adv=self.LAMBDA_ADV,
            domain_loss_weight=self.DOMAIN_LOSS_WEIGHT,
            discriminator_hidden_dim=self.DISCRIMINATOR_HIDDEN_DIM,
        )

    def get_finetune_config(self) -> TSMixerConfig:
        """Generate a TSMixerConfig for fine-tuning."""
        config = self.get_tsmixer_config()
        config.learning_rate = self.FINETUNE_LR
        return config

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
        feature_pipeline = Pipeline([("scaler", StandardScaler())])

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
