import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Any, List
import os
from src.models.TSMixer import TSMixerConfig


@dataclass
class ExperimentConfig:
    """Configuration for the merged dataset experiment."""

    EXPERIMENT_NAME: str = "merged_dataset"
    # Base configuration
    GROUP_IDENTIFIER: str = "gauge_id"
    BATCH_SIZE: int = 2048  # Updated batch size
    MAX_EPOCHS: int = 100
    ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_RUNS: int = 1
    MAX_WORKERS: int = min(6, os.cpu_count())

    # Data splitting configuration
    USE_PROPORTIONAL_SPLIT: bool = True  # Enable proportional splitting
    TRAIN_PROP: float = 0.5             # 50% of data for training
    VAL_PROP: float = 0.25              # 25% of data for validation
    TEST_PROP: float = 0.25             # 25% of data for testing

    # Learning rates with scheduling
    LR_SCHEDULER_PATIENCE: int = 5
    LR_SCHEDULER_FACTOR: float = 0.5

    # Future forcing configuration
    FUSION_METHOD: str = "add"  # Options: "add" or "concat"

    # Benchmark model configuration
    BENCHMARK_INPUT_LENGTH: int = 256
    BENCHMARK_OUTPUT_LENGTH: int = 10
    BENCHMARK_HIDDEN_SIZE: int = 64
    BENCHMARK_DROPOUT: float = 0.4
    BENCHMARK_NUM_LAYERS: int = 2
    BENCHMARK_STATIC_EMBEDDING_SIZE: int = 20
    BENCHMARK_LEARNING_RATE: float = 0.00085

    # Challenger model configuration
    CHALLENGER_INPUT_LENGTH: int = 256
    CHALLENGER_OUTPUT_LENGTH: int = 10
    CHALLENGER_HIDDEN_SIZE: int = 128
    CHALLENGER_DROPOUT: float = 0.3
    CHALLENGER_NUM_LAYERS: int = 13
    CHALLENGER_STATIC_EMBEDDING_SIZE: int = 9
    CHALLENGER_LEARNING_RATE: float = 2e-5

    # Dataset configuration
    TARGET: str = "streamflow"
    STATIC_FEATURES: List[str] = None  # Will be initialized in __post_init__
    FORCING_FEATURES: List[str] = None  # Will be initialized in __post_init__

    # Domain specific configs
    CA_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__
    CH_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__
    USA_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__

    # Visualization configs
    VIZ_DPI: int = 300  # DPI for saving visualizations

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
            "HUMAN_INFLUENCE_PATH": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
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
            "HUMAN_INFLUENCE_PATH": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }

        # USA configuration
        self.USA_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/USA/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/USA/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "USA",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 3,
            "MAX_MISSING_PCT": 10,
            "HUMAN_INFLUENCE_PATH": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }

        # Chile configuration
        self.CL_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CL/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CL/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CL",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 3,
            "MAX_MISSING_PCT": 10,
            "HUMAN_INFLUENCE_PATH": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.BATCH_SIZE <= 0:
            raise ValueError("Batch size must be positive")
        if self.MAX_WORKERS <= 0:
            raise ValueError("Max workers must be positive")
        if self.BENCHMARK_INPUT_LENGTH <= 0:
            raise ValueError("Benchmark input length must be positive")
        if self.CHALLENGER_INPUT_LENGTH <= 0:
            raise ValueError("Challenger input length must be positive")
            
        # Validate split proportions
        if self.USE_PROPORTIONAL_SPLIT:
            total_prop = self.TRAIN_PROP + self.VAL_PROP + self.TEST_PROP
            if not 0.999 <= total_prop <= 1.001:  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
            if any(p <= 0 for p in [self.TRAIN_PROP, self.VAL_PROP, self.TEST_PROP]):
                raise ValueError("All split proportions must be positive")

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

    def get_benchmark_tsmixer_config(self) -> TSMixerConfig:
        """Generate a TSMixerConfig for benchmark with specific hyperparameters."""
        # Calculate future_input_size as the number of forcing features (excluding target)
        future_input_size = len(self.FORCING_FEATURES)
        
        return TSMixerConfig(
            input_len=self.BENCHMARK_INPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            output_len=self.BENCHMARK_OUTPUT_LENGTH,
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            future_input_size=future_input_size,  # Add future forcing size
            hidden_size=self.BENCHMARK_HIDDEN_SIZE,
            static_embedding_size=self.BENCHMARK_STATIC_EMBEDDING_SIZE,
            num_layers=self.BENCHMARK_NUM_LAYERS,
            dropout=self.BENCHMARK_DROPOUT,
            learning_rate=self.BENCHMARK_LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            lr_scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=self.LR_SCHEDULER_FACTOR,
            fusion_method=self.FUSION_METHOD,  # Add fusion method
        )

    def get_challenger_tsmixer_config(self) -> TSMixerConfig:
        """Generate a TSMixerConfig for challenger with specific hyperparameters."""
        # Calculate future_input_size as the number of forcing features (excluding target)
        future_input_size = len(self.FORCING_FEATURES)
        
        return TSMixerConfig(
            input_len=self.CHALLENGER_INPUT_LENGTH,
            input_size=len(self.FORCING_FEATURES) + 1,  # +1 for target
            output_len=self.CHALLENGER_OUTPUT_LENGTH,
            static_size=len(self.STATIC_FEATURES) - 1,  # -1 for gauge_id
            future_input_size=future_input_size,  # Add future forcing size
            hidden_size=self.CHALLENGER_HIDDEN_SIZE,
            static_embedding_size=self.CHALLENGER_STATIC_EMBEDDING_SIZE,
            num_layers=self.CHALLENGER_NUM_LAYERS,
            dropout=self.CHALLENGER_DROPOUT,
            learning_rate=self.CHALLENGER_LEARNING_RATE,
            group_identifier=self.GROUP_IDENTIFIER,
            lr_scheduler_patience=self.LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=self.LR_SCHEDULER_FACTOR,
            fusion_method=self.FUSION_METHOD,  # Add fusion method
        )

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