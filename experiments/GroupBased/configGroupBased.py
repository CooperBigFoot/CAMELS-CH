import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
import pandas as pd
from src.models.TSMixer import TSMixerConfig


@dataclass
class ExperimentConfig:
    """Configuration for the merged dataset experiment with group-based training."""

    EXPERIMENT_NAME: str = "group_based_transfer"
    # Base configuration
    GROUP_IDENTIFIER: str = "gauge_id"
    BATCH_SIZE: int = 2048  # Updated batch size
    MAX_EPOCHS: int = 55
    ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_RUNS: int = 1
    MAX_WORKERS: int = os.cpu_count()
    
    # Path to pretrained checkpoint for fine-tuning
    PRETRAINED_CHECKPOINT_PATH: Optional[str] = None
    
    # Input/Output lengths used by both benchmark and challenger models
    INPUT_LENGTH: int = 256  # Default to challenger input length
    OUTPUT_LENGTH: int = 10
    
    # Data splitting configuration
    USE_PROPORTIONAL_SPLIT: bool = True  # Enable proportional splitting
    TRAIN_PROP: float = 0.5             # 50% of data for training
    VAL_PROP: float = 0.25              # 25% of data for validation
    TEST_PROP: float = 0.25             # 25% of data for testing

    # Group-based training configuration
    GROUP_TRAINING_ENABLED: bool = True
    CA_GROUPS_PATH: str = "/workspace/CAMELS-CH/classification_results/final_basin_assignments_for_shifted_15_clusters.csv"
    SOURCE_CLUSTERS_PATH: str = (
        "/workspace/CAMELS-CH/clustering_results/cluster_assignments_shifted.csv"
    )
    GROUP_MAPPINGS: Dict[str, Dict] = None  # Will be initialized in __post_init__

    # Learning rates with scheduling
    LR_SCHEDULER_PATIENCE: int = 5
    LR_SCHEDULER_FACTOR: float = 0.5
    
    # Future forcing configuration
    FUSION_METHOD: str = "add"  # Options: "add" or "concat"

    # Benchmark model configuration
    BENCHMARK_INPUT_LENGTH: int = 33
    BENCHMARK_OUTPUT_LENGTH: int = 10
    BENCHMARK_HIDDEN_SIZE: int = 127
    BENCHMARK_DROPOUT: float = 0.2
    BENCHMARK_NUM_LAYERS: int = 15
    BENCHMARK_STATIC_EMBEDDING_SIZE: int = 16
    BENCHMARK_LEARNING_RATE: float = 6.5e-5
    BENCHMARK_FUSION_METHOD: str = "add"  # Options: "add" or "concat"

    # Challenger model configuration
    CHALLENGER_INPUT_LENGTH: int = 256
    CHALLENGER_OUTPUT_LENGTH: int = 10
    CHALLENGER_HIDDEN_SIZE: int = 128
    CHALLENGER_DROPOUT: float = 0.3
    CHALLENGER_NUM_LAYERS: int = 13
    CHALLENGER_STATIC_EMBEDDING_SIZE: int = 9
    CHALLENGER_LEARNING_RATE: float = 2e-5
    CHALLENGER_FUSION_METHOD: str = "add"  # Options: "add" or "concat"

    # Dataset configuration
    TARGET: str = "streamflow"
    STATIC_FEATURES: List[str] = None  # Will be initialized in __post_init__
    FORCING_FEATURES: List[str] = None  # Will be initialized in __post_init__

    # Domain specific configs
    CA_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__
    CH_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__
    USA_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__
    CL_CONFIG: Dict[str, Any] = None  # Will be initialized in __post_init__

    # Cache for loaded training data
    TRAINING_DATA_CACHE: Dict = None

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
            "TEST_YEARS": 5,
            "MAX_MISSING_PCT": 15,
        }

        # Switzerland configuration
        self.CH_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CH/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CH/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CH",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 5,
            "MAX_MISSING_PCT": 15,
        }

        # USA configuration
        self.USA_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/USA/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/USA/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "USA",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 5,
            "MAX_MISSING_PCT": 15,
        }

        # Chile configuration
        self.CL_CONFIG = {
            "ATTRIBUTE_DIR": "/workspace/CARAVANIFY/CL/post_processed/attributes",
            "TIMESERIES_DIR": "/workspace/CARAVANIFY/CL/post_processed/timeseries/csv",
            "GAUGE_ID_PREFIX": "CL",
            "MIN_TRAIN_YEARS": 8,
            "VAL_YEARS": 2,
            "TEST_YEARS": 5,
            "MAX_MISSING_PCT": 15,
        }

        # Define group mappings
        self.GROUP_MAPPINGS = {
            "group1": {
                "name": "Group 1 [13, 14]",
                "clusters": [13, 14],
                "ca_group_label": "Group 1 [13, 14]",
            },
        }

        # Initialize training data cache
        self.TRAINING_DATA_CACHE = {}

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

        if self.GROUP_TRAINING_ENABLED:
            if not os.path.exists(self.CA_GROUPS_PATH):
                raise ValueError(f"CA groups file not found: {self.CA_GROUPS_PATH}")
            if not os.path.exists(self.SOURCE_CLUSTERS_PATH):
                raise ValueError(
                    f"Source clusters file not found: {self.SOURCE_CLUSTERS_PATH}"
                )
                
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
            fusion_method=self.BENCHMARK_FUSION_METHOD,  # Add fusion method
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
            fusion_method=self.CHALLENGER_FUSION_METHOD,  # Add fusion method
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

    def extract_source_basins_for_training(self) -> Dict:
        """
        Extract source basins corresponding to CA target basins grouped by their major groups.

        Returns:
            Dictionary with training data structure organized by target groups
        """
        # Check if already loaded (caching)
        if "training_data" in self.TRAINING_DATA_CACHE:
            return self.TRAINING_DATA_CACHE["training_data"]

        # Load the data
        ca_groups = pd.read_csv(self.CA_GROUPS_PATH)
        source_clusters = pd.read_csv(self.SOURCE_CLUSTERS_PATH)

        training_data = {}

        # Process each group
        for group_key, group_info in self.GROUP_MAPPINGS.items():
            # Get CA basins for this group
            ca_group_basins = ca_groups[
                ca_groups["major_group"] == group_info["ca_group_label"]
            ]["gauge_id"].tolist()

            # Get source basins for this group by cluster
            source_group_basins = source_clusters[
                source_clusters["cluster"].isin(group_info["clusters"])
            ]["gauge_id"].tolist()

            # Split source basins by country
            country_basins = {
                "CH": [
                    basin for basin in source_group_basins if basin.startswith("CH_")
                ],
                "USA": [
                    basin for basin in source_group_basins if basin.startswith("USA_")
                ],
                "CL": [
                    basin for basin in source_group_basins if basin.startswith("CL_")
                ],
            }

            # Store in training data structure
            training_data[group_key] = {
                "target": ca_group_basins,
                "source": country_basins,
            }

            # Print summary statistics
            print(f"{group_key} - CA target basins: {len(ca_group_basins)}")
            for country, basins in country_basins.items():
                print(f"{group_key} - {country} source basins: {len(basins)}")
            print()

        # Cache for future use
        self.TRAINING_DATA_CACHE["training_data"] = training_data

        return training_data

    def load_data_for_group(self, group_key: str) -> Dict:
        """
        Load all data for a specific group using Caravanify.

        Args:
            group_key: Key of the group to load data for (e.g., 'group1')

        Returns:
            Dictionary with loaded time series and static data
        """
        from src.data_models.caravanify import Caravanify, CaravanifyConfig

        # Check cache first
        cache_key = f"data_{group_key}"
        if cache_key in self.TRAINING_DATA_CACHE:
            return self.TRAINING_DATA_CACHE[cache_key]

        # Get training data mapping
        training_data = self.extract_source_basins_for_training()

        print(f"Loading data for {group_key}...")

        # Get relevant basin IDs
        ca_basins = training_data[group_key]["target"]

        # Dictionary to store data from each country
        data = {
            "ca_ts_data": None,
            "ca_static_data": None,
            "source_ts_data": [],
            "source_static_data": [],
        }

        # Load CA data
        ca_config = CaravanifyConfig(
            attributes_dir=self.CA_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.CA_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.CA_CONFIG["GAUGE_ID_PREFIX"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        ca_caravan = Caravanify(ca_config)
        ca_caravan.load_stations(ca_basins)
        data["ca_ts_data"] = ca_caravan.get_time_series()
        data["ca_static_data"] = ca_caravan.get_static_attributes()

        # Load source data for each country
        country_configs = {
            "CH": self.CH_CONFIG,
            "USA": self.USA_CONFIG,
            "CL": self.CL_CONFIG,
        }

        for country, basins in training_data[group_key]["source"].items():
            if not basins:  # Skip if no basins for this country
                continue

            country_cfg = country_configs[country]
            source_config = CaravanifyConfig(
                attributes_dir=country_cfg["ATTRIBUTE_DIR"],
                timeseries_dir=country_cfg["TIMESERIES_DIR"],
                gauge_id_prefix=country_cfg["GAUGE_ID_PREFIX"],
                use_hydroatlas_attributes=True,
                use_caravan_attributes=True,
                use_other_attributes=True,
            )

            source_caravan = Caravanify(source_config)
            source_caravan.load_stations(basins)

            data["source_ts_data"].append(source_caravan.get_time_series())
            data["source_static_data"].append(source_caravan.get_static_attributes())

        # Cache the data
        self.TRAINING_DATA_CACHE[cache_key] = data

        return data