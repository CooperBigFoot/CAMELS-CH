"""Configuration for similarity-based transfer learning experiment."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import random
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Configuration for similarity-based transfer learning experiment.

    This class provides parameters for evaluating knowledge transfer from data-rich
    regions (CH, USA, CL) to data-sparse Central Asian catchments using similarity groups.
    """

    # Experiment metadata
    experiment_name: str = "similarity_based_transfer"

    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
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

    # Group-based configuration
    ca_groups_path: str = "/workspace/CAMELS-CH/classification_results/final_basin_assignments_for_shifted_15_clusters.csv"
    source_clusters_path: str = (
        "/workspace/CAMELS-CH/clustering_results/cluster_assignments_shifted.csv"
    )

    # Define group mappings - can be expanded to include more groups
    group_mappings: Dict[str, Dict] = field(
        default_factory=lambda: {
            "group1": {
                "name": "Group 1 [13, 14]",
                "clusters": [13, 14],
                "ca_group_label": "Group 1 [13, 14]",
            },
        }
    )

    # Target groups to process (subset of group_mappings keys)
    target_groups: List[str] = field(default_factory=lambda: ["group1"])

    # Dataset paths configuration
    ca_config: Dict[str, str] = field(
        default_factory=lambda: {
            "attribute_dir": "/workspace/CARAVANIFY/CA/post_processed/attributes",
            "timeseries_dir": "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv",
            "gauge_id_prefix": "CA",
            "human_influence_path": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }
    )

    ch_config: Dict[str, str] = field(
        default_factory=lambda: {
            "attribute_dir": "/workspace/CARAVANIFY/CH/post_processed/attributes",
            "timeseries_dir": "/workspace/CARAVANIFY/CH/post_processed/timeseries/csv",
            "gauge_id_prefix": "CH",
            "human_influence_path": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }
    )

    usa_config: Dict[str, str] = field(
        default_factory=lambda: {
            "attribute_dir": "/workspace/CARAVANIFY/USA/post_processed/attributes",
            "timeseries_dir": "/workspace/CARAVANIFY/USA/post_processed/timeseries/csv",
            "gauge_id_prefix": "USA",
            "human_influence_path": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }
    )

    cl_config: Dict[str, str] = field(
        default_factory=lambda: {
            "attribute_dir": "/workspace/CARAVANIFY/CL/post_processed/attributes",
            "timeseries_dir": "/workspace/CARAVANIFY/CL/post_processed/timeseries/csv",
            "gauge_id_prefix": "CL",
            "human_influence_path": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
        }
    )

    # Common training parameters
    batch_size: int = 2048
    max_epochs: int = 100
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs: int = 1
    max_workers: int = min(6, os.cpu_count())

    # Early stopping configuration
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001

    # Data splitting configuration
    use_proportional_split: bool = True
    train_prop: float = 0.5
    val_prop: float = 0.25
    test_prop: float = 0.25

    # Model types to evaluate
    model_types: List[str] = field(
        default_factory=lambda: ["tide", "tsmixer", "ealstm", "tft"]
    )

    # Output settings
    output_dir: str = "experiments/SimilarityBasedTransfer/output"
    save_top_k: int = 1
    save_last: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")

        # Validate split proportions
        if self.use_proportional_split:
            total_prop = self.train_prop + self.val_prop + self.test_prop
            if (
                not 0.999 <= total_prop <= 1.001
            ):  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
            if any(p <= 0 for p in [self.train_prop, self.val_prop, self.test_prop]):
                raise ValueError("All split proportions must be positive")

        # Validate group configuration
        if not os.path.exists(self.ca_groups_path):
            raise ValueError(f"CA groups file not found: {self.ca_groups_path}")
        if not os.path.exists(self.source_clusters_path):
            raise ValueError(
                f"Source clusters file not found: {self.source_clusters_path}"
            )

        # Check target groups exist in mappings
        for group in self.target_groups:
            if group not in self.group_mappings:
                raise ValueError(f"Target group '{group}' not found in group_mappings")

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

    def get_checkpoint_dir(self, group: str, model_type: str) -> Path:
        """Get checkpoint directory for a specific group and model type."""
        return Path(self.output_dir) / "checkpoints" / group / model_type

    def get_logs_dir(self, group: str, model_type: str) -> Path:
        """Get logs directory for a specific group and model type."""
        return Path(self.output_dir) / "logs" / group / model_type

    def get_results_dir(self, group: str, model_type: str = "") -> Path:
        """Get results directory for a specific group and model type."""
        if model_type:
            return Path(self.output_dir) / "results" / group / model_type
        else:
            return Path(self.output_dir) / "results" / group
