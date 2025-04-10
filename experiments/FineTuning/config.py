from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Configuration for fine-tuning experiments."""
    # Experiment metadata
    experiment_name: str = "fine_tuning"
    
    # Model parameters
    model_type: str = "tide"  # One of ['tide', 'tsmixer', 'ealstm', 'tft']
    checkpoint_path: str = ""  # Path to the pre-trained model checkpoint
    yaml_path: str = ""  # Path to model hyperparameter YAML file
    
    # Fine-tuning parameters
    lr_factor: float = 10.0  # Factor to reduce learning rate by for fine-tuning
    target_country: str = ""  # Country to fine-tune on
    num_runs: int = 1  # Number of fine-tuning runs to perform
    
    # Data parameters
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    forcing_features: List[str] = field(default_factory=lambda: [
        "snow_depth_water_equivalent_mean",
        "surface_net_solar_radiation_mean",
        "surface_net_thermal_radiation_mean",
        "potential_evaporation_sum_ERA5_LAND",
        "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "total_precipitation_sum",
    ])
    static_features: List[str] = field(default_factory=lambda: [
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
    ])
    
    # Dataset paths
    attribute_dir: str = "/workspace/CARAVANIFY/CA/post_processed/attributes"
    timeseries_dir: str = "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv"
    gauge_id_prefix: str = "CA"
    human_influence_path: str = "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv"
    min_train_years: int = 5
    
    # Training parameters
    batch_size: int = 2048
    num_workers: int = 4
    max_epochs: int = 100
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0001
    
    # Data splitting parameters
    train_prop: float = 0.6
    val_prop: float = 0.2
    test_prop: float = 0.2
    use_proportional_split: bool = True
    
    # Output parameters
    output_dir: str = "experiments/FineTuning/results"
    save_top_k: int = 1
    save_last: bool = True
    
    def validate(self):
        """Validate configuration parameters."""
        # Check required paths
        if not self.checkpoint_path:
            raise ValueError("Checkpoint path must be provided")
        
        if not Path(self.checkpoint_path).exists():
            raise ValueError(f"Checkpoint file not found: {self.checkpoint_path}")
            
        if not self.yaml_path:
            raise ValueError("YAML configuration path must be provided")
            
        if not Path(self.yaml_path).exists():
            raise ValueError(f"YAML file not found: {self.yaml_path}")
        
        # Check model type
        if self.model_type not in ["tide", "tsmixer", "ealstm", "tft"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Check target country if specified
        if self.target_country and self.target_country.lower() not in ["tajikistan", "kyrgyzstan", "combined"]:
            raise ValueError(f"Unsupported country: {self.target_country}")
            
        # Validate fine-tuning parameters
        if self.lr_factor <= 0:
            raise ValueError(f"Learning rate factor must be positive, got {self.lr_factor}")
            
        # Validate split proportions
        if self.use_proportional_split:
            total_prop = self.train_prop + self.val_prop + self.test_prop
            if abs(total_prop - 1.0) > 0.001:  # Allow for small floating point errors
                raise ValueError(f"Split proportions must sum to 1.0, got {total_prop}")
        
        # Validate num_runs
        if self.num_runs < 1:
            raise ValueError(f"Number of runs must be at least 1, got {self.num_runs}")
                
    def get_preprocessing_config(self) -> Dict[str, Dict[str, Any]]:
        """Create standard preprocessing configuration."""
        from sklearn.pipeline import Pipeline
        from src.preprocessing.log_scale import LogTransformer
        from src.preprocessing.standard_scale import StandardScaleTransformer
        from src.preprocessing.grouped import GroupedTransformer
        
        # Use GroupedTransformer for target
        feature_pipeline = Pipeline([("scaler", StandardScaleTransformer())])
        
        target_pipeline = GroupedTransformer(
            Pipeline([("log", LogTransformer()), ("scaler", StandardScaleTransformer())]),
            columns=[self.target],
            group_identifier=self.group_identifier,
            n_jobs=self.num_workers
        )
        
        static_pipeline = Pipeline([("scaler", StandardScaleTransformer())])
        
        return {
            "features": {"pipeline": feature_pipeline},
            "target": {"pipeline": target_pipeline},
            "static_features": {"pipeline": static_pipeline},
        }
        
    def get_checkpoint_dir(self, run_idx: int = None) -> Path:
        """Get directory to save fine-tuned model checkpoints.
        
        Args:
            run_idx: Optional run index for multiple runs
            
        Returns:
            Path to checkpoint directory
        """
        if self.target_country:
            base_dir = Path(self.output_dir) / "checkpoints" / self.target_country.lower() / self.model_type
        else:
            base_dir = Path(self.output_dir) / "checkpoints" / self.model_type
            
        if run_idx is not None:
            return base_dir / f"run_{run_idx}"
        return base_dir
            
    def get_logs_dir(self, run_idx: int = None) -> Path:
        """Get directory to save fine-tuning logs.
        
        Args:
            run_idx: Optional run index for multiple runs
            
        Returns:
            Path to logs directory
        """
        if self.target_country:
            base_dir = Path(self.output_dir) / "logs" / self.target_country.lower() / self.model_type
        else:
            base_dir = Path(self.output_dir) / "logs" / self.model_type
            
        if run_idx is not None:
            return base_dir / f"run_{run_idx}"
        return base_dir
        
    def get_results_dir(self) -> Path:
        """Get directory to save results.
        
        Returns:
            Path to results directory
        """
        if self.target_country:
            return Path(self.output_dir) / "results" / self.target_country.lower()
        else:
            return Path(self.output_dir) / "results"
