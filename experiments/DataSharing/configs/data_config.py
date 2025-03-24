"""Data configuration for the data sharing experiment."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from sklearn.pipeline import Pipeline
from src.preprocessing.grouped import GroupedTransformer
from src.preprocessing.log_scale import LogTransformer
from src.preprocessing.standard_scale import StandardScaleTransformer
from src.model_evaluation.hp_from_yaml import hp_from_yaml


@dataclass
class DataSharingDataConfig:
    """Configuration for data preparation in the data sharing experiment."""
    
    # Paths to data
    ca_attribute_dir: str = "/workspace/CARAVANIFY/CA/post_processed/attributes"
    ca_timeseries_dir: str = "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv"
    ca_shapefile_dir: str = "/workspace/CARAVANIFY/CA/post_processed/shapefiles"
    
    # Hyperparameter files path
    hyperparams_dir: str = field(default_factory=lambda: os.path.join(
        Path(__file__).resolve().parents[1], "hyperparams"
    ))
    
    # Data parameters
    gauge_id_prefix: str = "CA"
    group_identifier: str = "gauge_id"
    target: str = "streamflow"
    
    # Feature selection
    forcing_features: List[str] = field(default_factory=lambda: [
        "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
        "temperature_2m_mean",
        "total_precipitation_sum",
        "snow_depth_water_equivalent_mean"
    ])
    
    static_features: List[str] = field(default_factory=lambda: [
        "gauge_id",  # Needed for grouping
        "p_mean",
        "area",
        "ele_mt_sav",
        "high_prec_dur",
        "frac_snow",
        "high_prec_freq",
        "slp_dg_sav", 
        "cly_pc_sav",
        "aridity_ERA5_LAND",
        "aridity_FAO_PM"
    ])
    
    # DataModule parameters
    input_length: int = 60  # Default, will be overridden by model-specific values
    output_length: int = 10  # Same output length for all models
    batch_size: int = 2048   # Same batch size for all models
    num_workers: int = 4
    min_train_years: int = 5
    
    # Time splits
    use_proportional_split: bool = True
    train_prop: float = 0.5   # Updated to 50%
    val_prop: float = 0.25    # Updated to 25%
    test_prop: float = 0.25   # Updated to 25%
    
    # Filtering options
    max_missing_pct: float = 10.0  # Updated to 10%
    
    def get_model_hp_path(self, model_type: str) -> str:
        """Get path to model hyperparameters YAML file.
        
        Args:
            model_type: Type of model ('tide', 'tsmixer', 'tft', 'ealstm')
            
        Returns:
            Path to hyperparameters YAML file
        """
        return os.path.join(self.hyperparams_dir, f"{model_type}_best.yaml")
    
    def get_model_hyperparams(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for specific model type from YAML.
        
        Args:
            model_type: Type of model ('tide', 'tsmixer', 'tft', 'ealstm')
            
        Returns:
            Dictionary of hyperparameters from YAML file
        """
        hp_path = self.get_model_hp_path(model_type)
        return hp_from_yaml(model_type, hp_path)
    
    def get_caravan_config(self) -> CaravanifyConfig:
        """Create a CaravanifyConfig for loading CARAVAN data."""
        return CaravanifyConfig(
            attributes_dir=self.ca_attribute_dir,
            timeseries_dir=self.ca_timeseries_dir,
            shapefile_dir=self.ca_shapefile_dir,
            gauge_id_prefix=self.gauge_id_prefix,
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )
    
    def get_preprocessing_config(self) -> Dict:
        """Create preprocessing configuration for the data module."""
        feature_cols = self.forcing_features
        static_feature_cols = [c for c in self.static_features if c != self.group_identifier]
        target_cols = [self.target]
        
        # Feature pipeline: Standard scale
        feature_pipeline = Pipeline([
            ("scaler", StandardScaleTransformer(columns=feature_cols))
        ])
        
        # Target pipeline: Log transform + Standard scale, grouped by basin
        target_pipeline = GroupedTransformer(
            Pipeline([
                ("log", LogTransformer(columns=target_cols)),
                ("scaler", StandardScaleTransformer(columns=target_cols))
            ]),
            columns=target_cols,
            group_identifier=self.group_identifier,
            n_jobs=-1,
        )
        
        # Static feature pipeline: Standard scale
        static_pipeline = Pipeline([
            ("scaler", StandardScaleTransformer(columns=static_feature_cols))
        ])
        
        return {
            "features": {"pipeline": feature_pipeline, "columns": feature_cols},
            "target": {"pipeline": target_pipeline, "columns": target_cols},
            "static_features": {"pipeline": static_pipeline, "columns": static_feature_cols},
        }
    
    def load_data(self) -> Tuple[Caravanify, List[str], List[str]]:
        """Load CARAVAN data and get basin IDs for different countries.
        
        Returns:
            Tuple containing:
                caravan: Loaded Caravanify instance
                kgz_ids: List of Kyrgyzstan basin IDs
                tjk_ids: List of Tajikistan basin IDs
        """
        # Initialize Caravanify
        config = self.get_caravan_config()
        caravan = Caravanify(config)
        
        # Get all basin IDs without human influence filtering
        all_ids = caravan.get_all_gauge_ids()
        
        print(f"Found {len(all_ids)} total CA basins")
        
        # Load stations
        caravan.load_stations(all_ids)
        
        # Get static data to filter by country
        static_data = caravan.get_static_attributes()
        
        # Filter by country
        kgz_ids = static_data[static_data["country"] == "Kyrgyzstan"][self.group_identifier].unique().tolist()
        tjk_ids = static_data[static_data["country"] == "Tajikistan"][self.group_identifier].unique().tolist()
        
        print(f"Number of gauges in Kyrgyzstan: {len(kgz_ids)}")
        print(f"Number of gauges in Tajikistan: {len(tjk_ids)}")
        
        return caravan, kgz_ids, tjk_ids
            
    def prepare_model_datamodule(
        self, 
        caravan: Caravanify, 
        basin_ids: List[str],
        model_type: str,
        domain_id: str = "CA"
    ) -> HydroDataModule:
        """Create a model-specific HydroDataModule.
        
        Args:
            caravan: Loaded Caravanify instance
            basin_ids: List of basin IDs to include
            model_type: Type of model ('tide', 'tsmixer', 'tft', 'ealstm')
            domain_id: Domain identifier for the dataset
            
        Returns:
            Configured HydroDataModule for the specific model type
        """
        # Get model hyperparameters from YAML
        model_hp = self.get_model_hyperparams(model_type)
        
        # Extract input_len from hyperparameters, but use common output_len
        input_len = model_hp.get("input_len", self.input_length)
        
        # Load only the specified basins
        caravan.load_stations(basin_ids)
        
        # Get time series and static data
        ts_data = caravan.get_time_series()
        static_data = caravan.get_static_attributes()
        
        # Filter columns
        ts_columns = self.forcing_features + [self.target, "date", self.group_identifier]
        ts_data = ts_data[ts_columns]
        
        static_columns = self.static_features
        static_data = static_data[static_columns]
        
        # Create datamodule with model-specific parameters
        datamodule = HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.group_identifier,
            preprocessing_config=self.get_preprocessing_config(),
            batch_size=self.batch_size,  # Use the common batch size
            input_length=input_len,
            output_length=self.output_length,  # Use the common output length
            num_workers=self.num_workers,
            features=self.forcing_features + [self.target],
            static_features=[c for c in self.static_features if c != self.group_identifier],
            target=self.target,
            use_proportional_split=self.use_proportional_split,
            train_prop=self.train_prop,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
            max_missing_pct=self.max_missing_pct,
            min_train_years=self.min_train_years,
            domain_id=domain_id
        )
        
        print(f"Created {model_type} DataModule with input_length={input_len}, "
              f"output_length={self.output_length}, batch_size={self.batch_size}")
        
        return datamodule
    
    # For backward compatibility
    def prepare_datamodule(self, 
                           caravan: Caravanify, 
                           basin_ids: List[str],
                           domain_id: str = "CA") -> HydroDataModule:
        """Create a HydroDataModule with the specified basin IDs (legacy method).
        
        Args:
            caravan: Loaded Caravanify instance
            basin_ids: List of basin IDs to include
            domain_id: Domain identifier for the dataset
            
        Returns:
            Configured HydroDataModule
        """
        # Load only the specified basins
        caravan.load_stations(basin_ids)
        
        # Get time series and static data
        ts_data = caravan.get_time_series()
        static_data = caravan.get_static_attributes()
        
        # Filter columns
        ts_columns = self.forcing_features + [self.target, "date", self.group_identifier]
        ts_data = ts_data[ts_columns]
        
        static_columns = self.static_features
        static_data = static_data[static_columns]
        
        # Create datamodule
        datamodule = HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.group_identifier,
            preprocessing_config=self.get_preprocessing_config(),
            batch_size=self.batch_size,
            input_length=self.input_length,
            output_length=self.output_length,
            num_workers=self.num_workers,
            features=self.forcing_features + [self.target],
            static_features=[c for c in self.static_features if c != self.group_identifier],
            target=self.target,
            use_proportional_split=self.use_proportional_split,
            train_prop=self.train_prop,
            val_prop=self.val_prop,
            test_prop=self.test_prop,
            max_missing_pct=self.max_missing_pct,
            min_train_years=self.min_train_years,
            domain_id=domain_id
        )
        
        return datamodule
