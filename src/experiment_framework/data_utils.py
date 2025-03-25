"""Data handling utilities for hydrological forecasting experiments."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


def create_datamodule(
    time_series_df: pd.DataFrame,
    static_df: pd.DataFrame,
    config: Any,
    input_length: int,
    output_length: int,
    batch_size: Optional[int] = None,
    features: Optional[List[str]] = None,
    static_features: Optional[List[str]] = None,
) -> Any:
    """Create a HydroDataModule from loaded data.
    
    Args:
        time_series_df: DataFrame with time series data
        static_df: DataFrame with static catchment attributes
        config: Experiment configuration
        input_length: Length of input sequence
        output_length: Length of output sequence
        batch_size: Optional batch size override
        features: Optional list of features to use
        static_features: Optional list of static features to use
        
    Returns:
        Configured HydroDataModule
    """
    try:
        from src.data_models.datamodule import HydroDataModule
    except ImportError:
        raise ImportError(
            "HydroDataModule not found. Make sure src.data_models.datamodule is available."
        )
    
    # Set defaults from config if not provided
    batch_size = batch_size or config.batch_size
    features = features or (config.forcing_features + [config.target])
    
    # Handle static features - exclude group_identifier if it's in the list
    if static_features is None:
        static_features = [f for f in config.static_features if f != config.group_identifier]
    
    # Get preprocessing config from the experiment config
    preprocessing_config = config.get_preprocessing_config()
    
    # Create data module
    data_module = HydroDataModule(
        time_series_df=time_series_df,
        static_df=static_df,
        group_identifier=config.group_identifier,
        preprocessing_config=preprocessing_config,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        num_workers=min(config.max_workers, 8),  # Cap at 8 workers
        features=features,
        static_features=static_features,
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
        min_train_years=config.min_train_years,
    )
    
    return data_module


def setup_preprocessing(config: Any) -> Dict[str, Dict[str, Any]]:
    """Set up preprocessing pipelines based on config.
    
    This is a wrapper around config.get_preprocessing_config() that allows
    for additional customization or validation.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing preprocessing configurations
    """
    preprocessing_config = config.get_preprocessing_config()
    
    # Validate the preprocessing config
    expected_keys = {"features", "target", "static_features"}
    missing_keys = expected_keys - set(preprocessing_config.keys())
    
    if missing_keys:
        raise ValueError(f"Missing preprocessing configurations for: {missing_keys}")
        
    return preprocessing_config


def validate_data(
    time_series_df: pd.DataFrame,
    static_df: pd.DataFrame,
    config: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and clean data before using in experiments.
    
    Args:
        time_series_df: Time series data frame
        static_df: Static data frame
        config: Experiment configuration
        
    Returns:
        Cleaned time_series_df and static_df
    """
    # Check for required columns
    required_ts_columns = ["date", config.group_identifier, config.target] + config.forcing_features
    missing_ts_columns = set(required_ts_columns) - set(time_series_df.columns)
    
    if missing_ts_columns:
        raise ValueError(f"Missing required columns in time series data: {missing_ts_columns}")
    
    required_static_columns = [config.group_identifier] + [
        col for col in config.static_features if col != config.group_identifier
    ]
    missing_static_columns = set(required_static_columns) - set(static_df.columns)
    
    if missing_static_columns:
        raise ValueError(f"Missing required columns in static data: {missing_static_columns}")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(time_series_df["date"]):
        time_series_df["date"] = pd.to_datetime(time_series_df["date"])
    
    # Check for NaN values
    ts_nan_cols = time_series_df[required_ts_columns].columns[
        time_series_df[required_ts_columns].isna().any()
    ].tolist()
    
    if ts_nan_cols:
        logger.warning(f"NaN values found in time series columns: {ts_nan_cols}")
        
    static_nan_cols = static_df[required_static_columns].columns[
        static_df[required_static_columns].isna().any()
    ].tolist()
    
    if static_nan_cols:
        logger.warning(f"NaN values found in static columns: {static_nan_cols}")
    
    # Ensure matching gauge IDs
    ts_gauge_ids = set(time_series_df[config.group_identifier].unique())
    static_gauge_ids = set(static_df[config.group_identifier].unique())
    
    missing_in_static = ts_gauge_ids - static_gauge_ids
    if missing_in_static:
        logger.warning(
            f"Found {len(missing_in_static)} gauge IDs in time series but not in static data"
        )
        # Filter time series to only include gauges with static data
        time_series_df = time_series_df[
            time_series_df[config.group_identifier].isin(static_gauge_ids)
        ]
    
    missing_in_ts = static_gauge_ids - ts_gauge_ids
    if missing_in_ts:
        logger.warning(
            f"Found {len(missing_in_ts)} gauge IDs in static data but not in time series data"
        )
        # Filter static to only include gauges with time series data
        static_df = static_df[
            static_df[config.group_identifier].isin(ts_gauge_ids)
        ]
    
    return time_series_df, static_df


def load_country_data(
    config: Any,
    country: Optional[str] = None,
    caravanify_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load and prepare datasets with optional country filtering.
    
    This is a utility function for loading data from a Caravanify dataset
    with optional filtering by country.
    
    Args:
        config: Experiment configuration
        country: Optional country filter
        caravanify_config: Optional configuration for Caravanify
        
    Returns:
        Dictionary containing time series and static data frames
    """
    try:
        from src.data_models.caravanify import Caravanify, CaravanifyConfig
    except ImportError:
        raise ImportError(
            "Caravanify not found. Make sure src.data_models.caravanify is available."
        )
    
    # Configure default Caravanify settings if not provided
    if caravanify_config is None:
        if not hasattr(config, "CA_CONFIG"):
            raise ValueError(
                "CA_CONFIG not found in config and no caravanify_config provided"
            )
        caravanify_config = config.CA_CONFIG
    
    # Initialize Caravanify configuration
    ca_config = CaravanifyConfig(
        attributes_dir=caravanify_config.get("ATTRIBUTE_DIR"),
        timeseries_dir=caravanify_config.get("TIMESERIES_DIR"),
        gauge_id_prefix=caravanify_config.get("GAUGE_ID_PREFIX", "CA"),
        human_influence_path=caravanify_config.get("HUMAN_INFLUENCE_PATH"),
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )
    
    # Initialize and load all basins
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()
    
    # Apply human influence filtering if available
    if hasattr(ca_caravan, "filter_gauge_ids_by_human_influence"):
        logger.info(f"Found {len(ca_basins)} total basins")
        ca_basins, discarded_ca = ca_caravan.filter_gauge_ids_by_human_influence(
            ca_basins, ["Low", "Medium"]
        )
        logger.info(f"Using {len(ca_basins)} basins after human influence filtering")
        logger.info(f"Discarded {len(discarded_ca)} basins with high human influence")
    
    ca_caravan.load_stations(ca_basins)
    
    # Prepare columns
    ts_columns = config.forcing_features + [config.target]
    static_columns = config.static_features
    
    # Add date and group identifier columns to time series
    ts_columns_with_date = ts_columns + ["date"] + [config.group_identifier]
    
    # Get all data
    all_ts_data = ca_caravan.get_time_series()[ts_columns_with_date]
    all_static_data = ca_caravan.get_static_attributes()[static_columns]
    
    # Apply country filtering if specified
    if country and country.lower() != "combined" and "country" in all_static_data.columns:
        logger.info(f"Filtering data for country: {country}")
        
        # Get gauge IDs for specified country
        country_gauge_ids = all_static_data[all_static_data["country"] == country][
            config.group_identifier
        ].unique()
        
        if len(country_gauge_ids) == 0:
            raise ValueError(f"No basins found for country: {country}")
        
        # Filter time series and static data
        ts_data = all_ts_data[
            all_ts_data[config.group_identifier].isin(country_gauge_ids)
        ]
        static_data = all_static_data[
            all_static_data[config.group_identifier].isin(country_gauge_ids)
        ]
        
        logger.info(f"Selected {len(country_gauge_ids)} basins in {country}")
    else:
        # Use all data
        ts_data = all_ts_data
        static_data = all_static_data
        logger.info(f"Using all {len(ca_basins)} basins (Combined dataset)")
    
    # Validate and clean data
    ts_data, static_data = validate_data(ts_data, static_data, config)

    # Remove country column if it exists
    if "country" in static_data.columns:
        static_data = static_data.drop(columns=["country"])
        logger.info("Removed 'country' column from static data")
    
    # Return the filtered data
    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(static_data[config.group_identifier].unique()),
    }
