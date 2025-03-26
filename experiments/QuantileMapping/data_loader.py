"""Data loading utilities for quantile mapping experiment."""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def load_data(config: Any, data_source: str = "original", **kwargs) -> Dict[str, Any]:
    """
    Load data for the quantile mapping experiment.

    Args:
        config: Experiment configuration
        data_source: Which data source to use ('original' or 'quantile_mapped')
        **kwargs: Additional arguments

    Returns:
        Dictionary containing:
        - 'time_series': pd.DataFrame - Time series data
        - 'static': pd.DataFrame - Static catchment attributes
        - 'basin_count': Number of unique basins
        - 'forcing_features': List of forcing features
    """
    if data_source.lower() == "original":
        return load_original_data(config)
    elif data_source.lower() == "quantile_mapped":
        return load_quantile_mapped_data(config)
    else:
        raise ValueError(
            f"Unsupported data source: {data_source}. Use 'original' or 'quantile_mapped'"
        )


def load_original_data(config: Any) -> Dict[str, Any]:
    """
    Load original time series data with reduced feature set.

    Args:
        config: Configuration object with dataset paths

    Returns:
        Dictionary containing time series and static data
    """
    print("Loading original time series data with reduced feature set")

    # Configure Caravan dataset
    ca_config = CaravanifyConfig(
        attributes_dir=config.ca_config["ATTRIBUTE_DIR"],
        timeseries_dir=config.ca_config["TIMESERIES_DIR"],
        gauge_id_prefix=config.ca_config["GAUGE_ID_PREFIX"],
        human_influence_path=config.ca_config["HUMAN_INFLUENCE_PATH"],
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    # Initialize and load all basins
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()

    # Apply human influence filtering
    print(f"Found {len(ca_basins)} total basins")
    ca_basins, discarded_ca = ca_caravan.filter_gauge_ids_by_human_influence(
        ca_basins, ["Low", "Medium"]
    )
    print(f"Using {len(ca_basins)} basins after human influence filtering")
    print(f"Discarded {len(discarded_ca)} basins with high human influence")

    ca_caravan.load_stations(ca_basins)

    # Get static data
    static_columns = config.static_features
    static_data = ca_caravan.get_static_attributes()[static_columns]
    gauge_ids = static_data[config.group_identifier].unique()

    # Define reduced forcing features
    reduced_forcing_features = config.forcing_features

    # Load time series data but use only the reduced feature set
    ts_columns = reduced_forcing_features + [
        config.target,
        "date",
        config.group_identifier,
    ]
    ts_data = ca_caravan.get_time_series()[ts_columns]

    # Report data statistics
    print(f"Loaded {len(ts_data)} time series records across {len(gauge_ids)} basins")
    print(f"Time range: {ts_data['date'].min()} to {ts_data['date'].max()}")
    print(f"Using forcing features: {reduced_forcing_features}")

    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(gauge_ids),
        "forcing_features": reduced_forcing_features,
    }


def load_quantile_mapped_data(config: Any) -> Dict[str, Any]:
    """
    Load quantile-mapped time series data.

    Args:
        config: Configuration object with dataset paths

    Returns:
        Dictionary containing time series and static data
    """
    print("Loading quantile mapped time series data")

    if not config.quantile_mapped_folder:
        raise ValueError("Quantile mapped folder path must be provided")

    # First load static data using original method to get gauge IDs
    ca_config = CaravanifyConfig(
        attributes_dir=config.ca_config["ATTRIBUTE_DIR"],
        timeseries_dir=config.ca_config["TIMESERIES_DIR"],
        gauge_id_prefix=config.ca_config["GAUGE_ID_PREFIX"],
        human_influence_path=config.ca_config["HUMAN_INFLUENCE_PATH"],
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    # Initialize and get filtered basins for static data
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()
    ca_basins, _ = ca_caravan.filter_gauge_ids_by_human_influence(
        ca_basins, ["Low", "Medium"]
    )

    ca_caravan.load_stations(ca_basins)

    # Get static data (same for both sources)
    static_columns = config.static_features
    static_data = ca_caravan.get_static_attributes()[static_columns]
    gauge_ids = static_data[config.group_identifier].unique()

    # Define reduced forcing features
    reduced_forcing_features = config.forcing_features

    # Load quantile mapped time series using ThreadPoolExecutor
    ts_dir = Path(config.quantile_mapped_folder)
    if not ts_dir.exists():
        raise FileNotFoundError(f"Quantile mapped folder does not exist: {ts_dir}")

    file_paths = []
    for gauge_id in gauge_ids:
        fp = ts_dir / f"{gauge_id}.csv"
        if not fp.exists():
            print(f"Warning: Timeseries file {fp} not found")
            continue
        file_paths.append(fp)

    if not file_paths:
        raise FileNotFoundError(f"No valid timeseries files found in {ts_dir}")

    print(
        f"Found {len(file_paths)} quantile-mapped files out of {len(gauge_ids)} basins"
    )

    def read_single(fp: Path) -> pd.DataFrame:
        # Always use pyarrow engine for faster parsing if available
        try:
            df = pd.read_csv(fp, parse_dates=["date"], engine="pyarrow")
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            df = pd.DataFrame()
        df["gauge_id"] = fp.stem
        return df

    time_series_dfs = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        dfs = list(executor.map(read_single, file_paths))
        time_series_dfs.extend(dfs)

    ts_data = pd.concat(time_series_dfs, ignore_index=True)

    # Ensure all required columns are present
    required_columns = reduced_forcing_features + [
        config.target,
        "date",
        config.group_identifier,
    ]

    # Check if gauge_id needs to be renamed to match config.group_identifier
    if "gauge_id" in ts_data.columns and config.group_identifier != "gauge_id":
        ts_data = ts_data.rename(columns={"gauge_id": config.group_identifier})

    # Check for missing columns
    missing_columns = set(required_columns) - set(ts_data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in quantile mapped data: {missing_columns}")

    # Select only the columns we need
    ts_data = ts_data[required_columns]

    # Report data statistics
    print(
        f"Loaded {len(ts_data)} quantile-mapped time series records across {len(file_paths)} basins"
    )
    print(f"Time range: {ts_data['date'].min()} to {ts_data['date'].max()}")
    print(f"Using forcing features: {reduced_forcing_features}")

    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(file_paths),
        "forcing_features": reduced_forcing_features,
    }
