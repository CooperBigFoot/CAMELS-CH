"""Data loading utilities for quantile mapping experiment."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def load_data_by_source(
    config: Any,
    data_source: str = "original",
    quantile_mapped_folder: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load either original or quantile mapped time series data with reduced feature set.
    
    Args:
        config: Configuration object with dataset paths
        data_source: Data source type - either "original" or "quantile_mapped"
        quantile_mapped_folder: Path to folder with quantile mapped data
        
    Returns:
        Dictionary containing time series and static data
    """
    # Configure Caravan dataset
    ca_config = CaravanifyConfig(
        attributes_dir=config.CA_CONFIG["ATTRIBUTE_DIR"],
        timeseries_dir=config.CA_CONFIG["TIMESERIES_DIR"],
        gauge_id_prefix=config.CA_CONFIG["GAUGE_ID_PREFIX"],
        human_influence_path=config.CA_CONFIG["HUMAN_INFLUENCE_PATH"],
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    # Initialize and load all basins
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()

    # Apply human influence filtering
    print(f"Found {len(ca_basins)} total CA basins")
    ca_basins, discarded_ca = ca_caravan.filter_gauge_ids_by_human_influence(
        ca_basins, ["Low", "Medium"]
    )
    print(f"Using {len(ca_basins)} CA basins after human influence filtering")
    print(f"Discarded {len(discarded_ca)} CA basins with high human influence")

    ca_caravan.load_stations(ca_basins)

    # Get static data (same for both sources)
    static_columns = config.STATIC_FEATURES
    static_data = ca_caravan.get_static_attributes()[static_columns]
    gauge_ids = static_data[config.GROUP_IDENTIFIER].unique()
    
    # Define reduced forcing features for both data sources
    reduced_forcing_features = config.FORCING_FEATURES
    
    if data_source.lower() == "original":
        # Load original timeseries data but use only the reduced feature set
        print("Loading original time series data with reduced feature set")
        ts_columns = reduced_forcing_features + [config.TARGET, "date", config.GROUP_IDENTIFIER]
        ts_data = ca_caravan.get_time_series()[ts_columns]
        
    elif data_source.lower() == "quantile_mapped":
        # Load quantile mapped timeseries data
        print("Loading quantile mapped time series data")
        if not quantile_mapped_folder:
            raise ValueError("Quantile mapped folder path must be provided")
            
        # Load quantile mapped time series using ThreadPoolExecutor
        ts_dir = Path(quantile_mapped_folder)
        file_paths = []
        for gauge_id in gauge_ids:
            fp = ts_dir / f"{gauge_id}.csv"
            if not fp.exists():
                print(f"Warning: Timeseries file {fp} not found")
                continue
            file_paths.append(fp)
            
        if not file_paths:
            raise FileNotFoundError(f"No valid timeseries files found in {ts_dir}")

        def read_single(fp: Path) -> pd.DataFrame:
            # Always use pyarrow engine for faster parsing if available
            try:
                df = pd.read_csv(fp, parse_dates=["date"], engine="pyarrow")
            except:
                df = pd.read_csv(fp, parse_dates=["date"])
            df["gauge_id"] = fp.stem
            return df

        time_series_dfs = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            dfs = list(executor.map(read_single, file_paths))
            time_series_dfs.extend(dfs)

        ts_data = pd.concat(time_series_dfs, ignore_index=True)
        
        # Ensure all required columns are present
        required_columns = reduced_forcing_features + [config.TARGET, "date", "gauge_id"]
        missing_columns = set(required_columns) - set(ts_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in quantile mapped data: {missing_columns}")
            
        # Select only the columns we need
        ts_data = ts_data[required_columns]
        
        # Rename gauge_id to match config.GROUP_IDENTIFIER if needed
        if "gauge_id" != config.GROUP_IDENTIFIER:
            ts_data = ts_data.rename(columns={"gauge_id": config.GROUP_IDENTIFIER})
        
    else:
        raise ValueError(f"Unsupported data source: {data_source}. Use 'original' or 'quantile_mapped'")
    
    # Validate the loaded data
    expected_columns = reduced_forcing_features + [config.TARGET, "date", config.GROUP_IDENTIFIER]
    for col in expected_columns:
        if col not in ts_data.columns:
            raise ValueError(f"Required column {col} not found in loaded data")
    
    # Report data statistics
    print(f"Loaded {len(ts_data)} time series records across {len(gauge_ids)} basins")
    print(f"Time range: {ts_data['date'].min()} to {ts_data['date'].max()}")
    print(f"Using forcing features: {reduced_forcing_features}")
    
    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(gauge_ids),
        "forcing_features": reduced_forcing_features
    }
