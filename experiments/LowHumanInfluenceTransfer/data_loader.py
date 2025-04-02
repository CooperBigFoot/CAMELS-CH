"""Data loading functions for the global hydrological pretraining experiment."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, Any

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def load_data(config: Any, **kwargs) -> Dict[str, Any]:
    """
    Load and prepare datasets from global regions (CH, CL, USA) with human influence filtering.
    
    Args:
        config: Experiment configuration object
        **kwargs: Additional arguments (unused)
        
    Returns:
        Dictionary containing:
        - 'time_series': List of DataFrames with time series data
        - 'static': List of DataFrames with static catchment attributes
        - 'basin_count': Dictionary with basin counts per region
    """
    regions = ["CH", "CL", "USA"]  # Explicitly exclude "CA"
    
    # Containers for results
    all_ts_data = []
    all_static_data = []
    basin_counts = {}
    total_basins = 0
    
    # Process each region
    for region in regions:
        print(f"\nLoading data for {region}...")
        
        # Get region-specific configuration
        region_config = getattr(config, f"{region}_CONFIG")
        
        # Initialize Caravanify
        caravanify_config = CaravanifyConfig(
            attributes_dir=region_config["ATTRIBUTE_DIR"],
            timeseries_dir=region_config["TIMESERIES_DIR"],
            gauge_id_prefix=region_config["GAUGE_ID_PREFIX"],
            human_influence_path=region_config["HUMAN_INFLUENCE_PATH"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )
        
        # Create Caravanify instance
        caravan = Caravanify(caravanify_config)
        
        # Get all basin IDs for the region
        all_basins = caravan.get_all_gauge_ids()
        print(f"Found {len(all_basins)} total {region} basins")
        
        # Filter basins by human influence
        filtered_basins, discarded_basins = caravan.filter_gauge_ids_by_human_influence(
            all_basins, ["Low", "Medium"]
        )
        print(f"Loading {len(filtered_basins)} {region} basins after human influence filtering")
        print(f"Discarded {len(discarded_basins)} {region} basins with high human influence")
        
        # Load station data
        caravan.load_stations(filtered_basins)
        
        # Extract columns
        ts_columns = config.forcing_features + [config.target, "date", config.group_identifier]
        static_columns = config.static_features
        
        # Extract data frames
        ts_data = caravan.get_time_series()[ts_columns]
        static_data = caravan.get_static_attributes()[static_columns]
        
        # Add domain identifier to avoid gauge_id conflicts
        ts_data["domain"] = region
        static_data["domain"] = region
        
        # Store data and counts
        all_ts_data.append(ts_data)
        all_static_data.append(static_data)
        basin_counts[region] = len(filtered_basins)
        total_basins += len(filtered_basins)
        
        print(f"Loaded {len(ts_data)} time series records from {len(filtered_basins)} {region} basins")
    
    # Add total to basin counts
    basin_counts["total"] = total_basins
    print(f"\nTotal basins across all regions: {total_basins}")
    
    return {
        "time_series": all_ts_data,
        "static": all_static_data,
        "basin_count": basin_counts,
    }
