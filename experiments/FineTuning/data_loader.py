"""Data loading utilities for the fine-tuning experiment."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, Any

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def load_data(config: Any, **kwargs) -> Dict[str, Any]:
    """
    Load data for the fine-tuning experiment.
    
    Args:
        config: Experiment configuration
        **kwargs: Additional arguments from command line
        
    Returns:
        Dictionary containing:
        - 'time_series': pd.DataFrame - Time series data
        - 'static': pd.DataFrame - Static catchment attributes
        - 'basin_count': int - Number of basins
    """
    # Use command line country if provided, otherwise use config
    country = kwargs.get("country", config.target_country)
    
    # Configure Caravan dataset
    ca_config = CaravanifyConfig(
        attributes_dir=config.attribute_dir,
        timeseries_dir=config.timeseries_dir,
        gauge_id_prefix=config.gauge_id_prefix,
        human_influence_path=config.human_influence_path,
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
    print(f"Loading {len(ca_basins)} CA basins after human influence filtering")
    print(f"Discarded {len(discarded_ca)} CA basins with high human influence")

    ca_caravan.load_stations(ca_basins)

    # Prepare columns
    ts_columns = config.forcing_features + [config.target]
    static_columns = config.static_features
    ts_columns_with_date = ts_columns + ["date"] + [config.group_identifier]

    # Get all data
    all_ts_data = ca_caravan.get_time_series()[ts_columns_with_date]
    all_static_data = ca_caravan.get_static_attributes()[static_columns]

    # Apply country filtering if specified
    if country and country.lower() != "combined":
        print(f"Filtering data for country: {country}")

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

        print(f"Selected {len(country_gauge_ids)} basins in {country}")
    else:
        # Use all data
        ts_data = all_ts_data
        static_data = all_static_data
        print(f"Using all {len(ca_basins)} basins (Combined dataset)")

    # Return the filtered data
    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(static_data[config.group_identifier].unique()),
    }
