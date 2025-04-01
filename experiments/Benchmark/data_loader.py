"""Data loading functions for the Central Asian data sharing experiment."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, Any, Optional

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def load_data(config: Any, country: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Load and prepare datasets from Central Asia with optional country filtering.

    Args:
        config: Experiment configuration object
        country: Optional country filter ('Tajikistan', 'Kyrgyzstan', or None)
                If None or 'Combined', no country filtering is applied
        **kwargs: Additional arguments (unused)

    Returns:
        Dictionary containing:
        - 'time_series': DataFrame with time series data
        - 'static': DataFrame with static catchment attributes
        - 'basin_count': Number of unique basins in the dataset
    """
    # Get Central Asia dataset configuration
    ca_config = CaravanifyConfig(
        attributes_dir=config.ca_attribute_dir,
        timeseries_dir=config.ca_timeseries_dir,
        gauge_id_prefix=config.ca_gauge_id_prefix,
        human_influence_path=config.ca_human_influence_path,
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    # Initialize and load all basins
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()

    # Apply human influence filtering - use only low and medium influence catchments
    print(f"Found {len(ca_basins)} total CA basins")
    filtered_basins, discarded_basins = ca_caravan.filter_gauge_ids_by_human_influence(
        ca_basins, ["Low", "Medium"]
    )
    print(f"Loading {len(filtered_basins)} CA basins after human influence filtering")
    print(f"Discarded {len(discarded_basins)} CA basins with high human influence")

    # Load stations data
    ca_caravan.load_stations(filtered_basins)

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
        country_name = "Combined"
        print(f"Using all {len(filtered_basins)} basins ({country_name} dataset)")

    # Return the filtered data
    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(static_data[config.group_identifier].unique()),
    }
