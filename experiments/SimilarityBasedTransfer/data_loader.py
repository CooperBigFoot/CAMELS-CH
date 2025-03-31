"""Data loading utilities for group-based transfer learning experiment."""

from pathlib import Path
import sys
import pandas as pd
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_models.caravanify import Caravanify, CaravanifyConfig


def extract_source_basins_for_training(config: Any) -> Dict:
    """
    Extract source basins corresponding to CA target basins grouped by their major groups.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with training data structure organized by target groups
    """
    # Load the cluster data
    ca_groups = pd.read_csv(config.ca_groups_path)
    source_clusters = pd.read_csv(config.source_clusters_path)

    training_data = {}

    # Process each group
    for group_key, group_info in config.group_mappings.items():
        # Skip if not in target groups
        if group_key not in config.target_groups:
            continue

        # Get CA basins for this group - now used only for reference/verification
        ca_group_basins = ca_groups[
            ca_groups["major_group"] == group_info["ca_group_label"]
        ]["gauge_id"].tolist()

        # Get source basins for this group by cluster
        source_group_basins = source_clusters[
            source_clusters["cluster"].isin(group_info["clusters"])
        ]["gauge_id"].tolist()

        # Split source basins by country
        country_basins = {
            "CH": [basin for basin in source_group_basins if basin.startswith("CH_")],
            "USA": [basin for basin in source_group_basins if basin.startswith("USA_")],
            "CL": [basin for basin in source_group_basins if basin.startswith("CL_")],
        }

        # Store in training data structure - only source basins are relevant for training
        training_data[group_key] = {
            "target": ca_group_basins,  # Only kept for reference
            "source": country_basins,
        }

        # Print summary statistics
        print(f"{group_key} - Target CA basins (for reference only): {len(ca_group_basins)}")
        for country, basins in country_basins.items():
            print(f"{group_key} - {country} source basins for training: {len(basins)}")
        print()

    return training_data


def load_data_for_group(config: Any, group_key: str) -> Dict[str, Any]:
    """
    Load source data for a specific group using Caravanify.
    Target (CA) data is not included in the training dataset.

    Args:
        config: Experiment configuration
        group_key: Key of the group to load data for (e.g., 'group1')

    Returns:
        Dictionary with loaded time series and static data
    """
    # Get training data mapping
    training_data = extract_source_basins_for_training(config)

    if group_key not in training_data:
        raise ValueError(
            f"Group key '{group_key}' not found in training data. "
            f"Available groups: {list(training_data.keys())}"
        )

    print(f"Loading source data for {group_key}...")

    # Dictionary to store data from each source country
    data = {
        "source_ts_data": [],
        "source_static_data": [],
    }

    # Load source data for each country
    country_configs = {
        "CH": config.ch_config,
        "USA": config.usa_config,
        "CL": config.cl_config,
    }

    total_source_basins = 0

    for country, basins in training_data[group_key]["source"].items():
        if not basins:  # Skip if no basins for this country
            continue

        country_cfg = country_configs[country]
        source_config = CaravanifyConfig(
            attributes_dir=country_cfg["attribute_dir"],
            timeseries_dir=country_cfg["timeseries_dir"],
            gauge_id_prefix=country_cfg["gauge_id_prefix"],
            human_influence_path=country_cfg["human_influence_path"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        source_caravan = Caravanify(source_config)
        source_caravan.load_stations(basins)

        source_ts_data = source_caravan.get_time_series()
        source_static_data = source_caravan.get_static_attributes()

        data["source_ts_data"].append(source_ts_data)
        data["source_static_data"].append(source_static_data)
        
        total_source_basins += len(basins)
        
        print(
            f"Loaded {len(basins)} {country} basins with "
            f"{len(source_ts_data)} time series records"
        )

    # Prepare combined source data
    # Extract required columns
    ts_columns = config.forcing_features + [
        config.target,
        "date",
        config.group_identifier,
    ]
    static_columns = config.static_features

    # Filter and combine source data
    all_source_ts_data = []
    all_source_static_data = []

    for ts_df, static_df in zip(data["source_ts_data"], data["source_static_data"]):
        filtered_ts = ts_df[ts_columns]
        filtered_static = static_df[static_columns]
        all_source_ts_data.append(filtered_ts)
        all_source_static_data.append(filtered_static)

    # Check if we have any source data
    if not all_source_ts_data:
        raise ValueError(f"No source data found for group {group_key}")
        
    # Combine source data
    combined_ts_data = pd.concat(all_source_ts_data, ignore_index=True)
    combined_static_data = pd.concat(all_source_static_data, ignore_index=True)

    combined_data = {
        "time_series": combined_ts_data,
        "static": combined_static_data,
        "basin_count": total_source_basins,
    }

    print(
        f"Combined source data for {group_key}: {len(combined_data['time_series'])} time series records, "
        f"{len(combined_static_data[config.group_identifier].unique())} unique basins"
    )

    return combined_data


def load_data(config: Any, group_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Main data loading function for the experiment.

    Args:
        config: Experiment configuration
        group_key: Optional group key to load specific group data
        **kwargs: Additional arguments

    Returns:
        Dictionary containing loaded data
    """
    if group_key:
        # Load specific group data (source only)
        return load_data_for_group(config, group_key)
    else:
        # Load training data mapping only
        return {"training_data": extract_source_basins_for_training(config)}
