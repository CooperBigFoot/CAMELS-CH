"""Data loading utilities for the Central Asian data sharing experiment."""

import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import framework utilities
from src.experiment_framework.data_utils import load_country_data

# Configure logging
logger = logging.getLogger(__name__)

def load_data(config: Any, **kwargs) -> Dict[str, Any]:
    """
    Load and prepare datasets from Central Asia with optional country filtering.
    
    This function leverages the framework's load_country_data utility to load
    data from Central Asian basins with specific configuration for this experiment.
    
    Args:
        config: Configuration object with dataset paths
        **kwargs: Additional keyword arguments from CLI
            - country: Optional country filter ('Tajikistan', 'Kyrgyzstan', or 'Combined')
                      If 'Combined', no country filtering is applied
        
    Returns:
        Dictionary containing:
        - 'time_series': pd.DataFrame - Time series data
        - 'static': pd.DataFrame - Static catchment attributes
        - 'basin_count': int - Number of basins
        - 'country': str - The country filter used
    """
    # Extract country from kwargs or use default
    country = kwargs.get("country", "Combined")
    
    # Setup caravanify config for the framework function
    caravanify_config = {
        "ATTRIBUTE_DIR": config.ca_config["attribute_dir"],
        "TIMESERIES_DIR": config.ca_config["timeseries_dir"],
        "GAUGE_ID_PREFIX": config.ca_config["gauge_id_prefix"],
        "HUMAN_INFLUENCE_PATH": config.ca_config["human_influence_path"],
        "MIN_TRAIN_YEARS": config.ca_config["min_train_years"],
    }
    
    # Use the framework's data loading utility with country filtering
    data = load_country_data(
        config=config,
        country=None if country.lower() == "combined" else country,
        caravanify_config=caravanify_config
    )
    
    # Add country to the returned data
    data["country"] = country
    
    # Log the data loading results
    logger.info(f"Loaded data for {country} scenario with {data['basin_count']} basins")
    
    return data
