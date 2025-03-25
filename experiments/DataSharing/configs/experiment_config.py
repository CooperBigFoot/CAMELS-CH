"""Configuration for Central Asian data sharing experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

from src.experiment_framework.config import BaseExperimentConfig


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    """Configuration for the Central Asian data sharing experiment.
    
    This class provides configuration parameters for evaluating the impact 
    of data sharing between Tajikistan and Kyrgyzstan on hydrological model
    performance through various scenarios.
    """
    # Country scenarios - unique to this experiment
    countries: List[str] = field(
        default_factory=lambda: ["Tajikistan", "Kyrgyzstan", "Combined"]
    )
    
    # CA-specific configuration
    ca_config: Dict[str, Any] = field(default_factory=lambda: {
        "attribute_dir": "/workspace/CARAVANIFY/CA/post_processed/attributes",
        "timeseries_dir": "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv",
        "gauge_id_prefix": "CA",
        "min_train_years": 5,
        "human_influence_path": "/workspace/CAMELS-CH/src/human_influence_index/results/human_influence_classification.csv",
    })
    
    # Override default static features to include country for filtering
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
    
    # Override default forcing features with CA-specific ones
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

    def get_country_dir(self, country: str) -> Dict[str, Path]:
        """Get country-specific directories.
        
        Args:
            country: Country name (Tajikistan, Kyrgyzstan, or Combined)
            
        Returns:
            Dictionary of country-specific paths for checkpoints, logs, and results
        """
        dirs = self.get_output_dirs()
        
        country_dirs = {
            "checkpoints": dirs["checkpoints"] / country.lower(),
            "logs": dirs["logs"] / country.lower(),
            "results": dirs["results"] / country.lower(),
        }
        
        # Create directories if they don't exist
        for dir_path in country_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return country_dirs
