from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Literal
import pandas as pd
from enum import Enum


@dataclass
class CamelsCHConfig:
    # Time series data
    timeseries_dir: Union[str, Path]
    timeseries_pattern: str

    # Static attributes
    static_attributes_dir: Union[str, Path]

    # Which static attribute files to load
    use_climate: bool = True
    use_geology: bool = True
    use_glacier: bool = True
    use_human_influence: bool = True
    use_hydrogeology: bool = True
    use_hydrology: bool = True
    use_landcover: bool = True
    use_soil: bool = True
    use_topographic: bool = True

    def __post_init__(self):
        self.timeseries_dir = Path(self.timeseries_dir)
        self.static_attributes_dir = Path(self.static_attributes_dir)


def get_all_gauge_ids(config: CamelsCHConfig) -> List[str]:
    """Get all gauge IDs from the timeseries directory"""
    pattern = config.timeseries_pattern.replace("*", "*")
    files = list(config.timeseries_dir.glob(pattern))
    return [file.stem.split("_")[-1] for file in files]


class StaticAttributeType(Enum):
    CLIMATE = "climate"
    GEOLOGY = "geology"
    GLACIER = "glacier"
    HUMAN_INFLUENCE = "humaninfluence"
    HYDROGEOLOGY = "hydrogeology"
    HYDROLOGY = "hydrology"
    LANDCOVER = "landcover"
    SOIL = "soil"
    TOPOGRAPHIC = "topographic"


class CamelsCH:
    def __init__(self, config: CamelsCHConfig):
        self.config = config
        self.time_series = {}
        self.static = {attr_type: pd.DataFrame() for attr_type in StaticAttributeType}

    def load_stations(self, gauge_ids: List[str]) -> None:
        for gauge_id in gauge_ids:
            self._check_gauge_id(gauge_id)

        self._load_timeseries(gauge_ids)
        self._load_static(gauge_ids)

    def _load_timeseries(self, gauge_ids: List[str]) -> None:
        """Internal method to load timeseries data"""

        for basin_id in gauge_ids:
            pattern = self.config.timeseries_pattern.replace("*", basin_id)
            file_path = next(self.config.timeseries_dir.glob(pattern))

            self.time_series[basin_id] = pd.read_csv(
                file_path, parse_dates=["date"], index_col="date"
            )

        print(f"Loaded time series data for {len(gauge_ids)} stations")

    def _load_static(self, gauge_ids: List[str]) -> None:
        """Load enabled static attributes for specified basins"""

        attr_map = {
            StaticAttributeType.CLIMATE: self.config.use_climate,
            StaticAttributeType.GEOLOGY: self.config.use_geology,
            StaticAttributeType.GLACIER: self.config.use_glacier,
            StaticAttributeType.HUMAN_INFLUENCE: self.config.use_human_influence,
            StaticAttributeType.HYDROGEOLOGY: self.config.use_hydrogeology,
            StaticAttributeType.HYDROLOGY: self.config.use_hydrology,
            StaticAttributeType.LANDCOVER: self.config.use_landcover,
            StaticAttributeType.SOIL: self.config.use_soil,
            StaticAttributeType.TOPOGRAPHIC: self.config.use_topographic,
        }

        if not any(attr_map.values()):
            return

        for attr_type, enabled in attr_map.items():
            if enabled:
                print(f"Loading {attr_type.value} attributes")

                self._load_attribute(attr_type, gauge_ids)

        print(f"Loaded static attributes for {len(gauge_ids)} stations")

    def _load_attribute(
        self, attr_type: StaticAttributeType, gauge_ids: List[str]
    ) -> None:
        """Load a specific static attribute for specified basins"""

        suffix = (
            "_obs"
            if attr_type in [StaticAttributeType.CLIMATE, StaticAttributeType.HYDROLOGY]
            else ""
        )
        filename = f"CAMELS_CH_{attr_type.value}_attributes{suffix}.csv"
        filepath = self.config.static_attributes_dir / filename

        df = pd.read_csv(
            filepath, comment="#", dtype={"gauge_id": str}, encoding="latin1"
        )
        df = df[df["gauge_id"].isin(gauge_ids)].set_index("gauge_id")
        self.static[attr_type] = df

    def get_static_attributes(self) -> pd.DataFrame:
        """Combine all loaded static attributes into a single DataFrame"""
        dfs = []
        for attr_type in StaticAttributeType:
            if attr_type in self.static and not self.static[attr_type].empty:
                df = self.static[attr_type].reset_index()
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Start with first DataFrame
        result = dfs[0]
        # Merge remaining DataFrames one by one
        for df in dfs[1:]:
            result = pd.merge(result, df, on="gauge_id", how="outer")

        return result

    def get_time_series(self, gauge_ids: List[str] = None) -> pd.DataFrame:
        """Get all time series data as a single DataFrame with repeated gauge_id column."""
        if gauge_ids is None:
            gauge_ids = list(self.time_series.keys())

        dfs = []
        for gauge_id in gauge_ids:
            if gauge_id in self.time_series:
                df = self.time_series[gauge_id].copy()
                df = df.reset_index()
                df["gauge_id"] = gauge_id
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _check_gauge_id(self, gauge_id: str) -> None:
        """Check if gauge_id exists in the CAMELS-CH dataset"""
        all_gauge_ids = set(get_all_gauge_ids(self.config))

        if gauge_id not in all_gauge_ids:
            raise ValueError(f"Gauge ID {gauge_id} not found in the dataset")


if __name__ == "__main__":
    config = CamelsCHConfig(
        timeseries_dir="/Users/cooper/Desktop/CAMELS-CH/data/timeseries/observation_based/",
        timeseries_pattern="CAMELS_CH_obs_based_*.csv",
        static_attributes_dir="/Users/cooper/Desktop/CAMELS-CH/data/static_attributes",
    )

    camels = CamelsCH(config)
    camels.load_stations(["2018", "6005"])

    # Print time series data
    print(camels.get_time_series())
    print("-" * 80)

    # Print static attributes
    print(camels.get_static_attributes())
