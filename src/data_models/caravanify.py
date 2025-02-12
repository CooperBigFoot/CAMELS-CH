from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict
import pandas as pd


@dataclass
class CaravanifyConfig:
    """Configuration for loading Caravan-formatted datasets."""

    attributes_dir: Union[str, Path]
    timeseries_dir: Union[str, Path]
    gauge_id_prefix: str
    use_caravan_attributes: bool = True
    use_hydroatlas_attributes: bool = False
    use_other_attributes: bool = False

    def __post_init__(self):
        self.attributes_dir = Path(self.attributes_dir)
        self.timeseries_dir = Path(self.timeseries_dir)


class Caravanify:
    def __init__(self, config: CaravanifyConfig):
        self.config = config
        self.time_series: Dict[str, pd.DataFrame] = {}  # {gauge_id: DataFrame}
        self.static_attributes = pd.DataFrame()  # Combined static attributes

    def get_all_gauge_ids(self) -> List[str]:
        """Get all gauge IDs from the timeseries directory.

        Returns:
            List[str]: List of all gauge IDs found in the timeseries directory.
            Each ID will start with the configured gauge_id_prefix.

        Raises:
            FileNotFoundError: If the timeseries directory for the prefix doesn't exist.
        """
        # Construct path to prefix-specific directory
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix

        if not ts_dir.exists():
            raise FileNotFoundError(
                f"Timeseries directory not found for prefix {self.config.gauge_id_prefix}: {ts_dir}"
            )

        # Get all CSV files and extract gauge IDs from filenames
        gauge_ids = [f.stem for f in ts_dir.glob("*.csv")]

        # Validate that all gauge IDs start with the correct prefix
        prefix = f"{self.config.gauge_id_prefix}_"
        invalid_ids = [gid for gid in gauge_ids if not gid.startswith(prefix)]
        if invalid_ids:
            raise ValueError(
                f"Found gauge IDs that don't match prefix {prefix}: {invalid_ids}"
            )

        return sorted(gauge_ids)

    def load_stations(self, gauge_ids: List[str]) -> None:
        """Load data for specified gauge IDs."""
        self._validate_gauge_ids(gauge_ids)
        self._load_timeseries(gauge_ids)
        self._load_static_attributes(gauge_ids)

    def _load_timeseries(self, gauge_ids: List[str]) -> None:
        """Load timeseries CSVs from timeseries_dir/gauge_id_prefix/"""
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix

        for gauge_id in gauge_ids:
            file_path = ts_dir / f"{gauge_id}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Timeseries file {file_path} not found")

            # Read the CSV file but don't set date as index
            df = pd.read_csv(file_path, parse_dates=["date"])

            # Add gauge_id column
            df["gauge_id"] = gauge_id

            # Store DataFrame in dictionary
            self.time_series[gauge_id] = df

    def _load_static_attributes(self, gauge_ids: List[str]) -> None:
        """Load and merge enabled attribute files."""
        attr_dir = self.config.attributes_dir / self.config.gauge_id_prefix
        dfs = []

        # Load metadata (lat/lon/area/etc.)
        if self.config.use_other_attributes:
            other_path = (
                attr_dir / f"attributes_other_{self.config.gauge_id_prefix}.csv"
            )
            other_df = pd.read_csv(other_path, dtype={"gauge_id": str})
            dfs.append(other_df[other_df["gauge_id"].isin(gauge_ids)])

        # Load HydroATLAS attributes
        if self.config.use_hydroatlas_attributes:
            hydroatlas_path = (
                attr_dir / f"attributes_hydroatlas_{self.config.gauge_id_prefix}.csv"
            )
            hydro_df = pd.read_csv(hydroatlas_path, dtype={"gauge_id": str})
            dfs.append(hydro_df[hydro_df["gauge_id"].isin(gauge_ids)])

        # Load Caravan climate indices
        if self.config.use_caravan_attributes:
            caravan_path = (
                attr_dir / f"attributes_caravan_{self.config.gauge_id_prefix}.csv"
            )
            caravan_df = pd.read_csv(caravan_path, dtype={"gauge_id": str})
            dfs.append(caravan_df[caravan_df["gauge_id"].isin(gauge_ids)])

        # Merge all DataFrames
        if dfs:
            # Merge on gauge_id
            self.static_attributes = dfs[0]
            for df in dfs[1:]:
                self.static_attributes = pd.merge(
                    self.static_attributes, df, on="gauge_id", how="outer"
                )

    def _validate_gauge_ids(self, gauge_ids: List[str]) -> None:
        """Ensure all gauge IDs start with the configured prefix."""
        prefix = f"{self.config.gauge_id_prefix}_"
        for gid in gauge_ids:
            if not gid.startswith(prefix):
                raise ValueError(f"Gauge ID {gid} must start with '{prefix}'")

    def get_time_series(self) -> pd.DataFrame:
        """Return all loaded timeseries as a concatenated DataFrame with gauge_id and date as columns."""
        if not self.time_series:
            return pd.DataFrame()

        # Concatenate all DataFrames
        df = pd.concat(self.time_series.values(), ignore_index=True)

        # Ensure gauge_id and date are the first columns
        cols = df.columns.tolist()
        cols.remove("gauge_id")
        cols.remove("date")
        df = df[["gauge_id", "date"] + cols]

        return df

    def get_static_attributes(self) -> pd.DataFrame:
        """Return merged static attributes."""
        return self.static_attributes.copy()
