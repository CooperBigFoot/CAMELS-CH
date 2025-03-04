from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# TODO: Improve docstrings and type hints

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
        """Get all gauge IDs from the timeseries directory."""
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix

        if not ts_dir.exists():
            raise FileNotFoundError(
                f"Timeseries directory not found for prefix {self.config.gauge_id_prefix}: {ts_dir}"
            )

        gauge_ids = [f.stem for f in ts_dir.glob("*.csv")]
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
        """Load timeseries CSVs in parallel using multithreading."""
        ts_dir = self.config.timeseries_dir / self.config.gauge_id_prefix
        file_paths = []
        for gauge_id in gauge_ids:
            fp = ts_dir / f"{gauge_id}.csv"
            if not fp.exists():
                raise FileNotFoundError(f"Timeseries file {fp} not found")
            file_paths.append(fp)

        def read_single(fp: Path) -> pd.DataFrame:
            # Consider adding engine='pyarrow' if installed for faster parsing
            df = pd.read_csv(fp, parse_dates=["date"])  # , engine='pyarrow')
            df["gauge_id"] = fp.stem
            return df

        with ThreadPoolExecutor() as executor:
            dfs = list(executor.map(read_single, file_paths))

        for df in dfs:
            self.time_series[df["gauge_id"].iloc[0]] = df

    def _load_static_attributes(self, gauge_ids: List[str]) -> None:
        """Load and merge static attributes using efficient concatenation."""
        attr_dir = self.config.attributes_dir / self.config.gauge_id_prefix
        gauge_ids_set = set(gauge_ids)
        dfs = []

        # Helper function to load and process attributes
        def load_attributes(file_name: str) -> Union[pd.DataFrame, None]:
            file_path = attr_dir / file_name
            if not file_path.exists():
                return None

            df = pd.read_csv(file_path, dtype={"gauge_id": str}, engine="pyarrow")
            df = df[df["gauge_id"].isin(gauge_ids_set)]
            df.set_index("gauge_id", inplace=True)
            return df

        # Load enabled attribute types
        if self.config.use_other_attributes:
            other_df = load_attributes(
                f"attributes_other_{self.config.gauge_id_prefix}.csv"
            )
            if other_df is not None:
                dfs.append(other_df)

        if self.config.use_hydroatlas_attributes:
            hydro_df = load_attributes(
                f"attributes_hydroatlas_{self.config.gauge_id_prefix}.csv"
            )
            if hydro_df is not None:
                dfs.append(hydro_df)

        if self.config.use_caravan_attributes:
            caravan_df = load_attributes(
                f"attributes_caravan_{self.config.gauge_id_prefix}.csv"
            )
            if caravan_df is not None:
                dfs.append(caravan_df)

        # Concatenate all DataFrames horizontally
        if dfs:
            self.static_attributes = pd.concat(dfs, axis=1, join="outer").reset_index()

    def _validate_gauge_ids(self, gauge_ids: List[str]) -> None:
        """Ensure all gauge IDs start with the configured prefix."""
        prefix = f"{self.config.gauge_id_prefix}_"
        for gid in gauge_ids:
            if not gid.startswith(prefix):
                raise ValueError(f"Gauge ID {gid} must start with '{prefix}'")

    def get_time_series(self) -> pd.DataFrame:
        """Return concatenated time series data."""
        if not self.time_series:
            return pd.DataFrame()
        df = pd.concat(self.time_series.values(), ignore_index=True)
        return df[
            ["gauge_id", "date"]
            + [c for c in df.columns if c not in ("gauge_id", "date")]
        ]

    def get_static_attributes(self) -> pd.DataFrame:
        """Return merged static attributes."""
        return self.static_attributes.copy()
