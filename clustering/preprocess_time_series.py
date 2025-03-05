from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_timeseries_data(
    df: pd.DataFrame,
    basin_id_col: str = "gauge_id",
    date_col: str = "date",
    flow_col: str = "streamflow",
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare time series data for clustering by aggregating to weekly data and standardizing.

    Args:
        df: DataFrame with daily streamflow data
        basin_id_col: Column name for basin ID
        date_col: Column name for date
        flow_col: Column name for streamflow
        standardize: Whether to apply z-score standardization

    Returns:
        Tuple containing:
        - Array of (standardized) weekly time series (shape: n_basins x 52)
        - List of basin IDs corresponding to the time series
    """
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Add water year
    df["water_year"] = df[date_col].dt.year.where(
        df[date_col].dt.month < 10, df[date_col].dt.year + 1
    )

    # Add week of water year (1-52)
    df["day_of_water_year"] = df.groupby([basin_id_col, "water_year"])[
        date_col
    ].transform(lambda x: (x - x.min()).dt.days)
    df["week"] = (df["day_of_water_year"] // 7) + 1
    df["week"] = df["week"].clip(upper=52)  # Ensure we don't exceed 52 weeks

    # Group by basin, water year and week, calculate mean weekly flow
    weekly_df = (
        df.groupby([basin_id_col, "water_year", "week"])[flow_col].mean().reset_index()
    )

    # Calculate mean annual cycle (52 weeks) for each basin
    mean_annual_df = (
        weekly_df.groupby([basin_id_col, "week"])[flow_col].mean().reset_index()
    )

    # Reshape to wide format (each row is a basin, each column is a week)
    wide_df = mean_annual_df.pivot(index=basin_id_col, columns="week", values=flow_col)

    # Ensure all 52 weeks are present
    for week in range(1, 53):
        if week not in wide_df.columns:
            wide_df[week] = np.nan

    wide_df = wide_df.reindex(columns=range(1, 53))

    basin_ids = wide_df.index.tolist()
    ts_data = []

    for basin_id in basin_ids:
        series = wide_df.loc[basin_id].values
        # Fill any missing values with linear interpolation
        series = pd.Series(series).interpolate().values

        # Z-score standardization if requested
        if standardize:
            std_series = zscore(series)
            ts_data.append(std_series)
        else:
            ts_data.append(series)

    return np.array(ts_data), basin_ids


def plot_standardized_hydrographs(
    ts_data: np.ndarray,
    basin_ids: List[str],
    selected_basins: List[str] = None,
    max_display: int = 5,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot standardized weekly hydrographs for selected basins.

    Args:
        ts_data: Array of standardized weekly time series (shape: n_basins x 52)
        basin_ids: List of basin IDs corresponding to the time series
        selected_basins: List of basin IDs to plot (if None, will plot first max_display)
        max_display: Maximum number of basins to display
        figsize: Figure size (width, height)
    """
    basin_id_map = {id: i for i, id in enumerate(basin_ids)}

    if selected_basins is None:
        indices = list(range(min(len(basin_ids), max_display)))
        selected_basins = [basin_ids[i] for i in indices]
    else:
        indices = [
            basin_id_map[basin] for basin in selected_basins if basin in basin_id_map
        ]
        selected_basins = [basin_ids[i] for i in indices]

    plt.figure(figsize=figsize)
    weeks = np.arange(1, 53)

    # Generate colors from a colormap for distinct lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))

    for i, (idx, basin, color) in enumerate(zip(indices, selected_basins, colors)):
        plt.plot(weeks, ts_data[idx], label=f"Basin {basin}", color=color, linewidth=2)

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    plt.xlabel("Week of Water Year")
    plt.ylabel("Standardized Flow (Z-score)")
    plt.title("Standardized Weekly Hydrographs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.show()
