"""Visualization functions for hydrological forecasting evaluation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional


def plot_rolling_forecast(
    df: pd.DataFrame,
    horizon: int,
    group_identifier: str,
    fig_size: tuple = (12, 6),
    title: str = None,
    color_scheme: Dict[str, str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a rolling forecast plot for a specific basin and horizon.

    Args:
        df: DataFrame with prediction results (has columns: 'horizon', 'date', 'prediction', 'observed', 'basin_id')
        horizon: Forecast horizon to visualize
        group_identifier: Basin ID or group identifier to plot
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        color_scheme: Optional dictionary with colors for 'observed' and 'prediction'

    Returns:
        Tuple containing the figure and axes objects
    """
    # Set default color scheme if not provided
    if color_scheme is None:
        color_scheme = {"observed": "blue", "prediction": "red"}

    # Filter for the specific basin and horizon
    basin_df = df[(df["basin_id"] == group_identifier) & (df["horizon"] == horizon)]

    if basin_df.empty:
        available_ids = df["basin_id"].unique()
        raise ValueError(
            f"Group identifier '{group_identifier}' not found in results. Available IDs: {available_ids}"
        )

    # Create plot with Seaborn style
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot observations and predictions
    ax.plot(
        basin_df["date"],
        basin_df["observed"],
        color=color_scheme["observed"],
        label="Observed",
        linewidth=2,
    )

    ax.plot(
        basin_df["date"],
        basin_df["prediction"],
        color=color_scheme["prediction"],
        alpha=0.8,
        label=f"{horizon}-Day Forecast",
        linewidth=2,
    )

    # Set title and labels
    if title is None:
        title = f"{horizon}-day Forecast for {group_identifier}"

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Streamflow", fontsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Add legend and formatting
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.7)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_metric_boxplot(
    basin_metrics: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "NSE",
    horizons: Optional[List[int]] = None,
    fig_size: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    violin: bool = False,
    individual_points: bool = True,
    palette: str = "Blues",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a boxplot or violin plot of a specific metric across horizons.

    Args:
        basin_metrics: Nested dictionary of metrics by basin and horizon
        metric: Metric to visualize (default: "NSE")
        horizons: List of horizons to include (if None, use all available horizons)
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        violin: If True, use violin plots instead of boxplots
        individual_points: If True, show individual data points
        palette: Color palette for the plot

    Returns:
        Tuple containing the figure and axes objects
    """
    # Flatten the nested dictionary to a DataFrame
    rows = []
    for basin, horizon_data in basin_metrics.items():
        for horizon, metrics_dict in horizon_data.items():
            if metric in metrics_dict:
                rows.append(
                    {
                        "basin_id": basin,
                        "horizon": horizon,
                        "metric_value": metrics_dict[metric],
                    }
                )

    df = pd.DataFrame(rows)

    # Filter by horizons if specified
    if horizons:
        df = df[df["horizon"].isin(horizons)]

    # Determine available horizons after filtering
    available_horizons = sorted(df["horizon"].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Create violin plot or boxplot
    if violin:
        sns.violinplot(
            x="horizon",
            y="metric_value",
            data=df,
            palette=palette,
            inner="quartile",
            ax=ax,
        )
    else:
        sns.boxplot(x="horizon", y="metric_value", data=df, palette=palette, ax=ax)

    # Add individual points if requested
    if individual_points:
        sns.stripplot(
            x="horizon",
            y="metric_value",
            data=df,
            color="black",
            size=3,
            alpha=0.3,
            jitter=True,
            ax=ax,
        )

    # Add median value labels
    for i, horizon in enumerate(available_horizons):
        median = df[df["horizon"] == horizon]["metric_value"].median()
        ax.text(
            i,
            median + 0.02,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
            fontsize=9,
        )

    # Set title and labels
    ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
    ax.set_ylabel(f"{metric} Value", fontsize=12)

    if title is None:
        title = f"Distribution of {metric} Values by Horizon"
    ax.set_title(title, fontsize=14)

    # Add grid lines for better readability
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Apply styling
    sns.despine()
    plt.tight_layout()

    return fig, ax


def plot_metric_cdf(
    basin_metrics: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "NSE",
    horizon: int = 1,
    fig_size: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    color: str = "steelblue",
    threshold_lines: Optional[List[float]] = None,
    threshold_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the cumulative distribution function (CDF) of a metric for a specific horizon.

    Args:
        basin_metrics: Nested dictionary of metrics by basin and horizon
        metric: Metric to visualize (default: "NSE")
        horizon: Forecast horizon to visualize
        fig_size: Figure size as (width, height)
        title: Custom title for the plot
        color: Color for the CDF line
        threshold_lines: Optional list of thresholds to mark on the plot
        threshold_labels: Optional labels for threshold lines

    Returns:
        Tuple containing the figure and axes objects
    """
    # Extract metric values for the specified horizon
    metric_values = []
    for basin, horizon_data in basin_metrics.items():
        if horizon in horizon_data and metric in horizon_data[horizon]:
            value = horizon_data[horizon][metric]
            # Skip NaN values
            if not np.isnan(value):
                metric_values.append(value)

    if not metric_values:
        raise ValueError(
            f"No data available for metric '{metric}' at horizon {horizon}"
        )

    # Sort values for CDF
    sorted_values = np.sort(metric_values)

    # Calculate cumulative probabilities
    n = len(sorted_values)
    cumulative_prob = np.arange(1, n + 1) / n

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot CDF
    ax.plot(
        sorted_values,
        cumulative_prob,
        color=color,
        linewidth=2.5,
        label=f"{metric} CDF",
    )

    # Add median line
    median_value = np.median(sorted_values)
    ax.axvline(
        x=median_value,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Median: {median_value:.3f}",
    )

    # Add threshold lines if provided
    if threshold_lines:
        if not threshold_labels:
            threshold_labels = [f"Threshold: {t}" for t in threshold_lines]

        for threshold, label in zip(threshold_lines, threshold_labels):
            # Find y-value (probability) at threshold
            idx = np.searchsorted(sorted_values, threshold)
            prob_at_threshold = cumulative_prob[min(idx, len(cumulative_prob) - 1)]

            # Horizontal line at threshold
            ax.axvline(
                x=threshold, color="green", linestyle="-.", alpha=0.7, label=label
            )

            # Add text annotation
            ax.text(
                threshold + 0.02,
                0.1,
                f"{prob_at_threshold * 100:.1f}%",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                fontweight="bold",
            )

    # Calculate percentage of values above/below zero (useful for NSE)
    if metric == "NSE":
        pct_above_zero = sum(v > 0 for v in metric_values) / len(metric_values) * 100
        ax.axvline(
            x=0,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"NSE > 0: {pct_above_zero:.1f}%",
        )

    # Set labels and title
    ax.set_xlabel(f"{metric} Value", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)

    if title is None:
        title = f"CDF of {metric} for {horizon}-day Forecast Horizon"
    ax.set_title(title, fontsize=14)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    # Apply styling
    sns.despine()
    plt.tight_layout()

    return fig, ax
