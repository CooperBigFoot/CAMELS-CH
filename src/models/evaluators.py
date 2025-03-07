import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns


class TSForecastEvaluator:
    """Evaluator for time series forecasting models with per-basin metrics support."""

    def __init__(self, datamodule, horizons: List[int]):
        self.datamodule = datamodule
        self.horizons = horizons

    def evaluate(
        self, test_results: Dict[str, torch.Tensor]
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        # Data extraction and processing
        basin_ids = np.array(test_results["basin_ids"]).flatten()
        preds = test_results["predictions"].cpu().numpy()
        obs = test_results["observations"].cpu().numpy()

        # Expand basin IDs for each horizon
        basin_ids_expanded = np.repeat(basin_ids, preds.shape[1])
        preds_flat = preds.flatten()
        obs_flat = obs.flatten()

        # Inverse transformations
        if hasattr(self.datamodule, "inverse_transform_predictions"):
            preds_flat = self.datamodule.inverse_transform_predictions(
                preds_flat, basin_ids_expanded
            )
            obs_flat = self.datamodule.inverse_transform_predictions(
                obs_flat, basin_ids_expanded
            )

        # Create evaluation dataframe
        horizons_expanded = np.tile(self.horizons, len(basin_ids))
        df = pd.DataFrame(
            {
                "horizon": horizons_expanded,
                "prediction": preds_flat,
                "observed": obs_flat,
                "basin_id": basin_ids_expanded,
            }
        )

        # Calculate overall metrics
        overall_metrics = {}
        for h in self.horizons:
            horizon_data = df[df["horizon"] == h]
            overall_metrics[h] = self._calculate_metrics(horizon_data)

        # Calculate per-basin metrics
        basin_metrics = {}
        for basin in df["basin_id"].unique():
            basin_metrics[basin] = {}
            basin_data = df[df["basin_id"] == basin]

            for h in self.horizons:
                horizon_data = basin_data[basin_data["horizon"] == h]
                basin_metrics[basin][h] = self._calculate_metrics(horizon_data)

        return df, overall_metrics, basin_metrics

    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Helper method to calculate metrics for a subset of data."""
        if len(data) == 0:
            return {metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]}

        pred = data["prediction"].values
        obs = data["observed"].values
        return {
            "MSE": self.calculate_mse(pred, obs),
            "MAE": self.calculate_mae(pred, obs),
            "NSE": self.calculate_nse(pred, obs),
            "RMSE": self.calculate_rmse(pred, obs),
        }

    def summarize_metrics(self, metrics: Dict, per_basin: bool = False) -> pd.DataFrame:
        """Create a summary DataFrame of metrics.

        Args:
            metrics: Dictionary of metrics (either overall or per-basin)
            per_basin: Whether metrics are per-basin

        Returns:
            DataFrame with metrics as columns and appropriate index
        """
        rows = []

        if per_basin:
            for basin, basin_data in metrics.items():
                for horizon, horizon_metrics in basin_data.items():
                    rows.append(
                        {"basin_id": basin, "horizon": horizon, **horizon_metrics}
                    )
            return pd.DataFrame(rows).set_index(["basin_id", "horizon"])

        else:
            for horizon, horizon_metrics in metrics.items():
                rows.append({"horizon": horizon, **horizon_metrics})
            return pd.DataFrame(rows).set_index("horizon")

    def plot_rolling_forecast(
        self,
        horizon: int,
        group_identifier: str,
        datamodule,
        fig_size: tuple = (12, 6),
        title: str = None,
        date_format: str = "%Y-%m-%d",
        y_label: str = "Streamflow",
        color_observed: str = "blue",
        color_forecast: str = "red",
        alpha_forecast: float = 1.0,
        line_style_forecast: str = "--",
        line_width_forecast: float = 2.0,
        debug: bool = False,
    ) -> tuple:
        """Create a rolling forecast plot for a specific basin and horizon.

        Args:
            horizon: Forecast horizon in days
            group_identifier: Identifier for the group (e.g., basin ID)
            datamodule: Data module containing the dataset and inverse transformation methods
            fig_size: Size of the figure
            title: Title of the plot
            date_format: Date format for x-axis
            y_label: Y-axis label
            color_observed: Color for observed data
            color_forecast: Color for forecast data
            alpha_forecast: Alpha transparency for forecast line
            line_style_forecast: Line style for forecast line
            line_width_forecast: Line width for forecast line
            debug: If True, print debug information

        Returns:
            Tuple of figure and axis objects
        """

        # Validate horizon
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} not in available horizons: {self.horizons}"
            )

        # Extract test results data
        basin_ids = np.array(self.test_results["basin_ids"]).flatten()
        preds = self.test_results["predictions"].cpu().numpy()
        obs = self.test_results["observations"].cpu().numpy()

        # Access the forecast dates for the specific horizon (if available)
        forecast_dates = None
        slice_indices = None
        if "slice_idx" in self.test_results:
            slice_indices = self.test_results["slice_idx"]

        if "forecast_dates" in self.test_results:
            all_forecast_dates = self.test_results["forecast_dates"]
            horizon_idx = self.horizons.index(horizon)
            forecast_dates = [dates[horizon_idx] for dates in all_forecast_dates]

        # Find all indices for the requested group_identifier
        mask = np.array([bid == group_identifier for bid in basin_ids])
        if not np.any(mask):
            available_ids = np.unique(basin_ids)
            raise ValueError(
                f"Group identifier '{group_identifier}' not found in test results. Available IDs: {available_ids}"
            )

        if debug:
            print(f"Found {np.sum(mask)} matches for {group_identifier}")

        group_indices = np.where(mask)[0]

        # Get horizon-specific data
        horizon_idx = self.horizons.index(horizon)

        # Extract predictions and observations for this group and horizon
        group_preds = preds[mask, horizon_idx]
        group_obs = obs[mask, horizon_idx]

        # Extract corresponding forecast dates for this group if available
        if forecast_dates is not None:
            group_forecast_dates = [forecast_dates[i] for i in group_indices]
        else:
            group_forecast_dates = None

        if debug:
            print(
                f"Extracted {len(group_preds)} predictions and {len(group_obs)} observations"
            )
            print(f"Predictions range: [{np.min(group_preds)}, {np.max(group_preds)}]")
            print(f"Observations range: [{np.min(group_obs)}, {np.max(group_obs)}]")

        # Apply inverse transformation
        basin_ids_expanded = np.repeat([group_identifier], len(group_preds))
        group_preds = datamodule.inverse_transform_predictions(
            group_preds, basin_ids_expanded
        )
        group_obs = datamodule.inverse_transform_predictions(
            group_obs, basin_ids_expanded
        )

        if debug:
            print(f"After inverse transform:")
            print(f"Predictions range: [{np.min(group_preds)}, {np.max(group_preds)}]")
            print(f"Observations range: [{np.min(group_obs)}, {np.max(group_obs)}]")

        # Get the dataset's sorted dataframe
        if not hasattr(datamodule, "test_dataset") or datamodule.test_dataset is None:
            raise ValueError(
                "Datamodule missing test_dataset. Ensure the datamodule has been properly set up."
            )

        df_sorted = datamodule.test_dataset.df_sorted
        if df_sorted is None or df_sorted.empty:
            raise ValueError("Dataset's sorted dataframe is empty or missing.")

        # If we don't have forecast dates from the test results, derive them from the test dataset
        if group_forecast_dates is None:
            if debug:
                print(
                    "Forecast dates not available in test results, deriving from dataset"
                )

            # Get the forecast dates from the test dataset's index
            test_index = datamodule.test_dataset.index
            basin_rows = test_index[
                test_index[datamodule.group_identifier] == group_identifier
            ]

            # Generate target dates for the specific horizon
            input_end_dates = basin_rows["input_end_date"].values
            group_forecast_dates = [
                date + pd.Timedelta(days=horizon) for date in input_end_dates
            ]

            # Ensure we have the right number of dates
            if len(group_forecast_dates) > len(group_preds):
                group_forecast_dates = group_forecast_dates[-len(group_preds) :]
            elif len(group_forecast_dates) < len(group_preds):
                print(
                    f"Warning: Not enough dates ({len(group_forecast_dates)}) for predictions ({len(group_preds)})"
                )
                # Extend dates by adding days to the last date
                last_date = group_forecast_dates[-1]
                additional_dates = [
                    last_date + pd.Timedelta(days=i + 1)
                    for i in range(len(group_preds) - len(group_forecast_dates))
                ]
                group_forecast_dates = group_forecast_dates + additional_dates

        # Create plot with Seaborn style
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot observations as a continuous line
        ax.plot(
            group_forecast_dates,
            group_obs,
            color=color_observed,
            label="Observed",
            linewidth=2,
            zorder=10,
        )

        # Plot predictions as a single line
        ax.plot(
            group_forecast_dates,
            group_preds,
            color=color_forecast,
            alpha=alpha_forecast,
            label=f"{horizon}-Day Forecast",
            linestyle=line_style_forecast,
            linewidth=line_width_forecast,
            zorder=15,
        )

        # Set title and labels
        if title is None:
            title = f"{horizon}-day Forecast for {group_identifier}"

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

        # Add legend with distinctive appearance
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Clean up the plot
        sns.despine()

        # Format y-axis to avoid scientific notation
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

        # Tight layout for better appearance
        fig.tight_layout()

        return fig, ax

    # Existing static metric calculation methods remain unchanged
    @staticmethod
    def calculate_mse(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.mean((pred - obs) ** 2)

    @staticmethod
    def calculate_mae(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.mean(np.abs(pred - obs))

    @staticmethod
    def calculate_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
        return np.sqrt(np.mean((pred - obs) ** 2))

    @staticmethod
    def calculate_nse(pred: np.ndarray, obs: np.ndarray) -> float:
        return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
