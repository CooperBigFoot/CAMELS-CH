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
        input_end_dates = test_results["input_end_date"]

        # Expand basin IDs and calculate dates for each horizon
        basin_ids_expanded = np.repeat(basin_ids, preds.shape[1])
        preds_flat = preds.flatten()
        obs_flat = obs.flatten()

        # Create expanded dates for each horizon
        dates_expanded = []
        for i, input_date in enumerate(input_end_dates):
            input_date_dt = pd.to_datetime(input_date)
            for horizon in self.horizons:
                # Calculate forecast date by adding horizon days to input end date
                forecast_date = input_date_dt + pd.Timedelta(days=horizon)
                dates_expanded.append(forecast_date)

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
                "date": dates_expanded,
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
        """Create a rolling forecast plot for a specific basin and horizon."""
        # Validate horizon
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} not in available horizons: {self.horizons}"
            )

        # Use the evaluation dataframe that now has dates
        df, _, _ = self.evaluate(self.test_results)

        # Filter for the specific basin and horizon
        basin_df = df[(df["basin_id"] == group_identifier) & (df["horizon"] == horizon)]

        if basin_df.empty:
            available_ids = df["basin_id"].unique()
            raise ValueError(
                f"Group identifier '{group_identifier}' not found in test results. Available IDs: {available_ids}"
            )

        if debug:
            print(f"Found {len(basin_df)} data points for {group_identifier}")
            print(
                f"Predictions range: [{basin_df['prediction'].min()}, {basin_df['prediction'].max()}]"
            )
            print(
                f"Observations range: [{basin_df['observed'].min()}, {basin_df['observed'].max()}]"
            )

        # Create plot with Seaborn style
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot observations and predictions
        ax.plot(
            basin_df["date"],
            basin_df["observed"],
            color=color_observed,
            label="Observed",
            linewidth=2,
            zorder=10,
        )

        ax.plot(
            basin_df["date"],
            basin_df["prediction"],
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

        # Add legend and formatting
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        sns.despine()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
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
