import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import torch


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
