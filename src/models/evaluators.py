import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pytorch_lightning as pl
import copy


class TSForecastEvaluator:
    """Evaluator for time series forecasting models with per-basin metrics support."""

    def __init__(
        self,
        datamodule,
        horizons: List[int],
        models: Dict[str, pl.LightningModule] = None,
        benchmark_model: str = None,
        trainer_kwargs: Dict = None,
    ):
        self.datamodule = datamodule
        self.horizons = horizons
        

        # Create deep copies of the models to avoid shared state issues
        self.models = {}
        if models:
            for name, model in models.items():
                self.models[name] = copy.deepcopy(model)

        self.benchmark_model = benchmark_model
        self.trainer_kwargs = trainer_kwargs or {"accelerator": "cpu", "devices": 1}
        self.results = {}

    def test_models(self, datamodule=None):
        """Test all registered models and evaluate results"""
        if datamodule is None:
            datamodule = self.datamodule

        for name, model in self.models.items():
            print(f"Testing {name}...")

            # Create a trainer and run test
            trainer = pl.Trainer(**self.trainer_kwargs)
            trainer.test(model, datamodule=datamodule)

            # Verify model has test results
            if not hasattr(model, "test_results"):
                raise AttributeError(
                    f"Model {name} doesn't have test_results attribute. "
                    "Ensure your LightningModule stores test outputs in self.test_results."
                )

            # Debug the shapes
            print(f"Predictions shape: {model.test_results['predictions'].shape}")
            print(f"Observations shape: {model.test_results['observations'].shape}")
            print(f"Number of basin_ids: {len(model.test_results['basin_ids'])}")
            if "input_end_date" in model.test_results:
                print(
                    f"Number of input_end_dates: {len(model.test_results['input_end_date'])}"
                )
            print(f"Configured horizons: {self.horizons}")

            # Extract and evaluate results
            df, metrics, basin_metrics = self.evaluate(model.test_results)
            self.results[name] = {
                "df": df,
                "metrics": metrics,
                "basin_metrics": basin_metrics,
            }

        return self.results

    def evaluate(
        self, test_results: Dict[str, torch.Tensor]
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Evaluate model test results and compute metrics.
        
        Args:
            test_results: Dictionary containing model outputs and observations
                
        Returns:
            Tuple containing:
                - DataFrame with predictions, observations, and metadata
                - Dictionary with overall metrics by horizon
                - Dictionary with per-basin metrics by horizon
        """
        # Create evaluation dataframe from test results
        df = self._prepare_evaluation_dataframe(test_results)
        
        # Calculate overall metrics for each horizon
        overall_metrics = self._calculate_overall_metrics(df)
        
        # Calculate per-basin metrics
        basin_metrics = self._calculate_basin_metrics(df)
        
        return df, overall_metrics, basin_metrics

    def _prepare_evaluation_dataframe(
        self, test_results: Dict[str, torch.Tensor]
    ) -> pd.DataFrame:
        """
        Create a flattened dataframe with predictions, observations, and metadata.
        
        Args:
            test_results: Dictionary containing model outputs and observations
                
        Returns:
            DataFrame with predictions, observations, basin IDs, horizons and dates
        """
        # Data extraction
        basin_ids = np.array(test_results["basin_ids"])
        preds = test_results["predictions"].cpu().numpy()
        obs = test_results["observations"].cpu().numpy()

        print(
            f"Evaluating results with shape: preds={preds.shape}, obs={obs.shape}, basin_ids={basin_ids.shape}"
        )

        # Ensure pred and obs dimensions match
        if preds.shape != obs.shape:
            raise ValueError(
                f"Prediction shape {preds.shape} doesn't match observation shape {obs.shape}"
            )

        # Create expanded basin IDs and horizons
        if preds.ndim == 2:  # [batch_size, pred_len]
            horizons_per_sample = preds.shape[1]
            
            # Handle horizon mismatch - don't modify self.horizons, use a local variable
            current_horizons = self.horizons
            
            if horizons_per_sample != len(current_horizons):
                print(
                    f"Warning: Model output has {horizons_per_sample} horizons but evaluator configured with {len(current_horizons)} horizons"
                )
                # Use the actual horizons from model output
                current_horizons = list(range(1, horizons_per_sample + 1))
                print(f"Using adjusted horizons: {current_horizons}")
            
            # Flatten predictions and observations
            preds_flat = preds.flatten()
            obs_flat = obs.flatten()
            
            # Repeat each basin ID for each horizon in the output
            basin_ids_expanded = np.repeat(basin_ids, horizons_per_sample)
            
            # Create repeated horizons array matching the model's output structure
            horizons_expanded = np.tile(current_horizons, len(basin_ids))
            
            # Verify all arrays have matching lengths
            assert len(preds_flat) == len(obs_flat) == len(basin_ids_expanded) == len(horizons_expanded), \
                f"Array length mismatch: preds_flat={len(preds_flat)}, obs_flat={len(obs_flat)}, " \
                f"basin_ids_expanded={len(basin_ids_expanded)}, horizons_expanded={len(horizons_expanded)}"

            # Create dates if available
            if "input_end_date" in test_results:
                input_end_dates = test_results["input_end_date"]

                # Ensure input_end_dates matches basin_ids length
                if len(input_end_dates) != len(basin_ids):
                    print(
                        f"Warning: input_end_dates length ({len(input_end_dates)}) doesn't match basin_ids length ({len(basin_ids)})"
                    )
                    # Adjust to match basin_ids
                    if len(input_end_dates) < len(basin_ids):
                        input_end_dates = input_end_dates + [input_end_dates[-1]] * (
                            len(basin_ids) - len(input_end_dates)
                        )
                    else:
                        input_end_dates = input_end_dates[: len(basin_ids)]

                # Create expanded dates for each horizon - use current_horizons not self.horizons
                dates_expanded = []
                for i, input_date in enumerate(input_end_dates):
                    input_date_dt = pd.to_datetime(input_date)
                    for horizon in current_horizons:
                        # Calculate forecast date by adding horizon days to input end date
                        forecast_date = input_date_dt + pd.Timedelta(days=horizon)
                        dates_expanded.append(forecast_date)
                        
                # Verify dates_expanded length matches other arrays        
                assert len(dates_expanded) == len(preds_flat), \
                    f"dates_expanded length {len(dates_expanded)} doesn't match preds_flat length {len(preds_flat)}"
            else:
                # Create dummy dates if not available
                print("Warning: No input_end_dates found, using dummy dates")
                dates_expanded = [pd.Timestamp.now()] * len(preds_flat)
        else:
            raise ValueError(
                f"Unexpected prediction shape {preds.shape}, expected 2D array [batch_size, pred_len]"
            )

        # Inverse transformations if datamodule supports it
        if hasattr(self.datamodule, "inverse_transform_predictions"):
            try:
                preds_flat = self.datamodule.inverse_transform_predictions(
                    preds_flat, basin_ids_expanded
                )
                obs_flat = self.datamodule.inverse_transform_predictions(
                    obs_flat, basin_ids_expanded
                )
            except Exception as e:
                print(f"Warning: Failed to inverse transform predictions: {e}")

        # Create evaluation dataframe
        df = pd.DataFrame(
            {
                "horizon": horizons_expanded,
                "prediction": preds_flat,
                "observed": obs_flat,
                "basin_id": basin_ids_expanded,
                "date": dates_expanded,
            }
        )

        return df

    def _calculate_overall_metrics(self, df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Calculate overall metrics for each forecast horizon.
        
        Args:
            df: DataFrame with predictions and observations
                
        Returns:
            Dictionary with metrics for each horizon
        """
        overall_metrics = {}
        for h in self.horizons:
            if h > max(df["horizon"]):
                print(f"Warning: Horizon {h} exceeds maximum available horizon {max(df['horizon'])}")
                continue
                
            horizon_data = df[df["horizon"] == h]
            if not horizon_data.empty:
                overall_metrics[h] = self._calculate_metrics(horizon_data)
            else:
                print(f"Warning: No data available for horizon {h}")
                overall_metrics[h] = {metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]}
        
        return overall_metrics

    def _calculate_basin_metrics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Calculate per-basin metrics for each forecast horizon.
        
        Args:
            df: DataFrame with predictions and observations
                
        Returns:
            Nested dictionary with metrics by basin and horizon
        """
        basin_metrics = {}
        for basin in df["basin_id"].unique():
            basin_metrics[basin] = {}
            basin_data = df[df["basin_id"] == basin]

            for h in self.horizons:
                if h > max(df["horizon"]):
                    continue
                    
                horizon_data = basin_data[basin_data["horizon"] == h]
                if not horizon_data.empty:
                    basin_metrics[basin][h] = self._calculate_metrics(horizon_data)
                else:
                    basin_metrics[basin][h] = {metric: np.nan for metric in ["MSE", "MAE", "NSE", "RMSE"]}
        
        return basin_metrics

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
        """Create a summary DataFrame of metrics."""
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
        df: pd.DataFrame,
        horizon: int,
        group_identifier: str,
        fig_size: tuple = (12, 6),
        title: str = None,
    ) -> tuple:
        """Create a rolling forecast plot for a specific basin and horizon."""
        # Validate horizon
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon} not in available horizons: {self.horizons}"
            )

        # Filter for the specific basin and horizon
        basin_df = df[(df["basin_id"] == group_identifier) & (df["horizon"] == horizon)]

        if basin_df.empty:
            available_ids = df["basin_id"].unique()
            raise ValueError(
                f"Group identifier '{group_identifier}' not found in test results. Available IDs: {available_ids}"
            )

        # Create plot with Seaborn style
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot observations and predictions
        ax.plot(
            basin_df["date"],
            basin_df["observed"],
            color="blue",
            label="Observed",
            linewidth=2,
        )

        ax.plot(
            basin_df["date"],
            basin_df["prediction"],
            color="red",
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

    # Static metric calculation methods
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

    def flatten_basin_metrics(self, basin_metrics: Dict) -> pd.DataFrame:
        """
        Convert nested basin metrics dictionary to a flattened DataFrame.

        Args:
            basin_metrics: Nested dictionary of metrics by basin and horizon

        Returns:
            DataFrame with columns for basin_id, horizon, and metrics
        """
        rows = []
        for basin, horizons in basin_metrics.items():
            for horizon, metrics in horizons.items():
                row = {"basin_id": basin, "horizon": horizon}
                row.update(metrics)
                rows.append(row)
        return pd.DataFrame(rows)
    
    def compare_models(
        self, 
        model1_name: str, 
        model2_name: str, 
        metric: str = "NSE", 
        threshold: float = 0.05
    ) -> Dict:
        """
        Compare two models across all horizons based on a specified metric.

        Args:
            model1_name: Name of first model (benchmark)
            model2_name: Name of second model (challenger)
            metric: Metric to compare (default: NSE)
            threshold: Threshold for significant difference (default: 0.05)

        Returns:
            Dictionary with comparison results and DataFrames
        """
        if model1_name not in self.results or model2_name not in self.results:
            missing = []
            if model1_name not in self.results:
                missing.append(model1_name)
            if model2_name not in self.results:
                missing.append(model2_name)
            raise ValueError(f"Models not found in results: {', '.join(missing)}")
            
        # Extract basin metrics for both models
        benchmark_metrics = self.results[model1_name]["basin_metrics"]
        challenger_metrics = self.results[model2_name]["basin_metrics"]
        
        # Flatten the nested dictionaries into DataFrames
        benchmark_df = self.flatten_basin_metrics(benchmark_metrics)
        challenger_df = self.flatten_basin_metrics(challenger_metrics)
        
        # Ensure metric exists in both DataFrames
        if metric not in benchmark_df.columns or metric not in challenger_df.columns:
            raise ValueError(f"Metric '{metric}' not found in model results")
        
        # Rename the metric columns to identify models after merge
        benchmark_df = benchmark_df.rename(columns={metric: f"{metric}_benchmark"})
        challenger_df = challenger_df.rename(columns={metric: f"{metric}_challenger"})
        
        # Merge DataFrames on basin_id and horizon
        comparison = pd.merge(
            challenger_df,
            benchmark_df,
            on=["basin_id", "horizon"],
            suffixes=("_challenger", "_benchmark")
        )
        
        # Calculate performance comparison by horizon
        comparison_results = self._calculate_performance_comparison(
            comparison, 
            metric_name=metric,
            threshold=threshold
        )
        
        return {
            "comparison_df": comparison_results,
            "full_comparison": comparison,
            "challenger_name": model2_name,
            "benchmark_name": model1_name,
            "metric": metric,
            "threshold": threshold
        }
    
    def _calculate_performance_comparison(
        self, 
        comparison: pd.DataFrame, 
        metric_name: str = "NSE", 
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Calculate performance comparison metrics between challenger and benchmark models.

        Args:
            comparison: DataFrame with merged model metrics
            metric_name: Name of the metric being compared
            threshold: Threshold for significant difference

        Returns:
            DataFrame with performance comparison results by horizon
        """
        # Get unique horizons
        horizons = sorted(comparison["horizon"].unique())
        
        # Initialize results dictionary
        results = {
            "horizon": [],
            "better": [],  # Percentage where challenger significantly outperforms (>= threshold)
            "insignificant": [],  # Percentage where change is insignificant (< threshold)
            "worse": [],  # Percentage where benchmark significantly outperforms (<= -threshold)
        }
        
        # For each horizon, compare metric scores
        for horizon in horizons:
            horizon_data = comparison[comparison["horizon"] == horizon]
            
            # Calculate total number of basins for this horizon
            total_basins = len(horizon_data)
            
            if total_basins == 0:
                continue
                
            # Compute difference between challenger and benchmark
            diff = horizon_data[f"{metric_name}_challenger"] - horizon_data[f"{metric_name}_benchmark"]
            
            # Count significant differences based on threshold
            challenger_wins = (diff >= threshold).sum()
            benchmark_wins = (diff <= -threshold).sum()
            insignificant = (abs(diff) < threshold).sum()
            
            # Calculate percentages
            pct_better = (challenger_wins / total_basins) * 100
            pct_insig = (insignificant / total_basins) * 100
            pct_worse = (benchmark_wins / total_basins) * 100
            
            # Store results
            results["horizon"].append(horizon)
            results["better"].append(pct_better)
            results["insignificant"].append(pct_insig)
            results["worse"].append(pct_worse)
        
        return pd.DataFrame(results)
    
    def plot_model_comparison(
        self, 
        comparison_results: Dict, 
        figsize: Tuple[int, int] = (12, 6),
        colors: Dict[str, str] = None,
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a stacked bar chart visualizing model comparison results.
        
        Args:
            comparison_results: Results from the compare_models method
            figsize: Figure size as (width, height)
            colors: Dictionary with colors for 'better', 'insignificant', and 'worse'
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure and axes object
        """
        comparison_df = comparison_results["comparison_df"]
        challenger_name = comparison_results["challenger_name"]
        benchmark_name = comparison_results["benchmark_name"]
        metric = comparison_results["metric"]
        threshold = comparison_results["threshold"]
        
        # Set default colors if not provided
        if colors is None:
            colors = {
                "better": "#6BA292",      # Pale green for challenger outperforms
                "insignificant": "#F9E79F", # Yellowish for insignificant differences
                "worse": "#93827F"       # Light pink for benchmark outperforms
            }
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the stacked bars
        ax.bar(
            comparison_df["horizon"],
            comparison_df["better"],
            label=f"{challenger_name} Outperforms",
            color=colors["better"],
        )
        ax.bar(
            comparison_df["horizon"],
            comparison_df["insignificant"],
            bottom=comparison_df["better"],
            label="Insignificant",
            color=colors["insignificant"],
        )
        ax.bar(
            comparison_df["horizon"],
            comparison_df["worse"],
            bottom=comparison_df["better"] + comparison_df["insignificant"],
            label=f"{benchmark_name} Outperforms",
            color=colors["worse"],
        )
        
        # Set labels and title
        ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
        ax.set_ylabel("Percentage of Basins", fontsize=12)
        
        # Set custom or default title
        if title is None:
            title = f"Model Comparison by {metric} (±{threshold} threshold)"
        ax.set_title(title, fontsize=14)
        
        # Add legend and styling
        ax.legend(loc="upper right")
        ax.set_ylim(0, 100)
        
        # Configure ticks
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
        ax.set_xticks(np.arange(comparison_df["horizon"].min(), comparison_df["horizon"].max() + 1, 1))
        
        # Apply seaborn style
        sns.despine()
        
        return fig, ax
    
    def summarize_comparison(self, comparison_results: Dict) -> Dict[str, float]:
        """
        Generate a text summary of model comparison results.
        
        Args:
            comparison_results: Results from the compare_models method
            
        Returns:
            Dictionary with summary statistics
        """
        comparison_df = comparison_results["comparison_df"]
        challenger_name = comparison_results["challenger_name"]
        benchmark_name = comparison_results["benchmark_name"]
        
        summary = {
            "avg_better": comparison_df["better"].mean(),
            "avg_insignificant": comparison_df["insignificant"].mean(),
            "avg_worse": comparison_df["worse"].mean(),
            "best_horizon": comparison_df.loc[comparison_df["better"].idxmax(), "horizon"],
            "best_horizon_pct": comparison_df["better"].max(),
            "worst_horizon": comparison_df.loc[comparison_df["better"].idxmin(), "horizon"],
            "worst_horizon_pct": comparison_df["better"].min()
        }
        
        return summary
    
    def analyze_basin_performance(
        self, 
        comparison_results: Dict, 
        improvement_threshold: float = 0.0
    ) -> Dict:
        """
        Analyze basin-by-basin performance across horizons.
        
        Args:
            comparison_results: Results from the compare_models method
            improvement_threshold: Threshold for considering improvement (default: 0.0)
            
        Returns:
            Dictionary with basin classifications and horizon-specific results
        """
        full_comparison = comparison_results["comparison_df"]
        metric = comparison_results["metric"]
        
        # Extract unique horizons and basin IDs
        horizons = sorted(full_comparison["horizon"].unique())
        all_basins = full_comparison["basin_id"].unique()
        
        # Initialize result containers
        consistently_better = set()
        consistently_worse = set()
        mixed_performance = set()
        horizon_results = {}
        
        # Analyze each horizon
        for horizon in horizons:
            horizon_data = full_comparison[full_comparison["horizon"] == horizon]
            
            # Skip if no data for this horizon
            if horizon_data.empty:
                continue
                
            # Calculate improvement for each basin at this horizon
            metric_diff = horizon_data[f"{metric}_challenger"] - horizon_data[f"{metric}_benchmark"]
            
            # Get better and worse basins for this horizon
            better_basins = set(horizon_data[metric_diff > improvement_threshold]["basin_id"])
            worse_basins = set(horizon_data[metric_diff <= improvement_threshold]["basin_id"])
            
            # Store results for this horizon
            horizon_results[horizon] = {
                "better_basins": better_basins,
                "worse_basins": worse_basins,
                "mean_improvement": metric_diff.mean()
            }
            
            # Update consistent performance sets
            if horizon == horizons[0]:
                consistently_better = better_basins
                consistently_worse = worse_basins
            else:
                consistently_better &= better_basins
                consistently_worse &= worse_basins
        
        # Identify basins with mixed performance
        mixed_performance = set(all_basins) - consistently_better - consistently_worse
        
        return {
            "consistently_better": consistently_better,
            "consistently_worse": consistently_worse,
            "mixed_performance": mixed_performance,
            "horizon_results": horizon_results
        }
    
    def plot_nse_comparison(
        self, 
        model1_name: str, 
        model2_name: str, 
        metric: str = "NSE", 
        figsize: Tuple[int, int] = (14, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar chart comparing average metric values across horizons.
        
        Args:
            model1_name: Name of first model (benchmark)
            model2_name: Name of second model (challenger) 
            metric: Metric to compare (default: NSE)
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure and axes object
        """
        if model1_name not in self.results or model2_name not in self.results:
            missing = []
            if model1_name not in self.results:
                missing.append(model1_name)
            if model2_name not in self.results:
                missing.append(model2_name)
            raise ValueError(f"Models not found in results: {', '.join(missing)}")
        
        # Extract overall metrics for both models
        benchmark_metrics = self.summarize_metrics(self.results[model1_name]["metrics"])
        challenger_metrics = self.summarize_metrics(self.results[model2_name]["metrics"])
        
        # Reset index to ensure horizon is a column
        benchmark_metrics = benchmark_metrics.reset_index()
        challenger_metrics = challenger_metrics.reset_index()
        
        # Merge data
        comparison = pd.merge(
            challenger_metrics,
            benchmark_metrics,
            on="horizon",
            suffixes=(f"_{model2_name}", f"_{model1_name}")
        )
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define plot style
        palette = sns.color_palette("Blues", n_colors=2)
        bar_width = 0.4
        x_pos = np.arange(len(comparison))
        
        # Create bars
        metric_col1 = f"{metric}_{model2_name}"
        metric_col2 = f"{metric}_{model1_name}"
        
        ax.bar(
            x_pos - bar_width / 2,
            comparison[metric_col1],
            width=bar_width,
            label=model2_name,
            color=palette[0],
        )
        
        ax.bar(
            x_pos + bar_width / 2,
            comparison[metric_col2],
            width=bar_width,
            label=model1_name,
            color=palette[1],
        )
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison["horizon"])
        ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
        ax.set_ylabel(f"Average {metric}", fontsize=12)
        ax.set_title(f"Model Performance Comparison by Horizon ({metric})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Add value labels
        for i, (val1, val2) in enumerate(zip(comparison[metric_col1], comparison[metric_col2])):
            ax.text(i - bar_width / 2, val1 + 0.02, f"{val1:.2f}", ha="center")
            ax.text(i + bar_width / 2, val2 + 0.02, f"{val2:.2f}", ha="center")
        
        # Set y-axis limit with some padding
        if metric == "NSE":
            ax.set_ylim(top=1.1)  # NSE can theoretically exceed 1
        else:
            y_max = max(comparison[metric_col1].max(), comparison[metric_col2].max())
            ax.set_ylim(top=y_max * 1.2)
        
        sns.despine()
        plt.tight_layout()
        
        return fig, ax

    def identify_improved_basins(
        self, 
        model1_name: str, 
        model2_name: str, 
        metric: str = "NSE", 
        threshold: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Identify basins where the challenger model improves or worsens compared to benchmark.
        
        Args:
            model1_name: Name of benchmark model
            model2_name: Name of challenger model
            metric: Metric to compare (default: NSE)
            threshold: Threshold for significant difference (default: 0.05)
            
        Returns:
            Dictionary with 'improved' and 'worsened' lists of basin IDs
        """
        if model1_name not in self.results or model2_name not in self.results:
            missing = []
            if model1_name not in self.results:
                missing.append(model1_name)
            if model2_name not in self.results:
                missing.append(model2_name)
            raise ValueError(f"Models not found in results: {', '.join(missing)}")
            
        # Extract basin metrics for both models
        benchmark_metrics = self.results[model1_name]["basin_metrics"]
        challenger_metrics = self.results[model2_name]["basin_metrics"]
        
        # Flatten the nested dictionaries into DataFrames
        benchmark_df = self.flatten_basin_metrics(benchmark_metrics)
        challenger_df = self.flatten_basin_metrics(challenger_metrics)
        
        # Merge DataFrames on basin_id and horizon
        comparison = pd.merge(
            challenger_df,
            benchmark_df,
            on=["basin_id", "horizon"],
            suffixes=("_challenger", "_benchmark")
        )
        
        # Calculate improvement for each basin-horizon pair
        comparison[f"delta_{metric}"] = comparison[f"{metric}_challenger"] - comparison[f"{metric}_benchmark"]
        
        # Group by basin_id to get average improvement across horizons
        basin_improvement = comparison.groupby("basin_id")[f"delta_{metric}"].mean().reset_index()
        
        # Identify improved and worsened basins based on threshold
        improved_basins = basin_improvement[basin_improvement[f"delta_{metric}"] >= threshold]["basin_id"].tolist()
        worsened_basins = basin_improvement[basin_improvement[f"delta_{metric}"] <= -threshold]["basin_id"].tolist()
        
        # For basins with changes within threshold, find those that are consistently better/worse
        marginal_basins = basin_improvement[
            (basin_improvement[f"delta_{metric}"] > -threshold) & 
            (basin_improvement[f"delta_{metric}"] < threshold)
        ]["basin_id"].tolist()
        
        consistent_improved = []
        consistent_worsened = []
        
        for basin in marginal_basins:
            basin_data = comparison[comparison["basin_id"] == basin]
            if (basin_data[f"delta_{metric}"] > 0).all():
                consistent_improved.append(basin)
            elif (basin_data[f"delta_{metric}"] < 0).all():
                consistent_worsened.append(basin)
        
        # Combine significantly improved with consistently improved
        all_improved = list(set(improved_basins + consistent_improved))
        all_worsened = list(set(worsened_basins + consistent_worsened))
        
        return {
            "improved": all_improved,
            "worsened": all_worsened,
            "improvement_data": basin_improvement  # Include the full data for additional analysis
        }

    def plot_delta_vs_benchmark(
        self,
        model1_name: str,
        model2_name: str,
        metric: str = "NSE",
        horizon: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8),
        alpha: float = 0.6,
        color: str = "steelblue",
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the relationship between benchmark performance and improvement.
        
        Args:
            model1_name: Name of benchmark model
            model2_name: Name of challenger model
            metric: Metric to compare (default: NSE)
            horizon: Specific forecast horizon to analyze (None for average across horizons)
            figsize: Figure size as (width, height)
            alpha: Transparency of scatter points
            color: Base color for the plot
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure and axes object
        """
        from scipy import stats
        
        if model1_name not in self.results or model2_name not in self.results:
            missing = []
            if model1_name not in self.results:
                missing.append(model1_name)
            if model2_name not in self.results:
                missing.append(model2_name)
            raise ValueError(f"Models not found in results: {', '.join(missing)}")
        
        # Extract basin metrics for both models
        benchmark_metrics = self.results[model1_name]["basin_metrics"]
        challenger_metrics = self.results[model2_name]["basin_metrics"]
        
        # Flatten the nested dictionaries into DataFrames
        benchmark_df = self.flatten_basin_metrics(benchmark_metrics)
        challenger_df = self.flatten_basin_metrics(challenger_metrics)
        
        # Merge DataFrames on basin_id and horizon
        comparison = pd.merge(
            challenger_df,
            benchmark_df,
            on=["basin_id", "horizon"],
            suffixes=("_challenger", "_benchmark")
        )
        
        # Calculate improvement
        comparison[f"delta_{metric}"] = comparison[f"{metric}_challenger"] - comparison[f"{metric}_benchmark"]
        
        # Filter by horizon if specified
        if horizon is not None:
            if horizon not in comparison["horizon"].unique():
                raise ValueError(f"Horizon {horizon} not found in data")
            plot_data = comparison[comparison["horizon"] == horizon].copy()
            horizon_label = f" (Horizon: {horizon})"
        else:
            # Group by basin_id to get average across horizons
            plot_data = comparison.groupby("basin_id").agg({
                f"{metric}_benchmark": "mean",
                f"delta_{metric}": "mean"
            }).reset_index()
            horizon_label = " (Average Across Horizons)"
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(
            plot_data[f"{metric}_benchmark"], 
            plot_data[f"delta_{metric}"]
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with regression line
        sns.regplot(
            x=f"{metric}_benchmark",
            y=f"delta_{metric}",
            data=plot_data,
            scatter_kws={"alpha": alpha, "color": color},
            line_kws={"color": "red"},
            ax=ax
        )
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        
        # Add text with correlation information
        ax.text(
            0.05, 0.95, 
            f"Correlation: {corr:.3f} (p={p_value:.4f})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        
        # Set labels and title
        ax.set_xlabel(f"Benchmark {metric}", fontsize=12)
        ax.set_ylabel(f"Δ{metric} (Challenger - Benchmark)", fontsize=12)
        
        if title is None:
            title = f"Relationship Between Benchmark Performance and Improvement{horizon_label}"
        ax.set_title(title, fontsize=14)
        
        # Add improvement zones with different background colors
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        
        # Color the background to show improvement/degradation regions
        rect_improved = plt.Rectangle(
            (xlim[0], 0), xlim[1] - xlim[0], ylim[1], 
            color='green', alpha=0.1, zorder=-100
        )
        rect_worsened = plt.Rectangle(
            (xlim[0], ylim[0]), xlim[1] - xlim[0], -ylim[0], 
            color='red', alpha=0.1, zorder=-100
        )
        ax.add_patch(rect_improved)
        ax.add_patch(rect_worsened)
        
        # Add legend for improvement zones
        improved_patch = plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.1)
        worsened_patch = plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.1)
        ax.legend(
            [improved_patch, worsened_patch], 
            ["Improvement Region", "Degradation Region"],
            loc='lower right'
        )
        
        # Apply styling
        sns.despine()
        plt.tight_layout()
        
        return fig, ax

    def plot_nse_comparison(
        self, 
        model1_name: str, 
        model2_name: str, 
        metric: str = "NSE", 
        figsize: Tuple[int, int] = (14, 6),
        plot_type: str = "bar",  # Added parameter for plot type
        violin: bool = False,    # Option to use violin plots instead of box plots
        individual_points: bool = True  # Option to show individual data points
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar chart or box plot comparing metric values across horizons.
        
        Args:
            model1_name: Name of first model (benchmark)
            model2_name: Name of second model (challenger) 
            metric: Metric to compare (default: NSE)
            figsize: Figure size as (width, height)
            plot_type: Type of plot ('bar' or 'box')
            violin: If True and plot_type='box', use violin plots instead
            individual_points: If True and plot_type='box', show individual points
                
        Returns:
            Matplotlib figure and axes object
        """
        if model1_name not in self.results or model2_name not in self.results:
            missing = []
            if model1_name not in self.results:
                missing.append(model1_name)
            if model2_name not in self.results:
                missing.append(model2_name)
            raise ValueError(f"Models not found in results: {', '.join(missing)}")
        
        # For bar plots, use overall metrics
        if plot_type == "bar":
            # Extract overall metrics for both models
            benchmark_metrics = self.summarize_metrics(self.results[model1_name]["metrics"])
            challenger_metrics = self.summarize_metrics(self.results[model2_name]["metrics"])
            
            # Reset index to ensure horizon is a column
            benchmark_metrics = benchmark_metrics.reset_index()
            challenger_metrics = challenger_metrics.reset_index()
            
            # Merge data
            comparison = pd.merge(
                challenger_metrics,
                benchmark_metrics,
                on="horizon",
                suffixes=(f"_{model2_name}", f"_{model1_name}")
            )
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize)
            
            # Define plot style
            palette = sns.color_palette("Blues", n_colors=2)
            bar_width = 0.4
            x_pos = np.arange(len(comparison))
            
            # Create bars
            metric_col1 = f"{metric}_{model2_name}"
            metric_col2 = f"{metric}_{model1_name}"
            
            ax.bar(
                x_pos - bar_width / 2,
                comparison[metric_col1],
                width=bar_width,
                label=model2_name,
                color=palette[0],
            )
            
            ax.bar(
                x_pos + bar_width / 2,
                comparison[metric_col2],
                width=bar_width,
                label=model1_name,
                color=palette[1],
            )
            
            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(comparison["horizon"])
            ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
            ax.set_ylabel(f"Average {metric}", fontsize=12)
            ax.set_title(f"Model Performance Comparison by Horizon ({metric})", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle="--")
            
            # Add value labels
            for i, (val1, val2) in enumerate(zip(comparison[metric_col1], comparison[metric_col2])):
                ax.text(i - bar_width / 2, val1 + 0.02, f"{val1:.2f}", ha="center")
                ax.text(i + bar_width / 2, val2 + 0.02, f"{val2:.2f}", ha="center")
            
            # Set y-axis limit with some padding
            if metric == "NSE":
                ax.set_ylim(top=1.1)  # NSE can theoretically exceed 1
            else:
                y_max = max(comparison[metric_col1].max(), comparison[metric_col2].max())
                ax.set_ylim(top=y_max * 1.2)
        
        # For box plots, use basin-level metrics
        else:
            # Extract basin metrics for both models
            benchmark_basin_metrics = self.results[model1_name]["basin_metrics"]
            challenger_basin_metrics = self.results[model2_name]["basin_metrics"]
            
            # Flatten the nested dictionaries into DataFrames
            benchmark_df = self.flatten_basin_metrics(benchmark_basin_metrics)
            challenger_df = self.flatten_basin_metrics(challenger_basin_metrics)
            
            # Rename columns to identify model
            benchmark_df["model"] = model1_name
            challenger_df["model"] = model2_name
            
            # Rename metric column to common name
            benchmark_df = benchmark_df.rename(columns={metric: "metric_value"})
            challenger_df = challenger_df.rename(columns={metric: "metric_value"})
            
            # Combine data
            combined_df = pd.concat([benchmark_df, challenger_df])
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Determine horizons to plot
            horizons = sorted(combined_df["horizon"].unique())
            
            # Set color palette
            palette = {"challenger": sns.color_palette("Blues")[3], 
                      "benchmark": sns.color_palette("Blues")[5]}
            model_colors = {model1_name: palette["benchmark"], model2_name: palette["challenger"]}
            
            if violin:
                # Create violin plots
                sns.violinplot(
                    x="horizon",
                    y="metric_value",
                    hue="model",
                    data=combined_df,
                    palette=model_colors,
                    split=True,
                    inner="quartile",
                    ax=ax
                )
            else:
                # Create box plots
                sns.boxplot(
                    x="horizon",
                    y="metric_value",
                    hue="model",
                    data=combined_df,
                    palette=model_colors,
                    ax=ax
                )
                
                # Add individual points if requested
                if individual_points:
                    sns.stripplot(
                        x="horizon",
                        y="metric_value",
                        hue="model",
                        data=combined_df,
                        palette=model_colors,
                        dodge=True,
                        alpha=0.3,
                        size=3,
                        ax=ax,
                        legend=False
                    )
            
            # Add median value labels to boxes/violins
            for i, horizon in enumerate(horizons):
                for j, model in enumerate([model1_name, model2_name]):
                    median = combined_df[(combined_df["horizon"] == horizon) & 
                                        (combined_df["model"] == model)]["metric_value"].median()
                    
                    # Position for label depends on whether using violins or not
                    if violin:
                        x_pos = i + (0.15 if j == 0 else -0.15)
                    else:
                        x_pos = i - 0.2 + (j * 0.4)
                        
                    ax.text(
                        x_pos, median + 0.03,
                        f"{median:.2f}",
                        ha='center', va='bottom',
                        color='black', fontweight='bold',
                        fontsize=8
                    )
            
            # Customize plot
            ax.set_xlabel("Forecast Horizon (days)", fontsize=12)
            ax.set_ylabel(f"{metric} Value", fontsize=12)
            ax.set_title(f"Distribution of {metric} Values by Horizon", fontsize=14)
            
            # Adjust legend to show model names
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], title="Model")
            
            # Add grid lines for better readability
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Apply styling to either plot type
        sns.despine()
        plt.tight_layout()
        
        return fig, ax
