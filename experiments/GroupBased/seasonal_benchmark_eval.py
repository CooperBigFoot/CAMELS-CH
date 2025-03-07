import sys
from pathlib import Path
import torch
import argparse
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List


sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytorch_lightning as pl
from experiments.GroupBased.configGroupBased import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer, TSMixerConfig
from src.models.evaluators import TSForecastEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark model on growing season data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to benchmark model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./seasonal_results",
        help="Directory to save output files",
    )
    return parser.parse_args()


def load_model(model_path):
    """Load model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading benchmark model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Get TSMixer config from checkpoint
    config_dict = checkpoint.get("config", None)
    if not config_dict:
        raise ValueError(f"Could not find config in checkpoint: {model_path}")

    # Create config and model
    config = TSMixerConfig.from_dict(config_dict)
    model = LitTSMixer(config=config)

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])

    print(f"Successfully loaded model (val_loss: {checkpoint.get('val_loss', 'N/A')})")

    return model, config


def prepare_data_module(config):
    """Prepare benchmark data module with CA data only."""
    # Load Central Asia dataset directly (benchmark scenario)
    print("Preparing CA data for benchmark evaluation")
    ca_config = CaravanifyConfig(
        attributes_dir=config.CA_CONFIG["ATTRIBUTE_DIR"],
        timeseries_dir=config.CA_CONFIG["TIMESERIES_DIR"],
        gauge_id_prefix=config.CA_CONFIG["GAUGE_ID_PREFIX"],
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()
    ca_caravan.load_stations(ca_basins)

    # Prepare data frames
    ts_columns = config.FORCING_FEATURES + [config.TARGET]
    static_columns = config.STATIC_FEATURES

    ca_ts_data = ca_caravan.get_time_series()[
        ts_columns + ["date"] + [config.GROUP_IDENTIFIER]
    ]
    ca_static_data = ca_caravan.get_static_attributes()[static_columns]

    # Create data module for CA data only
    preprocessing_configs = config.get_preprocessing_config()

    # Create data module
    data_module = HydroDataModule(
        time_series_df=ca_ts_data,
        static_df=ca_static_data,
        group_identifier=config.GROUP_IDENTIFIER,
        preprocessing_config=preprocessing_configs,
        batch_size=config.BATCH_SIZE,
        input_length=config.INPUT_LENGTH,
        output_length=config.OUTPUT_LENGTH,
        num_workers=min(config.MAX_WORKERS, os.cpu_count() or 1),
        features=config.FORCING_FEATURES + [config.TARGET],
        static_features=config.STATIC_FEATURES,
        target=config.TARGET,
        min_train_years=config.CA_CONFIG["MIN_TRAIN_YEARS"],
        val_years=config.CA_CONFIG["VAL_YEARS"],
        test_years=config.CA_CONFIG["TEST_YEARS"],
        max_missing_pct=config.CA_CONFIG["MAX_MISSING_PCT"],
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup("test")

    return data_module


def filter_growing_season(results_df):
    """Filter results to include only growing season (April - October)."""
    # Get month from the date column
    results_df["month"] = pd.DatetimeIndex(results_df["date"]).month

    # Filter for growing season (April = 4, October = 10)
    seasonal_df = results_df[(results_df["month"] >= 4) & (results_df["month"] <= 10)]

    print(
        f"Filtered from {len(results_df)} to {len(seasonal_df)} points (growing season only)"
    )

    return seasonal_df


def evaluate_seasonal(model, data_module, output_dir, group_key=None):
    """Evaluate model and save seasonal basin metrics."""
    # Create test trainer
    trainer = pl.Trainer(accelerator="cpu", devices=1)

    # Run test
    print("Running test for model...")
    trainer.test(model, data_module)

    # Get results
    test_results = model.test_results

    # Initialize evaluator
    horizons = list(range(1, data_module.output_length + 1))
    evaluator = TSForecastEvaluator(data_module, horizons=horizons)

    # Calculate metrics
    results_df, overall_metrics, basin_metrics = evaluator.evaluate(test_results)

    # Add date information to results DataFrame using dataset information
    if "date" not in results_df.columns:
        # Extract basin IDs and their dates from the dataset
        basin_ids = np.array(test_results["basin_ids"]).flatten()
        
        # Creating a more robust approach to get dates
        # First try to use the dataset.index which contains input_end_date
        try:
            # Get all basin dates from the dataset
            basin_dates = {}
            for basin_id in np.unique(basin_ids):
                # Get dataset index entries for this basin
                basin_rows = data_module.test_dataset.index[
                    data_module.test_dataset.index[data_module.group_identifier] == basin_id
                ]
                
                if not basin_rows.empty:
                    # Get input_end_dates for this basin
                    input_end_dates = pd.to_datetime(basin_rows["input_end_date"].values)
                    basin_dates[basin_id] = []
                    
                    # For each input_end_date, calculate dates for all horizons
                    for end_date in input_end_dates:
                        for h in horizons:
                            forecast_date = end_date + pd.Timedelta(days=h)
                            basin_dates[basin_id].append((h, forecast_date))
            
            # Assign dates to results_df
            results_df["date"] = pd.NaT
            
            # Counter to track position in the basin's date list
            basin_counters = {basin_id: 0 for basin_id in basin_dates.keys()}
            
            for i, row in results_df.iterrows():
                basin_id = row["basin_id"]
                horizon = row["horizon"]
                
                if basin_id in basin_dates and basin_counters[basin_id] < len(basin_dates[basin_id]):
                    # Find the next matching horizon
                    dates_for_basin = basin_dates[basin_id]
                    date_index = basin_counters[basin_id]
                    
                    # Iterate until we find a matching horizon or run out of dates
                    while date_index < len(dates_for_basin):
                        if dates_for_basin[date_index][0] == horizon:
                            results_df.at[i, "date"] = dates_for_basin[date_index][1]
                            basin_counters[basin_id] = date_index + 1
                            break
                        date_index += 1
                
        except Exception as e:
            # If that doesn't work, try an alternative approach
            print(f"Warning: Error using dataset index for dates: {e}")
            print("Falling back to alternative date assignment method")
            
            # Fallback approach: use the dataset's df_sorted to reconstruct dates
            results_df["date"] = pd.NaT
            
            # Extract dates for each basin from the test dataset
            for basin_id in np.unique(basin_ids):
                basin_data = data_module.test_dataset.df_sorted[
                    data_module.test_dataset.df_sorted[data_module.group_identifier] == basin_id
                ]
                if not basin_data.empty:
                    # Get all dates for this basin (sorted)
                    dates = pd.to_datetime(basin_data["date"].sort_values().values)
                    
                    # Get basin indices in the results
                    basin_indices = np.where(basin_ids == basin_id)[0]
                    
                    # For each horizon, assign dates
                    for h in horizons:
                        horizon_mask = results_df["horizon"] == h
                        basin_horizon_indices = results_df.index[
                            horizon_mask & (results_df["basin_id"] == basin_id)
                        ]
                        
                        if len(basin_horizon_indices) > 0:
                            # Calculate the input length end dates
                            input_length = data_module.input_length
                            valid_dates = dates[input_length-1:]
                            
                            # Create forecast dates by adding horizon days
                            forecast_dates = valid_dates + pd.Timedelta(days=h)
                            
                            # Assign forecast dates to this basin/horizon combination
                            for i, idx in enumerate(basin_horizon_indices):
                                if i < len(forecast_dates):
                                    results_df.at[idx, "date"] = forecast_dates[i]

    # Filter for growing season (April-October)
    if results_df["date"].isna().all():
        print("Warning: No valid dates found, skipping seasonal filtering")
        seasonal_results = results_df
    else:
        seasonal_results = filter_growing_season(results_df)

    # Get basin_id and horizon pairs from filtered results
    basin_horizon_pairs = seasonal_results[["basin_id", "horizon"]].drop_duplicates()

    # Create list to hold seasonal basin metrics
    seasonal_basin_metrics = []

    # Calculate metrics for each basin and horizon in the growing season
    for _, row in basin_horizon_pairs.iterrows():
        basin_id = row["basin_id"]
        horizon = row["horizon"]

        # Get data for this basin and horizon in growing season
        basin_horizon_data = seasonal_results[
            (seasonal_results["basin_id"] == basin_id)
            & (seasonal_results["horizon"] == horizon)
        ]

        # Calculate metrics for this subset
        metrics = evaluator._calculate_metrics(basin_horizon_data)

        # Add to results
        seasonal_basin_metrics.append(
            {"basin_id": basin_id, "horizon": horizon, **metrics}
        )

    # Create DataFrame with seasonal metrics
    seasonal_metrics_df = pd.DataFrame(seasonal_basin_metrics)

    # Save results - determine filename based on input
    filename = "seasonal_basin_metrics_benchmark.csv"
    if group_key:
        filename = f"seasonal_basin_metrics_{group_key}.csv"

    # Save to CSV
    seasonal_metrics_df.to_csv(f"{output_dir}/{filename}", index=False)

    print(f"Saved seasonal metrics to {output_dir}/{filename}")

    return seasonal_metrics_df


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = ExperimentConfig()

    # Load benchmark model
    benchmark_model, benchmark_config = load_model(args.model_path)

    # Prepare data module
    data_module = prepare_data_module(config)

    # Evaluate seasonal performance
    evaluate_seasonal(benchmark_model, data_module, output_dir)

    print("Benchmark evaluation complete!")


if __name__ == "__main__":
    main()