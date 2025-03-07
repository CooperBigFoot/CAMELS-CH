import sys
from pathlib import Path
import torch
import argparse
import os
import numpy as np
import pandas as pd


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
        description="Evaluate challenger model on growing season data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to challenger model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./seasonal_results",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Group ID to evaluate (e.g., 'group1', 'group2')",
    )
    return parser.parse_args()


def load_model(model_path):
    """Load model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading challenger model from {model_path}")

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


def prepare_data_module(config, group_key):
    """Prepare data module for specific group."""
    print(f"Preparing data for group: {group_key}")

    # Get data for specific group
    group_data = config.load_data_for_group(group_key)

    # Extract CA data only
    ca_ts_data = group_data["ca_ts_data"]
    ca_static_data = group_data["ca_static_data"]

    # Create data module for CA data
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
        domain_id=f"group_{group_key}",
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


def evaluate_seasonal(model, data_module, output_dir):
    """Evaluate model and save seasonal basin metrics."""
    # Create test trainer
    trainer = pl.Trainer(accelerator="cpu", devices=1)

    # Run test
    print("Running test for benchmark model...")
    trainer.test(model, data_module)

    # Get results
    test_results = model.test_results

    # Initialize evaluator
    horizons = list(range(1, data_module.output_length + 1))
    evaluator = TSForecastEvaluator(data_module, horizons=horizons)

    # Calculate metrics
    results_df, overall_metrics, basin_metrics = evaluator.evaluate(test_results)

    # Add date information to results DataFrame
    basin_ids = results_df["basin_id"].unique()
    date_map = {}

    # Extract dates for each basin from the test dataset
    for basin_id in basin_ids:
        basin_data = data_module.test_dataset.df_sorted[
            data_module.test_dataset.df_sorted[data_module.group_identifier] == basin_id
        ]
        if not basin_data.empty:
            # Get all input window end dates (sorted)
            dates = basin_data["date"].sort_values().values
            input_length = data_module.input_length
            output_length = data_module.output_length

            # Calculate target dates for each horizon
            target_dates = []
            for i in range(len(dates) - input_length + 1):
                end_date = dates[i + input_length - 1]
                for h in range(1, output_length + 1):
                    target_date = end_date + pd.Timedelta(days=h)
                    target_dates.append(target_date)

            date_map[basin_id] = target_dates

    # Assign target dates to results_df
    results_df["date"] = pd.NaT
    for basin_id, dates in date_map.items():
        mask = results_df["basin_id"] == basin_id
        results_df.loc[mask, "date"] = dates

    # Filter for growing season (April-October)
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

    # Save results - use same format as regular basin_metrics files
    seasonal_metrics_df.to_csv(
        f"{output_dir}/seasonal_basin_metrics_benchmark.csv", index=False
    )

    print(f"Saved seasonal benchmark basin metrics to {output_dir}")

    return seasonal_metrics_df


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = ExperimentConfig()

    # Load challenger model
    challenger_model, challenger_config = load_model(args.model_path)

    # Prepare data module for specific group
    data_module = prepare_data_module(config, args.group)

    # Evaluate seasonal performance
    evaluate_seasonal(challenger_model, data_module, output_dir, args.group)

    print("Challenger evaluation complete!")


if __name__ == "__main__":
    main()
