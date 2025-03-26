"""
Main script for running the quantile mapping experiment.

This experiment evaluates whether using quantile mapped meteorological forcing data
improves hydrological model performance compared to the original forcing data. Both
data sources use the same reduced feature set containing only temperature_2m_mean and
total_precipitation_sum.
"""

import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.QuantileMapping.configs.qm_config import QuantileMappingConfig
from experiments.QuantileMapping.data_loader import load_data_by_source
from experiments.DataSharingOld.utils import (
    setup_dirs,
    train_and_save_model,
    save_experiment_results,
    load_model_configs_and_datamodules,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantile Mapping Experiment")

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to evaluate",
    )

    # YAML paths for hyperparameters
    parser.add_argument(
        "--tide-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/QuantileMapping/yaml_files/tide.yaml",
        help="Path to TiDE hyperparameter YAML",
    )

    parser.add_argument(
        "--tsmixer-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/QuantileMapping/yaml_files/tsmixer.yaml",
        help="Path to TSMixer hyperparameter YAML",
    )

    parser.add_argument(
        "--ealstm-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/QuantileMapping/yaml_files/ealstm.yaml",
        help="Path to EALSTM hyperparameter YAML",
    )

    parser.add_argument(
        "--tft-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/QuantileMapping/yaml_files/tft.yaml",
        help="Path to TFT hyperparameter YAML",
    )

    # Data source selection
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["original", "quantile_mapped"],
        default="original",
        help="Data source to use (original or quantile mapped)",
    )

    parser.add_argument(
        "--quantile-mapped-folder",
        type=str,
        required=False,
        help="Path to folder containing quantile mapped timeseries data (required if data-source is 'quantile_mapped')",
    )

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/QuantileMapping/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of runs for each model"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )

    # Optional feature override
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Override default forcing features (defaults to temperature_2m_mean and total_precipitation_sum)",
    )

    return parser.parse_args()


def main():
    """Run the quantile mapping experiment."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    pl.seed_everything(args.seed)

    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Setup directories
    setup_dirs(args.output_dir)

    # Load base configuration
    config = QuantileMappingConfig()
    config.NUM_RUNS = args.num_runs
    config.OUTPUT_DIR = args.output_dir
    config.DATA_SOURCE = args.data_source
    config.QUANTILE_MAPPED_FOLDER = args.quantile_mapped_folder

    # Override forcing features if specified
    if args.features:
        config.FORCING_FEATURES = args.features

    # Validate configuration
    config.validate()

    # Define YAML paths for hyperparameters
    yaml_paths = {
        "tide": args.tide_yaml,
        "tft": args.tft_yaml,
        "ealstm": args.ealstm_yaml,
        "tsmixer": args.tsmixer_yaml,
    }

    # Filter yaml_paths to only include requested models
    yaml_paths = {k: v for k, v in yaml_paths.items() if k in args.models}

    # Generate a descriptive name for the data source
    data_source_name = args.data_source.lower()
    print(f"\n{'=' * 50}")
    print(f"Processing {data_source_name} data")
    print(f"{'=' * 50}\n")

    # Load data based on specified source
    try:
        data = load_data_by_source(
            config=config,
            data_source=args.data_source,
            quantile_mapped_folder=args.quantile_mapped_folder,
        )

        # Load model configurations and create DataModules
        model_configs, data_modules = load_model_configs_and_datamodules(
            time_series_data=data["time_series"],
            static_data=data["static"],
            config=config,
            yaml_paths=yaml_paths,
        )

        # Store all results
        all_results = {}

        # Train each model type using its specific DataModule
        for model_type in args.models:
            if model_type not in model_configs:
                print(f"Skipping unknown model type: {model_type}")
                continue

            print(f"\nTraining {model_type.upper()} model on {data_source_name} data")

            # Get model-specific configuration and DataModule
            model_config = model_configs[model_type]
            data_module = data_modules[model_type]

            # Create and train the model
            model_results = train_and_save_model(
                model_type=model_type,
                model_config=model_config,
                data_module=data_module,
                country=data_source_name,  # Use data source name instead of country
                output_dir=args.output_dir,
                num_runs=args.num_runs,
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                save_top_k=config.SAVE_TOP_K,
                save_last=config.SAVE_LAST,
            )

            # Store results
            all_results[model_type] = model_results

        # Save results
        save_experiment_results(all_results, args.output_dir, data_source_name)

    except Exception as e:
        print(f"Error processing {data_source_name} data: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
