"""
Main script for running the Central Asian data sharing experiment.

This experiment evaluates the impact of data sharing between Tajikistan and Kyrgyzstan
on hydrological model performance, comparing models trained on individual country
data versus combined data.
"""

import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.DataSharingOld.configs.experiment_config import ExperimentConfig
from experiments.DataSharingOld.utils import (
    setup_dirs,
    prepare_country_scenario,
    train_and_save_model,
    save_experiment_results,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Central Asian Data Sharing Experiment"
    )

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
        default="/workspace/CAMELS-CH/experiments/DataSharing/yaml_files/tide.yaml",
        help="Path to TiDE hyperparameter YAML",
    )

    parser.add_argument(
        "--tsmixer-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/DataSharing/yaml_files/tsmixer.yaml",
        help="Path to TSMixer hyperparameter YAML",
    )

    parser.add_argument(
        "--ealstm-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/DataSharing/yaml_files/ealstm.yaml",
        help="Path to EALSTM hyperparameter YAML",
    )

    parser.add_argument(
        "--tft-yaml",
        type=str,
        default="/workspace/CAMELS-CH/experiments/DataSharing/yaml_files/tft.yaml",
        help="Path to TFT hyperparameter YAML",
    )

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/DataSharing",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs for each model/country combination",
    )

    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=["Tajikistan", "Kyrgyzstan", "Combined"],
        help="Countries to include in the experiment",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Run the data sharing experiment."""
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
    config = ExperimentConfig()
    config.NUM_RUNS = args.num_runs
    config.OUTPUT_DIR = args.output_dir

    # Define YAML paths for hyperparameters
    yaml_paths = {
        "tide": args.tide_yaml,
        "tft": args.tft_yaml,
        "ealstm": args.ealstm_yaml,
        "tsmixer": args.tsmixer_yaml,
    }

    # Filter yaml_paths to only include requested models
    yaml_paths = {k: v for k, v in yaml_paths.items() if k in args.models}

    # Define country scenarios
    countries = args.countries

    # Store all results
    all_results = {}

    # Process each country scenario
    for country in countries:
        print(f"\n{'=' * 50}")
        print(f"Processing {country} scenario")
        print(f"{'=' * 50}\n")

        country_results = {}

        # Prepare data and models for this country
        try:
            scenario = prepare_country_scenario(country, config, yaml_paths)

            # Train each model type using its specific DataModule
            for model_type in args.models:
                if model_type not in scenario["model_configs"]:
                    print(f"Skipping unknown model type: {model_type}")
                    continue

                print(f"\nTraining {model_type.upper()} model on {country} data")

                # Get model-specific configuration and DataModule
                model_config = scenario["model_configs"][model_type]
                data_module = scenario["data_modules"][model_type]

                # Create and train the model
                model_results = train_and_save_model(
                    model_type=model_type,
                    model_config=model_config,
                    data_module=data_module,
                    country=country,
                    output_dir=args.output_dir,
                    num_runs=args.num_runs,
                    early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                    save_top_k=config.SAVE_TOP_K,
                    save_last=config.SAVE_LAST,
                )

                # Store results
                country_results[model_type] = model_results

        except Exception as e:
            print(f"Error processing {country} scenario: {str(e)}")
            # Continue with next country
            continue

        # Save country-specific results
        all_results[country] = country_results
        save_experiment_results(country_results, args.output_dir, country)

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
