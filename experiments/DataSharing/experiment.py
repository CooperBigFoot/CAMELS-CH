"""
Main script for running the Central Asian data sharing experiment.

This experiment evaluates the impact of data sharing between Tajikistan and Kyrgyzstan
on hydrological model performance, comparing models trained on individual country
data versus combined data.
"""

import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

from config import ExperimentConfig
from data_loader import load_data
from utils import (
    setup_dirs,
    create_data_module,
    train_model,
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
        "--yaml-dir",
        type=str,
        default="experiments/DataSharing/yaml_files",
        help="Directory containing model YAML files",
    )

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/DataSharing/output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training (overrides config)",
    )

    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum number of training epochs"
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

    # Load configuration
    config = ExperimentConfig()
    config.model_types = args.models
    config.countries = args.countries
    config.output_dir = args.output_dir
    config.num_runs = args.num_runs
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs

    # Validate configuration
    config.validate()

    # Setup directories
    setup_dirs(config)

    # Prepare YAML paths
    yaml_dir = Path(args.yaml_dir)
    yaml_paths = {
        model_type: str(yaml_dir / f"{model_type}.yaml") for model_type in args.models
    }

    # Check that YAML files exist
    for model_type, yaml_path in yaml_paths.items():
        if not Path(yaml_path).exists():
            raise FileNotFoundError(
                f"YAML file for {model_type} not found: {yaml_path}"
            )

    # Store all results
    all_results = []

    # Process each country scenario
    for country in args.countries:
        print(f"\n{'=' * 50}")
        print(f"Processing {country} scenario")
        print(f"{'=' * 50}\n")

        # Load country-specific data
        try:
            data = load_data(config, country=country)
            time_series_data = data["time_series"]
            static_data = data["static"]

            # Process each model type
            for model_type in args.models:
                print(f"\n{'-' * 40}")
                print(f"Processing {model_type.upper()} model for {country}")
                print(f"{'-' * 40}\n")

                try:
                    # Get data module for this model
                    data_module, _ = create_data_module(
                        time_series_data=time_series_data,
                        static_data=static_data,
                        model_type=model_type,
                        yaml_path=yaml_paths[model_type],
                        config=config,
                    )

                    # Prepare data
                    data_module.prepare_data()
                    data_module.setup()

                    # Run multiple training runs
                    for run_idx in range(args.num_runs):
                        print(
                            f"\nStarting run {run_idx + 1}/{args.num_runs} for {model_type} on {country} data"
                        )

                        # Set seed for this run
                        run_seed = args.seed + run_idx
                        config.set_seed(run_seed)

                        # Train model
                        try:
                            run_results = train_model(
                                model_type=model_type,
                                yaml_path=yaml_paths[model_type],
                                data_module=data_module,
                                country=country,
                                config=config,
                                run_idx=run_idx,
                            )

                            # Add to results
                            all_results.append(run_results)
                        except Exception as e:
                            print(
                                f"Error in training run {run_idx} for {model_type} on {country}: {str(e)}"
                            )
                            continue

                        # Clean up CUDA memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(
                        f"Error processing model {model_type} for {country}: {str(e)}"
                    )
                    continue

        except Exception as e:
            print(f"Error processing country {country}: {str(e)}")
            continue

    # Save all results
    if all_results:
        save_experiment_results(all_results, config)

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
