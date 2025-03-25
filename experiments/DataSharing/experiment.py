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
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import framework utilities
from src.experiment_framework.utils import (
    setup_seeds,
    train_model,
    setup_dirs,
    create_experiment_parser,
)

# Import experiment-specific modules
from experiments.DataSharing.configs import ExperimentConfig
from experiments.DataSharing.data_loader import load_data
from experiments.DataSharing.utils import (
    create_country_run_name,
    load_model_configs_and_datamodules,  # Import the experiment-specific utility
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with framework defaults plus experiment specifics."""
    # Get the framework's standard parser
    parser = create_experiment_parser()

    # Add experiment-specific arguments
    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=["Tajikistan", "Kyrgyzstan", "Combined"],
        help="Countries to include in the experiment",
    )

    return parser.parse_args()


def main():
    """Run the data sharing experiment."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    setup_seeds(args.seed)

    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Create experiment configuration with timestamp in name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.exp_name}_{timestamp}"

    # Print experiment banner
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"{'=' * 60}")

    # Setup directories
    dirs = setup_dirs(args.output_dir, exp_name)

    # Create country-specific directories
    for country in args.countries:
        country_lower = country.lower()
        for model_type in args.models:
            # Create country-specific checkpoint and log directories
            (dirs["checkpoints"] / country_lower / model_type).mkdir(
                parents=True, exist_ok=True
            )
            (dirs["logs"] / country_lower / model_type).mkdir(
                parents=True, exist_ok=True
            )

    # Load base configuration
    config = ExperimentConfig()
    config.exp_name = exp_name
    config.model_types = args.models
    config.num_runs = args.num_runs
    config.output_dir = args.output_dir
    config.checkpoint_path = args.checkpoint_path
    config.finetune = args.finetune
    config.lr_factor = args.lr_factor
    config.reset_optimizer = args.reset_optimizer
    config.countries = args.countries
    config.max_epochs = args.max_epochs
    config.batch_size = args.batch_size

    # Define YAML paths for hyperparameters
    yaml_paths = {
        "tide": args.tide_yaml,
        "tft": args.tft_yaml,
        "ealstm": args.ealstm_yaml,
        "tsmixer": args.tsmixer_yaml,
    }

    # Filter yaml_paths to only include requested models
    yaml_paths = {k: v for k, v in yaml_paths.items() if k in args.models}

    # Process each country scenario
    for country in args.countries:
        logger.info(f"{'=' * 50}")
        logger.info(f"Processing {country} scenario")
        logger.info(f"{'=' * 50}")

        try:
            # Load data for this country using the standardized interface
            data = load_data(config, country=country)

            # Create country-specific run name
            country_run_name = create_country_run_name(country, exp_name)

            # Use experiment-specific utility to load model configs and datamodules
            model_configs, data_modules = load_model_configs_and_datamodules(
                time_series_data=data["time_series"],
                static_data=data["static"],
                config=config,
                yaml_paths=yaml_paths,
            )

            # Train each model type
            for model_type in args.models:
                if model_type not in model_configs:
                    logger.warning(f"Skipping unknown model type: {model_type}")
                    continue

                logger.info(f"Setting up {model_type.upper()} model for {country} data")

                # Get model-specific configuration and data module
                model_config = model_configs[model_type]
                data_module = data_modules[model_type]

                # Train the model using framework utility
                train_model(
                    model_type=model_type,
                    model_config=model_config,
                    data_module=data_module,
                    exp_name=country_run_name,
                    output_dir=args.output_dir,
                    checkpoint_path=args.checkpoint_path,
                    finetune=args.finetune,
                    lr_factor=args.lr_factor,
                    reset_optimizer=args.reset_optimizer,
                    num_runs=args.num_runs,
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_min_delta=config.early_stopping_min_delta,
                    save_top_k=config.save_top_k,
                    save_last=config.save_last,
                    seed=args.seed,
                )

        except Exception as e:
            logger.error(
                f"Error processing {country} scenario: {str(e)}", exc_info=True
            )
            # Continue with next country
            continue

    logger.info("\nExperiment completed!")


if __name__ == "__main__":
    main()
