"""
Main script for running the similarity-based transfer learning experiment.

This experiment evaluates transfer learning from data-rich regions (CH, USA, CL)
to data-sparse Central Asian catchments by grouping basins based on hydrological
similarity.
"""

import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import ExperimentConfig
from data_loader import extract_source_basins_for_training
from utils import (
    setup_dirs,
    train_model,
    save_experiment_results,
    load_model_configs_and_datamodules,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Similarity-Based Transfer Learning Experiment"
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to evaluate",
    )

    # Group selection
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["group1"],
        help="Groups to process",
    )

    # YAML paths for hyperparameters
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="experiments/SimilarityBasedTransfer/yaml_files",
        help="Directory containing model YAML files",
    )

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/SimilarityBasedTransfer/output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs for each model/group combination",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )

    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum number of training epochs"
    )

    return parser.parse_args()


def main():
    """Run the similarity-based transfer learning experiment."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    pl.seed_everything(args.seed)

    # Set CUDA precision if available
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Load configuration
    config = ExperimentConfig()
    config.model_types = args.models
    config.target_groups = args.groups
    config.output_dir = args.output_dir
    config.num_runs = args.num_runs

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Setup directories
    setup_dirs(config)

    # Prepare YAML paths for models
    yaml_dir = Path(args.yaml_dir)
    yaml_paths = {
        model_type: str(yaml_dir / f"{model_type}.yaml") for model_type in args.models
    }

    # Check that YAML files exist
    for model_type, yaml_path in yaml_paths.items():
        if not Path(yaml_path).exists():
            print(f"YAML file for {model_type} not found: {yaml_path}")
            sys.exit(1)

    # Print experiment summary
    print(f"\n{'=' * 50}")
    print("Starting Similarity-Based Transfer Learning Experiment")
    print(f"{'=' * 50}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Groups: {', '.join(args.groups)}")
    print(f"Number of runs: {args.num_runs}")
    print(
        f"Data split: {config.train_prop:.2f}/{config.val_prop:.2f}/{config.test_prop:.2f}"
    )
    print(f"{'=' * 50}\n")

    # Extract training data mapping to validate groups
    training_data = extract_source_basins_for_training(config)

    # Process each group
    for group_key in args.groups:
        if group_key not in training_data:
            print(f"Group '{group_key}' not found in training data or has no basins")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing {group_key}")
        print(f"{'=' * 50}\n")

        try:
            # Load model configurations and create DataModules for this group
            model_configs, data_modules = load_model_configs_and_datamodules(
                group_key=group_key,
                config=config,
                yaml_paths=yaml_paths,
            )

            # List to store results for this group
            group_results = []

            # Train each model type for this group
            for model_type in args.models:
                if model_type not in model_configs:
                    print(f"Skipping unknown model type: {model_type}")
                    continue

                print(f"\n{'-' * 40}")
                print(f"Training {model_type.upper()} models for {group_key}")
                print(f"{'-' * 40}\n")

                # Get data module for this model
                data_module = data_modules[model_type]

                # Run multiple training runs
                for run_idx in range(args.num_runs):
                    print(f"\nStarting run {run_idx + 1}/{args.num_runs}")

                    # Set seed for this run
                    run_seed = args.seed + run_idx
                    config.set_seed(run_seed)

                    # Train model
                    try:
                        run_results = train_model(
                            model_type=model_type,
                            yaml_path=yaml_paths[model_type],
                            data_module=data_module,
                            group=group_key,
                            config=config,
                            run_idx=run_idx,
                        )

                        # Add results to group results
                        group_results.append(run_results)

                    except Exception as e:
                        print(
                            f"Error in training run {run_idx} for {model_type} on {group_key}: {str(e)}"
                        )
                        import traceback

                        traceback.print_exc()
                        continue

                    # Clean up CUDA memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save results for this group
            save_experiment_results(group_results, config, group_key)

        except Exception as e:
            print(f"Error processing group {group_key}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
