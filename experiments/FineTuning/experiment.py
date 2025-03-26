"""
Main script for running fine-tuning experiments on pre-trained hydrological models.

This experiment loads pre-trained models and fine-tunes them with a reduced
learning rate on a specific dataset (e.g., country-specific Central Asian data).
"""

import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.model_factory import load_pretrained_model
from experiments.FineTuning.config import ExperimentConfig
from experiments.FineTuning.utils import (
    prepare_data_module,
    fine_tune_model,
    save_results,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained hydrological models"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tide", "tsmixer", "ealstm", "tft"],
        help="Type of model to fine-tune",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint",
    )

    parser.add_argument(
        "--yaml-path",
        type=str,
        required=True,
        help="Path to model hyperparameter YAML file",
    )

    # Optional arguments
    parser.add_argument(
        "--country",
        type=str,
        choices=["Tajikistan", "Kyrgyzstan", "Combined"],
        help="Country to fine-tune on (if not specified, uses all data)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/FineTuning/results",
        help="Directory to save fine-tuned checkpoints",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=10.0,
        help="Factor to reduce learning rate by (default: 10)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum fine-tuning epochs (default: 100)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for fine-tuning (default: 2048)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Run the fine-tuning experiment."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    pl.seed_everything(args.seed)

    # Set CUDA precision if available
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Load experiment configuration
    config = ExperimentConfig()

    # Update config with command line arguments
    config.model_type = args.model
    config.checkpoint_path = args.checkpoint_path
    config.yaml_path = args.yaml_path
    config.target_country = args.country
    config.output_dir = args.output_dir
    config.lr_factor = args.lr_factor
    config.max_epochs = args.epochs
    config.batch_size = args.batch_size

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    try:
        # Load pre-trained model
        print(f"Loading pre-trained model from {config.checkpoint_path}")
        model, model_hp = load_pretrained_model(
            model_type=config.model_type,
            yaml_path=config.yaml_path,
            checkpoint_path=config.checkpoint_path,
            lr_factor=config.lr_factor,
        )
        print("Model loaded successfully")

        # Prepare data module
        data_module = prepare_data_module(config, model_hp)

        print(f"The new learning rate is: {model_hp['learning_rate']}")

        # Fine-tune model
        results = fine_tune_model(model, model_hp, data_module, config)

        # Save results
        save_results(results, config)

        print("Fine-tuning completed successfully!")

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
