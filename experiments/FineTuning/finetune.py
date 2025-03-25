"""
Fine-tuning script for pre-trained forecasting models.

This script loads a pre-trained model checkpoint and fine-tunes it on specific
Central Asian country data with a reduced learning rate.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import project modules
from experiments.FineTuning.configs.finetune_config import FineTuningConfig
from experiments.DataSharing.utils import load_country_data
from experiments.DataSharing.model_factory import ModelFactory
from src.model_evaluation.hp_from_yaml import load_model_config
from src.data_models.datamodule import HydroDataModule


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for fine-tuning.

    Returns:
        Parsed command line arguments
    """
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
        "--country",
        type=str,
        required=True,
        choices=["Tajikistan", "Kyrgyzstan", "Combined"],
        help="Country to fine-tune on",
    )

    parser.add_argument(
        "--yaml-path",
        type=str,
        required=True,
        help="Path to model hyperparameter YAML file",
    )

    # Optional arguments
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


def setup_directories(config: FineTuningConfig) -> Tuple[Path, Path]:
    """Create and return necessary directories for fine-tuning outputs.

    Args:
        config: Fine-tuning configuration

    Returns:
        Tuple of (checkpoint_dir, logs_dir)
    """
    checkpoint_dir = config.get_checkpoint_dir()
    logs_dir = config.get_logs_dir()

    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, logs_dir


def prepare_data_module(
    config: FineTuningConfig,
    model_config: Any,
) -> HydroDataModule:
    """Prepare a HydroDataModule for fine-tuning on country-specific data.

    Args:
        config: Fine-tuning configuration
        model_config: Model-specific configuration

    Returns:
        Configured HydroDataModule for the target country
    """
    # Load country-specific data
    data = load_country_data(config, config.TARGET_COUNTRY)
    time_series_data = data["time_series"]
    static_data = data["static"]

    # Get model-specific parameters
    input_length = model_config.input_len
    output_length = model_config.output_len
    batch_size = getattr(model_config, "batch_size", config.batch_size)

    # Get preprocessing config
    preprocessing_config = config.get_preprocessing_config()

    # Create DataModule with model-specific parameters
    data_module = HydroDataModule(
        time_series_df=time_series_data,
        static_df=static_data,
        group_identifier=config.GROUP_IDENTIFIER,
        preprocessing_config=preprocessing_config,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        num_workers=min(config.MAX_WORKERS, os.cpu_count()),
        features=config.FORCING_FEATURES + [config.TARGET],
        static_features=[f for f in config.STATIC_FEATURES if f != "country"],
        target=config.TARGET,
        use_proportional_split=config.USE_PROPORTIONAL_SPLIT,
        train_prop=config.TRAIN_PROP,
        val_prop=config.VAL_PROP,
        test_prop=config.TEST_PROP,
        min_train_years=config.CA_CONFIG.get("MIN_TRAIN_YEARS", 5),
    )

    # Prepare data and setup
    data_module.prepare_data()
    data_module.setup()

    print(f"Prepared data module for {config.TARGET_COUNTRY}")
    print(f"Number of basins: {data['basin_count']}")

    return data_module


def load_pretrained_model(
    config: FineTuningConfig,
    model_config: Any,
) -> pl.LightningModule:
    """Load a pre-trained model from checkpoint with adjusted learning rate.

    Args:
        config: Fine-tuning configuration
        model_config: Model-specific configuration

    Returns:
        Pre-trained model loaded from checkpoint with adjusted learning rate
    """
    # Create model instance using factory
    model = ModelFactory.create_model(model_config, config.MODEL_TYPE)

    # Load checkpoint
    checkpoint_path = config.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loading checkpoint from {checkpoint_path}")

        # Handle different checkpoint formats (PyTorch Lightning saves differently)
        if "state_dict" in checkpoint:
            # PyTorch Lightning checkpoint
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)

        print("Successfully loaded model weights from checkpoint")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Adjust learning rate
    original_lr = model.hparams.learning_rate
    new_lr = original_lr / config.LR_FACTOR
    model.hparams.learning_rate = new_lr

    print(f"Adjusted learning rate from {original_lr:.6f} to {new_lr:.6f}")
    return model


def fine_tune_model(
    model: pl.LightningModule,
    data_module: HydroDataModule,
    config: FineTuningConfig,
) -> Dict[str, Any]:
    """Fine-tune a pre-trained model on the target country data.

    Args:
        model: Pre-trained model to fine-tune
        data_module: DataModule for the target country
        config: Fine-tuning configuration

    Returns:
        Dictionary with fine-tuning results
    """
    # Setup directories
    checkpoint_dir, logs_dir = setup_directories(config)

    # Configure logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir),
        name=f"{config.TARGET_COUNTRY}_{config.MODEL_TYPE}",
        version="fine_tuned",
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="min",
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{config.TARGET_COUNTRY}_{config.MODEL_TYPE}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.SAVE_TOP_K,
            save_last=config.SAVE_LAST,
        ),
    ]

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    # Train model
    print(f"Starting fine-tuning for {config.MODEL_TYPE} on {config.TARGET_COUNTRY}")
    trainer.fit(model, data_module)

    # Get the best validation loss
    best_val_loss = trainer.callback_metrics.get(
        "val_loss", torch.tensor(float("inf"))
    ).item()

    # Create results summary
    results = {
        "model_type": config.MODEL_TYPE,
        "country": config.TARGET_COUNTRY,
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(
            checkpoint_dir
            / f"{config.TARGET_COUNTRY}_{config.MODEL_TYPE}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
        ),
    }

    print(
        f"Fine-tuning completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
    )

    return results


def save_results(results: Dict[str, Any], config: FineTuningConfig) -> None:
    """Save fine-tuning results to a CSV file.

    Args:
        results: Fine-tuning results
        config: Fine-tuning configuration
    """
    # Create results directory
    results_dir = Path(config.OUTPUT_DIR) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create results DataFrame
    results_df = pd.DataFrame([results])

    # Generate output path
    output_path = (
        results_dir / f"{config.TARGET_COUNTRY}_{config.MODEL_TYPE}_results.csv"
    )

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main() -> None:
    """Main function to run the fine-tuning process."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    pl.seed_everything(args.seed)

    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Create fine-tuning configuration
    config = FineTuningConfig()
    config.MODEL_TYPE = args.model
    config.CHECKPOINT_PATH = args.checkpoint_path
    config.TARGET_COUNTRY = args.country
    config.YAML_PATH = args.yaml_path
    config.OUTPUT_DIR = args.output_dir
    config.LR_FACTOR = args.lr_factor
    config.MAX_EPOCHS = args.epochs
    config.batch_size = args.batch_size

    # Validate configuration
    config.validate()

    try:
        # Load model configuration from YAML
        model_params = load_model_config(config.MODEL_TYPE, config.YAML_PATH)

        # Create appropriate config class
        if config.MODEL_TYPE.lower() == "tide":
            from src.models.tide import TiDEConfig

            model_config = TiDEConfig(**model_params)
        elif config.MODEL_TYPE.lower() == "tsmixer":
            from src.models.tsmixer import TSMixerConfig

            model_config = TSMixerConfig(**model_params)
        elif config.MODEL_TYPE.lower() == "ealstm":
            from src.models.ealstm import EALSTMConfig

            model_config = EALSTMConfig(**model_params)
        elif config.MODEL_TYPE.lower() == "tft":
            from src.models.tft import TFTConfig

            model_config = TFTConfig(**model_params)

        # Prepare data module
        data_module = prepare_data_module(config, model_config)

        # Load pre-trained model
        model = load_pretrained_model(config, model_config)

        # Fine-tune model
        results = fine_tune_model(model, data_module, config)

        # Save results
        save_results(results, config)

        print("Fine-tuning completed successfully!")

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise


if __name__ == "__main__":
    main()
