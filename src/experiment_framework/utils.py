"""Utility functions for hydrological forecasting experiments."""

import os
import json
import random
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Configure logging
logger = logging.getLogger(__name__)


def setup_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def setup_dirs(output_dir: str, exp_name: Optional[str] = None) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.

    Args:
        output_dir: Base output directory path
        exp_name: Optional experiment name for subdirectories

    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path(output_dir)

    # Define directory structure
    dirs = {
        "base": base_dir,
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }

    # Create base directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # If experiment name is provided, create model-specific directories
    if exp_name:
        model_types = ["tide", "tsmixer", "ealstm", "tft"]

        # Create subdirectories for each model type
        for model_type in model_types:
            (dirs["checkpoints"] / model_type / exp_name).mkdir(
                parents=True, exist_ok=True
            )
            (dirs["logs"] / model_type / exp_name).mkdir(parents=True, exist_ok=True)
            (dirs["results"] / model_type / exp_name).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created directory structure in {base_dir}")
    return dirs


def setup_logging(
    log_dir: Union[str, Path], name: str, level: int = logging.INFO
) -> None:
    """Set up logging for the experiment.

    Args:
        log_dir: Directory for log files
        name: Name for the log file
        level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create file handler
    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")


def train_model(
    model_type: str,
    model_config: Any,
    data_module: Any,
    exp_name: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    finetune: bool = False,
    lr_factor: float = 10.0,
    reset_optimizer: bool = False,
    num_runs: int = 1,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0001,
    save_top_k: int = 1,
    save_last: bool = True,
    seed: Optional[int] = None,
    enable_progress_bar: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a model and save checkpoints with support for multiple runs.

    Args:
        model_type: Type of model to train
        model_config: Configuration for the model
        data_module: DataModule for the model
        exp_name: Experiment name for output
        output_dir: Base output directory
        checkpoint_path: Optional path to pre-trained model checkpoint
        finetune: Whether to fine-tune the model with reduced learning rate
        lr_factor: Factor to reduce learning rate by when fine-tuning
        reset_optimizer: Whether to reset optimizer when loading from checkpoint
        num_runs: Number of training runs with different seeds
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum change for early stopping
        save_top_k: Number of best models to save
        save_last: Whether to save the last model checkpoint
        seed: Base random seed (incremented for each run)
        enable_progress_bar: Whether to show progress bar during training
        **kwargs: Additional arguments to pass to Trainer

    Returns:
        Dictionary with training results
    """
    from src.experiment_framework.model_utils import create_model, load_pretrained_model

    # Prepare output directories
    dirs = setup_dirs(output_dir, exp_name)
    checkpoint_dir = dirs["checkpoints"] / model_type / exp_name
    logs_dir = dirs["logs"] / model_type / exp_name

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Results dictionary
    results = {"runs": [], "best_val_loss": float("inf"), "best_run": None}

    # Base seed
    base_seed = seed if seed is not None else 42

    # Train for multiple runs
    for run in range(num_runs):
        run_seed = base_seed + run
        logger.info(
            f"Starting run {run + 1}/{num_runs} for {model_type} with seed {run_seed}"
        )

        # Set seed for reproducibility
        setup_seeds(run_seed)

        # Create or load model
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load pre-trained model
            model = load_pretrained_model(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                model_config=model_config,
                finetune=finetune,
                lr_factor=lr_factor,
                reset_optimizer=reset_optimizer,
            )
            logger.info(f"Loaded pre-trained model from {checkpoint_path}")

            if finetune:
                logger.info(
                    f"Fine-tuning with learning rate reduced from "
                    f"{model.original_lr:.6f} to {model.fine_tuned_lr:.6f}"
                )
        else:
            # Create new model
            model = create_model(model_type, model_config)
            logger.info(f"Created new {model_type} model")

        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=str(logs_dir), name=f"{model_type}", version=f"run_{run}"
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=str(checkpoint_dir / f"run_{run}"),
                filename=f"{model_type}_run{run}_{{epoch:02d}}_{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=save_top_k,
                save_last=save_last,
            ),
        ]

        # LR verification callback for fine-tuning
        if finetune and hasattr(model, "fine_tuned_lr"):

            class LRVerificationCallback(pl.Callback):
                def on_train_start(self, trainer, pl_module):
                    optimizer = trainer.optimizers[0]
                    actual_lr = optimizer.param_groups[0]["lr"]
                    expected_lr = model.fine_tuned_lr

                    if abs(actual_lr - expected_lr) > 1e-6:
                        logger.warning(
                            f"Actual learning rate ({actual_lr:.8f}) "
                            f"differs from expected ({expected_lr:.8f})"
                        )
                        # Force correct learning rate
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = expected_lr
                        logger.info(f"Learning rate corrected to {expected_lr:.8f}")
                    else:
                        logger.info(f"Verified learning rate: {actual_lr:.8f}")

            callbacks.append(LRVerificationCallback())

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=kwargs.get("max_epochs", 100),
            accelerator=kwargs.get("accelerator", "auto"),
            devices=kwargs.get("devices", 1),
            logger=tb_logger,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["max_epochs", "accelerator", "devices"]
            },
        )

        # Train model
        try:
            trainer.fit(model, data_module)

            # Get the best validation loss
            best_val_loss = trainer.callback_metrics.get(
                "val_loss", torch.tensor(float("inf"))
            ).item()

            # Find best checkpoint path
            best_model_path = None
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint) and hasattr(
                    callback, "best_model_path"
                ):
                    best_model_path = callback.best_model_path
                    break

            # Store run results
            run_results = {
                "run": run,
                "seed": run_seed,
                "best_val_loss": best_val_loss,
                "best_epoch": trainer.current_epoch,
                "checkpoint_path": best_model_path,
                "is_fine_tuned": getattr(model, "is_fine_tuned", False),
            }

            # Add learning rate information if fine-tuned
            if getattr(model, "is_fine_tuned", False):
                run_results["original_lr"] = getattr(model, "original_lr", None)
                run_results["fine_tuned_lr"] = getattr(model, "fine_tuned_lr", None)

            results["runs"].append(run_results)

            # Update best run if needed
            if best_val_loss < results["best_val_loss"]:
                results["best_val_loss"] = best_val_loss
                results["best_run"] = run_results

            logger.info(
                f"Run {run + 1} completed with best val_loss: {best_val_loss:.6f} "
                f"at epoch {trainer.current_epoch}"
            )

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            # Log the error and continue with the next run
            results["runs"].append({"run": run, "seed": run_seed, "error": str(e)})

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results summary
    results_dir = dirs["results"] / model_type / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{model_type}_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def save_experiment_results(
    results: Dict[str, Any], output_dir: str, exp_name: str
) -> None:
    """Save experiment results to CSV and JSON files.

    Args:
        results: Dictionary with results for different model types
        output_dir: Base output directory
        exp_name: Experiment name for output files
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / "results" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create summary dataframe
    summary_rows = []

    for model_type, model_results in results.items():
        if "best_run" in model_results and model_results["best_run"]:
            best_run = model_results["best_run"]
            summary_rows.append(
                {
                    "model_type": model_type,
                    "experiment": exp_name,
                    "best_val_loss": best_run["best_val_loss"],
                    "best_epoch": best_run["best_epoch"],
                    "checkpoint_path": best_run["checkpoint_path"],
                    "is_fine_tuned": best_run.get("is_fine_tuned", False),
                }
            )

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved experiment summary to {summary_path}")

    # Save detailed results for each model
    for model_type, model_results in results.items():
        if "runs" in model_results and model_results["runs"]:
            # Save as CSV
            runs_df = pd.DataFrame(model_results["runs"])
            csv_path = results_dir / f"{model_type}_runs.csv"
            runs_df.to_csv(csv_path, index=False)

            # Save as JSON
            json_path = results_dir / f"{model_type}_results.json"
            with open(json_path, "w") as f:
                json.dump(model_results, f, indent=2)

            logger.info(f"Saved {model_type} results to {csv_path} and {json_path}")


def create_experiment_parser():
    """Create an argument parser with standard experiment arguments.

    Returns:
        ArgumentParser with standard arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Hydrological Forecasting Experiment")

    # Common arguments across all experiments
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (used for logging and checkpoints)",
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to evaluate",
    )

    # Training parameters
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Number of runs for each model"
    )

    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Maximum training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Base output directory",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )

    # Checkpoint loading arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint",
    )

    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Reset optimizer when loading from checkpoint",
    )

    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune pre-trained model with reduced learning rate",
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=10.0,
        help="Factor to reduce learning rate by when fine-tuning",
    )

    # YAML paths arguments
    parser.add_argument(
        "--tide-yaml", type=str, help="Path to TiDE hyperparameter YAML"
    )

    parser.add_argument(
        "--tsmixer-yaml", type=str, help="Path to TSMixer hyperparameter YAML"
    )

    parser.add_argument(
        "--ealstm-yaml", type=str, help="Path to EALSTM hyperparameter YAML"
    )

    parser.add_argument("--tft-yaml", type=str, help="Path to TFT hyperparameter YAML")

    return parser


def prepare_model_scenario(
    time_series_data: pd.DataFrame,
    static_data: pd.DataFrame,
    config: Any,
    yaml_paths: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Prepare model configs and DataModules for an experiment scenario.

    Args:
        time_series_data: Time series DataFrame
        static_data: Static attributes DataFrame
        config: Experiment configuration
        yaml_paths: Dictionary mapping model types to their YAML paths

    Returns:
        Dictionary with model configurations and data modules for each model type
    """
    from src.experiment_framework.model_utils import (
        load_model_configs_from_yaml,
        load_model_datamodules,
    )

    # Load model configurations from YAML files
    model_configs = load_model_configs_from_yaml(yaml_paths)

    # Create DataModules for each model type
    data_modules = load_model_datamodules(
        time_series_data=time_series_data,
        static_data=static_data,
        config=config,
        model_configs=model_configs,
    )

    return {"model_configs": model_configs, "data_modules": data_modules}
