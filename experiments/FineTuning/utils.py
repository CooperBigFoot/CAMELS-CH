"""Utility functions for fine-tuning experiments."""

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_models.datamodule import HydroDataModule
from experiments.FineTuning.config import ExperimentConfig


def setup_dirs(config: ExperimentConfig, run_idx: int = None) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.

    Args:
        config: Experiment configuration
        run_idx: Optional run index for multiple runs

    Returns:
        Dictionary of Path objects for different directories
    """
    checkpoint_dir = config.get_checkpoint_dir(run_idx)
    logs_dir = config.get_logs_dir(run_idx)
    results_dir = config.get_results_dir()

    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return {"checkpoints": checkpoint_dir, "logs": logs_dir, "results": results_dir}


def prepare_data_module(
    config: ExperimentConfig, model_hp: Dict[str, Any]
) -> HydroDataModule:
    """Prepare a HydroDataModule for the experiment.

    Args:
        config: Experiment configuration
        model_hp: Model hyperparameters

    Returns:
        Configured HydroDataModule
    """
    # Import data_loader at function level to avoid circular imports
    from experiments.FineTuning.data_loader import load_data

    # Load data
    data = load_data(config)
    time_series_data = data["time_series"]
    static_data = data["static"]

    # Get model-specific parameters
    input_length = model_hp["input_len"]
    output_length = model_hp["output_len"]
    batch_size = config.batch_size

    # Get preprocessing config
    preprocessing_config = config.get_preprocessing_config()

    # Create DataModule
    data_module = HydroDataModule(
        time_series_df=time_series_data,
        static_df=static_data,
        group_identifier=config.group_identifier,
        preprocessing_config=preprocessing_config,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        num_workers=min(config.num_workers, os.cpu_count() or 1),
        features=config.forcing_features + [config.target],
        static_features=[f for f in config.static_features if f != "country"],
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
        min_train_years=config.min_train_years,
    )

    # Prepare data and setup
    data_module.prepare_data()
    data_module.setup()

    print(f"Prepared data module for {config.target_country or 'all countries'}")
    print(f"Number of basins: {data['basin_count']}")

    return data_module


def fine_tune_model(
    model: pl.LightningModule,
    model_hp: Dict[str, Any],
    data_module: HydroDataModule,
    config: ExperimentConfig,
    run_idx: int = 0,
) -> Dict[str, Any]:
    """Fine-tune a pre-trained model.

    Args:
        model: Pre-trained model to fine-tune
        model_hp: Model hyperparameters
        data_module: DataModule for training
        config: Experiment configuration
        run_idx: Run index for multiple runs

    Returns:
        Dictionary with fine-tuning results
    """
    # Setup directories for this run
    dirs = setup_dirs(config, run_idx)

    # Configure logger name and version
    if config.target_country:
        logger_name = f"{config.target_country}_{config.model_type}"
    else:
        logger_name = config.model_type

    # Configure logger with run-specific version
    logger = TensorBoardLogger(
        save_dir=str(dirs["logs"]),
        name=logger_name,
        version=f"run_{run_idx}",
    )

    # Run-specific filename for checkpoints
    checkpoint_filename = f"{logger_name}_run{run_idx}_{{epoch:02d}}_{{val_loss:.4f}}"

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            mode="min",
            min_delta=config.early_stopping_min_delta,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(dirs["checkpoints"]),
            filename=checkpoint_filename,
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        ),
    ]

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    # Train model
    print(
        f"Starting fine-tuning run {run_idx} for {config.model_type} on {config.target_country or 'all'} data"
    )
    trainer.fit(model, data_module)

    # Get the best validation loss
    best_val_loss = trainer.callback_metrics.get(
        "val_loss", torch.tensor(float("inf"))
    ).item()

    # Create results summary
    results = {
        "model_type": config.model_type,
        "country": config.target_country or "all",
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(
            dirs["checkpoints"]
            / f"{checkpoint_filename.format(epoch=trainer.current_epoch, val_loss=best_val_loss)}.ckpt"
        ),
        "original_lr": getattr(model, "original_lr", "unknown"),
        "fine_tuned_lr": getattr(model, "fine_tuned_lr", "unknown"),
    }

    print(
        f"Fine-tuning run {run_idx} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
    )

    return results


def save_results(results: List[Dict[str, Any]], config: ExperimentConfig) -> None:
    """Save fine-tuning results to a CSV file.

    Args:
        results: List of fine-tuning results from multiple runs
        config: Experiment configuration
    """
    # Setup directories if not already done
    dirs = setup_dirs(config)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Generate output path
    output_path = dirs["results"] / f"{config.model_type}_results.csv"

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Also save the best results
    if not results_df.empty:
        best_idx = results_df["best_val_loss"].idxmin()
        best_result = results_df.loc[best_idx]
        
        best_output_path = dirs["results"] / f"{config.model_type}_best_result.csv"
        pd.DataFrame([best_result.to_dict()]).to_csv(best_output_path, index=False)
        
        print(f"Best result (run {best_result['run']}) saved to {best_output_path}")
        print(f"Best validation loss: {best_result['best_val_loss']:.4f}")
