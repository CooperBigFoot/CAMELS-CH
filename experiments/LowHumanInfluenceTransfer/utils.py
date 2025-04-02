"""Utility functions for the global hydrological pretraining experiment."""

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, Any, Tuple, List
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_models.datamodule import HydroDataModule
from src.models.model_factory import create_model

from experiments.LowHumanInfluenceTransfer.config import ExperimentConfig


def setup_dirs(config: ExperimentConfig) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path(config.output_dir)

    # Define directory structure - removed models directory
    dirs = {
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }

    # Create directories for each model type
    for model_type in config.model_types:
        (dirs["checkpoints"] / model_type).mkdir(parents=True, exist_ok=True)
        (dirs["logs"] / model_type).mkdir(parents=True, exist_ok=True)
        (dirs["results"] / model_type).mkdir(parents=True, exist_ok=True)

    return dirs


def create_data_module(
    time_series_data: List[pd.DataFrame],
    static_data: List[pd.DataFrame],
    model_type: str,
    yaml_path: str,
    config: ExperimentConfig,
) -> Tuple[HydroDataModule, Dict[str, Any]]:
    """
    Create a DataModule for the specified model type.

    Args:
        time_series_data: List of time series DataFrames
        static_data: List of static attribute DataFrames
        model_type: Type of model ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML
        config: Experiment configuration

    Returns:
        Tuple containing:
        - Configured DataModule
        - Dictionary of model parameters
    """
    # Create model to get hyperparameters
    model, model_params = create_model(model_type, yaml_path)

    # Extract model-specific parameters
    input_length = model_params.get("input_len", 365)
    output_length = model_params.get("output_len", 1)
    batch_size = config.batch_size

    # Get preprocessing config
    preprocessing_config = config.get_preprocessing_config()

    # Create DataModule with model-specific parameters
    data_module = HydroDataModule(
        time_series_df=time_series_data,  # Can be a list of DataFrames
        static_df=static_data,  # Can be a list of DataFrames
        group_identifier=config.group_identifier,
        preprocessing_config=preprocessing_config,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        num_workers=min(config.max_workers, os.cpu_count()),
        features=config.forcing_features + [config.target],
        static_features=config.static_features,
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
    )

    print(
        f"Created DataModule for {model_type} with input_length={input_length}, "
        f"output_length={output_length}, batch_size={batch_size}"
    )

    return data_module, model_params


def train_model(
    model_type: str,
    yaml_path: str,
    data_module: HydroDataModule,
    config: ExperimentConfig,
    run_idx: int,
) -> Dict[str, Any]:
    """
    Train a model and save checkpoints.

    Args:
        model_type: Type of model to train ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML
        data_module: Prepared DataModule
        config: Experiment configuration
        run_idx: Run index for this model

    Returns:
        Dictionary with training results
    """
    # Create model
    model, _ = create_model(model_type, yaml_path)

    # Prepare output directories
    checkpoint_dir = config.get_checkpoint_dir(model_type) / f"run_{run_idx}"
    logs_dir = config.get_logs_dir(model_type)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"global_{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        ),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir), name=f"global_{model_type}", version=f"run_{run_idx}"
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    # Train model
    trainer.fit(model, data_module)

    # Get the best validation loss
    best_val_loss = trainer.callback_metrics.get(
        "val_loss", torch.tensor(float("inf"))
    ).item()

    # Store run results - removed saving to models directory
    best_checkpoint_path = checkpoint_dir / f"global_{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
    
    run_results = {
        "run": run_idx,
        "model_type": model_type,
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(best_checkpoint_path),
    }

    print(
        f"Run {run_idx} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
    )

    return run_results


def save_experiment_results(
    results: List[Dict[str, Any]], config: ExperimentConfig
) -> None:
    """
    Save experiment results to CSV files.

    Args:
        results: List of result dictionaries from training runs
        config: Experiment configuration
    """
    # Group results by model type
    model_results = {}
    for result in results:
        model_type = result["model_type"]
        if model_type not in model_results:
            model_results[model_type] = []
        model_results[model_type].append(result)
    
    # Create summary dataframe
    summary_rows = []
    for result in results:
        summary_rows.append({
            "model_type": result["model_type"],
            "run": result["run"],
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "checkpoint_path": result["checkpoint_path"],
        })
    
    # Save summary for all models
    results_dir = Path(config.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(results_dir / "summary.csv", index=False)
        
        # Calculate and save average performance per model
        avg_results = []
        for model_type in config.model_types:
            model_df = summary_df[summary_df["model_type"] == model_type]
            if not model_df.empty:
                avg_results.append({
                    "model_type": model_type,
                    "avg_val_loss": model_df["best_val_loss"].mean(),
                    "std_val_loss": model_df["best_val_loss"].std(),
                    "min_val_loss": model_df["best_val_loss"].min(),
                    "max_val_loss": model_df["best_val_loss"].max(),
                    "avg_epochs": model_df["best_epoch"].mean(),
                    "runs": len(model_df),
                })
        
        if avg_results:
            avg_df = pd.DataFrame(avg_results)
            avg_df.to_csv(results_dir / "average_performance.csv", index=False)
    
    # Save model-specific results
    for model_type, model_data in model_results.items():
        model_dir = results_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_df = pd.DataFrame(model_data)
        model_df.to_csv(model_dir / "results.csv", index=False)
    
    print(f"Results saved to {config.output_dir}/results")
