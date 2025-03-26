"""Utility functions for the Central Asian data sharing experiment."""

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


def setup_dirs(config: Any) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path(config.output_dir)

    # Define directory structure
    dirs = {
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }

    # Create directories for each country
    countries = [c.lower() for c in config.countries]
    model_types = config.model_types

    for country in countries:
        for model_type in model_types:
            (dirs["checkpoints"] / country / model_type).mkdir(
                parents=True, exist_ok=True
            )
            (dirs["logs"] / country / model_type).mkdir(parents=True, exist_ok=True)
            (dirs["results"] / country / model_type).mkdir(parents=True, exist_ok=True)

    return dirs


def create_data_module(
    time_series_data: pd.DataFrame,
    static_data: pd.DataFrame,
    model_type: str,
    yaml_path: str,
    config: Any,
) -> Tuple[HydroDataModule, Dict[str, Any]]:
    """
    Create a DataModule for the specified model type.

    Args:
        time_series_data: Time series data
        static_data: Static attribute data
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
    batch_size = model_params.get("batch_size", config.batch_size)

    # Get preprocessing config
    preprocessing_config = config.get_preprocessing_config()

    # Create DataModule with model-specific parameters
    data_module = HydroDataModule(
        time_series_df=time_series_data,
        static_df=static_data,
        group_identifier=config.group_identifier,
        preprocessing_config=preprocessing_config,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        num_workers=min(config.max_workers, os.cpu_count()),
        features=config.forcing_features + [config.target],
        static_features=[f for f in config.static_features if f != "country"],
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
        min_train_years=config.ca_min_train_years,
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
    country: str,
    config: Any,
    run_idx: int,
) -> Dict[str, Any]:
    """
    Train a model for a specific country scenario and save checkpoints.

    Args:
        model_type: Type of model to train ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML
        data_module: Prepared DataModule
        country: Country scenario ('Tajikistan', 'Kyrgyzstan', or 'Combined')
        config: Experiment configuration
        run_idx: Run index for this model

    Returns:
        Dictionary with training results
    """
    # Create model
    model, _ = create_model(model_type, yaml_path)

    # Prepare output directories
    checkpoint_dir = config.get_checkpoint_dir(country, model_type) / f"run_{run_idx}"
    logs_dir = config.get_logs_dir(country, model_type)

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
            filename=f"{country}_{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        ),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir), name=f"{country}_{model_type}", version=f"run_{run_idx}"
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

    # Store run results
    run_results = {
        "run": run_idx,
        "model_type": model_type,
        "country": country,
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(
            checkpoint_dir
            / f"{country}_{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
        ),
    }

    print(
        f"Run {run_idx} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
    )

    return run_results


def save_experiment_results(results: List[Dict[str, Any]], config: Any) -> None:
    """
    Save experiment results to CSV files.

    Args:
        results: List of result dictionaries from training runs
        config: Experiment configuration
    """
    # Group results by country
    country_results = {}
    for result in results:
        country = result["country"]
        if country not in country_results:
            country_results[country] = []
        country_results[country].append(result)

    # Save results for each country
    for country, country_data in country_results.items():
        results_dir = config.get_results_dir(country, "")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create summary dataframe
        summary_rows = []
        for result in country_data:
            summary_rows.append(
                {
                    "model_type": result["model_type"],
                    "country": result["country"],
                    "run": result["run"],
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "checkpoint_path": result["checkpoint_path"],
                }
            )

        # Save summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(results_dir / "summary.csv", index=False)

            # Also save model-specific results
            for model_type in config.model_types:
                model_results = summary_df[summary_df["model_type"] == model_type]
                if not model_results.empty:
                    model_results.to_csv(
                        results_dir / f"{model_type}_results.csv", index=False
                    )

    print(f"Results saved to {config.output_dir}/results")
