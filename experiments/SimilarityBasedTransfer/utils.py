"""Utility functions for the similarity-based transfer learning experiment."""

import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_models.datamodule import HydroDataModule
from src.models.model_factory import create_model


def setup_dirs(config: Any) -> Dict[str, Path]:
    """
    Create necessary output directories for the experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of directory paths
    """
    base_dir = Path(config.output_dir)

    # Define main directories
    dirs = {
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }

    # Create base directories
    for dir_name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create group-specific directories
    for group in config.target_groups:
        for model_type in config.model_types:
            (dirs["checkpoints"] / group / model_type).mkdir(
                parents=True, exist_ok=True
            )
            (dirs["logs"] / group / model_type).mkdir(parents=True, exist_ok=True)
            (dirs["results"] / group).mkdir(parents=True, exist_ok=True)

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
        time_series_data: Time series data DataFrame
        static_data: Static attributes DataFrame
        model_type: Type of model to create ('tide', 'tsmixer', 'ealstm', 'tft')
        yaml_path: Path to model hyperparameter YAML file
        config: Experiment configuration

    Returns:
        Tuple containing:
        - Configured HydroDataModule
        - Dictionary of model hyperparameters
    """
    # Create model to get hyperparameters
    model, model_params = create_model(model_type, yaml_path)

    # Extract model-specific parameters
    input_length = model_params.get("input_len", 365)
    output_length = model_params.get("output_len", 1)
    batch_size = model_params.get("batch_size", config.batch_size)

    # Get preprocessing configuration
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
        num_workers=min(config.max_workers, os.cpu_count()),
        features=config.forcing_features,
        static_features=[
            f for f in config.static_features if f != config.group_identifier
        ],
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
    )

    # Store yaml path in model params for reference
    model_params["yaml_path"] = yaml_path

    return data_module, model_params


def train_model(
    model_type: str,
    yaml_path: str,
    data_module: HydroDataModule,
    group: str,
    config: Any,
    run_idx: int,
) -> Dict[str, Any]:
    """
    Train a model for a specific group and save checkpoints.

    Args:
        model_type: Type of model to train ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML file
        data_module: Prepared DataModule for training
        group: Group identifier (e.g., 'group1')
        config: Experiment configuration
        run_idx: Run index for this model

    Returns:
        Dictionary with training results
    """
    # Create model
    model, _ = create_model(model_type, yaml_path)

    # Prepare output directories
    checkpoint_dir = config.get_checkpoint_dir(group, model_type) / f"run_{run_idx}"
    logs_dir = config.get_logs_dir(group, model_type)

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
            filename=f"{group}_{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        ),
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir), name=f"{group}_{model_type}", version=f"run_{run_idx}"
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
        "group": group,
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(
            checkpoint_dir
            / f"{group}_{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
        ),
    }

    print(
        f"Run {run_idx} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
    )

    return run_results


def save_experiment_results(
    results: List[Dict[str, Any]], config: Any, group: str
) -> None:
    """
    Save experiment results to CSV files for a specific group.

    Args:
        results: List of result dictionaries from training runs
        config: Experiment configuration
        group: Group identifier (e.g., 'group1')
    """
    if not results:
        print(f"No results to save for {group}")
        return

    # Create results directory
    results_dir = config.get_results_dir(group)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create summary dataframe
    results_df = pd.DataFrame(results)

    # Save full summary
    results_df.to_csv(results_dir / "summary.csv", index=False)

    # Save model-specific results
    for model_type in results_df["model_type"].unique():
        model_results = results_df[results_df["model_type"] == model_type]
        model_results.to_csv(results_dir / f"{model_type}_results.csv", index=False)

    # Save best model results
    best_results = []
    for model_type in results_df["model_type"].unique():
        model_df = results_df[results_df["model_type"] == model_type]
        best_idx = model_df["best_val_loss"].idxmin()
        if not pd.isna(best_idx):  # Check if we have valid results
            best_model = model_df.loc[best_idx]
            best_results.append(best_model.to_dict())

    if best_results:
        best_df = pd.DataFrame(best_results)
        best_df.to_csv(results_dir / "best_model_results.csv", index=False)

    # Save as JSON for easier parsing
    with open(results_dir / "all_results.json", "w") as f:
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                k: str(v) if isinstance(v, Path) else v for k, v in result.items()
            }
            serializable_results.append(serializable_result)

        json.dump(serializable_results, f, indent=4)

    print(f"Results saved to {results_dir}")


def load_model_configs_and_datamodules(
    group_key: str,
    config: Any,
    yaml_paths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, HydroDataModule]]:
    """
    Load model configurations and create data modules for a specific group.

    Args:
        group_key: Group identifier (e.g., 'group1')
        config: Experiment configuration
        yaml_paths: Dictionary mapping model types to YAML file paths

    Returns:
        Tuple containing:
        - Dictionary of model configurations
        - Dictionary of HydroDataModules
    """
    from data_loader import load_data_for_group

    # Load group data
    group_data = load_data_for_group(config, group_key)

    model_configs = {}
    data_modules = {}

    for model_type, yaml_path in yaml_paths.items():
        try:
            # Create data module
            data_module, model_config = create_data_module(
                time_series_data=group_data["time_series"],
                static_data=group_data["static"],
                model_type=model_type,
                yaml_path=yaml_path,
                config=config,
            )

            # Store configuration and data module
            model_configs[model_type] = model_config
            data_modules[model_type] = data_module

            # Prepare data
            data_module.prepare_data()
            data_module.setup()

            # Log data splits
            print(f"Data splits for group {group_key}, model {model_type}:")
            print(f"  - Train: {len(data_module.train_dataset)} samples")
            print(f"  - Validation: {len(data_module.val_dataset)} samples")
            print(f"  - Test: {len(data_module.test_dataset)} samples")

        except Exception as e:
            print(f"Error creating data module for {model_type}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    return model_configs, data_modules
