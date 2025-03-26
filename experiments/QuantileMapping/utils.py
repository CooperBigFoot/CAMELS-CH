"""Utility functions for the quantile mapping experiment."""

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


def setup_dirs(output_dir: str) -> None:
    """
    Create necessary output directories for the experiment.
    
    Args:
        output_dir: Base output directory path
    """
    base_dir = Path(output_dir)
    
    # Create main directories
    for dir_name in ["checkpoints", "logs", "results"]:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create data source subdirectories
    for data_source in ["original", "quantile_mapped"]:
        for dir_name in ["checkpoints", "logs", "results"]:
            (base_dir / dir_name / data_source).mkdir(parents=True, exist_ok=True)


def create_data_module(
    time_series_data: pd.DataFrame,
    static_data: pd.DataFrame,
    model_type: str,
    yaml_path: str,
    config: Any
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
        static_features=[f for f in config.static_features if f != config.group_identifier],
        target=config.target,
        use_proportional_split=config.use_proportional_split,
        train_prop=config.train_prop,
        val_prop=config.val_prop,
        test_prop=config.test_prop,
    )
    
    return data_module, model_params


def train_model(
    model_type: str,
    yaml_path: str,
    data_module: HydroDataModule,
    data_source: str,
    config: Any,
    run_idx: int,
) -> Dict[str, Any]:
    """
    Train a model for a specific data source and save checkpoints.
    
    Args:
        model_type: Type of model to train ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML file
        data_module: Prepared DataModule for training
        data_source: Data source identifier ('original' or 'quantile_mapped')
        config: Experiment configuration
        run_idx: Run index for this model
        
    Returns:
        Dictionary with training results
    """
    # Create model
    model, _ = create_model(model_type, yaml_path)
    
    # Prepare output directories
    checkpoint_dir = config.get_checkpoint_dir(data_source, model_type) / f"run_{run_idx}"
    logs_dir = config.get_logs_dir(data_source, model_type)
    
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
            filename=f"{data_source}_{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            save_last=config.save_last,
        ),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(logs_dir), 
        name=f"{data_source}_{model_type}", 
        version=f"run_{run_idx}"
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
    best_val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()
    
    # Store run results
    run_results = {
        "run": run_idx,
        "model_type": model_type,
        "data_source": data_source,
        "best_val_loss": best_val_loss,
        "best_epoch": trainer.current_epoch,
        "checkpoint_path": str(
            checkpoint_dir / 
            f"{data_source}_{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
        ),
    }
    
    print(f"Run {run_idx} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}")
    
    return run_results


def save_experiment_results(
    all_results: List[Dict[str, Any]], 
    output_dir: str, 
    data_source: str
) -> None:
    """
    Save experiment results to CSV files.
    
    Args:
        all_results: List of result dictionaries from training runs
        output_dir: Base output directory
        data_source: Data source identifier ('original' or 'quantile_mapped')
    """
    if not all_results:
        print(f"No results to save for {data_source}")
        return
        
    # Create results directory
    results_dir = Path(output_dir) / "results" / data_source
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save full summary
    results_df.to_csv(results_dir / "all_results.csv", index=False)
    
    # Save model-specific results
    for model_type in results_df["model_type"].unique():
        model_results = results_df[results_df["model_type"] == model_type]
        model_results.to_csv(results_dir / f"{model_type}_results.csv", index=False)
    
    # Save best model results
    best_results = []
    for model_type in results_df["model_type"].unique():
        model_df = results_df[results_df["model_type"] == model_type]
        best_model = model_df.loc[model_df["best_val_loss"].idxmin()]
        best_results.append(best_model.to_dict())
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(results_dir / "best_model_results.csv", index=False)
    
    # Save as JSON for easier parsing
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Results saved to {results_dir}")


def load_model_configs_and_datamodules(
    time_series_data: pd.DataFrame,
    static_data: pd.DataFrame,
    config: Any,
    yaml_paths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, HydroDataModule]]:
    """
    Load model configurations and create data modules for multiple models.
    
    Args:
        time_series_data: Time series data DataFrame
        static_data: Static attributes DataFrame
        config: Experiment configuration
        yaml_paths: Dictionary mapping model types to YAML file paths
        
    Returns:
        Tuple containing:
        - Dictionary of model configurations
        - Dictionary of HydroDataModules
    """
    model_configs = {}
    data_modules = {}
    
    for model_type, yaml_path in yaml_paths.items():
        try:
            # Create data module
            data_module, model_config = create_data_module(
                time_series_data=time_series_data,
                static_data=static_data,
                model_type=model_type,
                yaml_path=yaml_path,
                config=config
            )
            
            # Store configuration and data module
            model_configs[model_type] = model_config
            data_modules[model_type] = data_module
            
            # Prepare data
            data_module.prepare_data()
            data_module.setup()
            
        except Exception as e:
            print(f"Error creating data module for {model_type}: {str(e)}")
            continue
    
    return model_configs, data_modules


def train_and_save_model(
    model_type: str,
    model_config: Dict[str, Any],
    data_module: HydroDataModule,
    data_source: str,
    output_dir: str,
    num_runs: int,
    early_stopping_patience: int = 5,
    save_top_k: int = 1,
    save_last: bool = True,
) -> List[Dict[str, Any]]:
    """
    Train a model multiple times and save results.
    
    Args:
        model_type: Type of model ('tide', 'tsmixer', etc.)
        model_config: Model configuration dictionary
        data_module: Prepared DataModule
        data_source: Data source identifier ('original' or 'quantile_mapped')
        output_dir: Base output directory
        num_runs: Number of training runs to perform
        early_stopping_patience: Patience for early stopping
        save_top_k: Number of best checkpoints to save
        save_last: Whether to save the last checkpoint
        
    Returns:
        List of result dictionaries from all runs
    """
    results = []
    
    # Create checkpoint directory
    checkpoint_dir = Path(output_dir) / "checkpoints" / data_source / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get YAML path from model_config (needed for train_model)
    yaml_path = model_config.get("yaml_path", f"yaml_files/{model_type}.yaml")
    
    for run_idx in range(num_runs):
        print(f"\nStarting run {run_idx + 1}/{num_runs} for {model_type} on {data_source} data")
        
        # Set seed for this run
        seed = 42 + run_idx
        pl.seed_everything(seed)
        
        try:
            # Create a fresh model for each run
            model, _ = create_model(model_type, yaml_path)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min"
                ),
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir / f"run_{run_idx}"),
                    filename=f"{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=save_top_k,
                    save_last=save_last
                ),
                LearningRateMonitor(logging_interval="epoch")
            ]
            
            # Setup logger
            logger = TensorBoardLogger(
                save_dir=str(Path(output_dir) / "logs" / data_source),
                name=model_type,
                version=f"run_{run_idx}"
            )
            
            # Configure trainer
            trainer = pl.Trainer(
                max_epochs=model_config.get("max_epochs", 100),
                callbacks=callbacks,
                logger=logger,
                accelerator="auto",
                devices=1
            )
            
            # Train model
            trainer.fit(model, data_module)
            
            # Get best validation loss
            best_val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()
            
            # Record results
            result = {
                "run": run_idx,
                "model_type": model_type,
                "data_source": data_source,
                "best_val_loss": best_val_loss,
                "best_epoch": trainer.current_epoch,
                "checkpoint_path": str(checkpoint_dir / f"run_{run_idx}" / 
                    f"{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt")
            }
            results.append(result)
            
            print(f"Run {run_idx + 1} completed: best_val_loss={best_val_loss:.4f}, epoch={trainer.current_epoch}")
            
        except Exception as e:
            print(f"Error in run {run_idx} for {model_type} on {data_source} data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results
