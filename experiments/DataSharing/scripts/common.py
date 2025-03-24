"""Common utilities for the data sharing experiment."""
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data_models.datamodule import HydroDataModule
from src.models.base.base_lit_model import BaseLitModel


def setup_logging(log_dir: Path, scenario: str) -> logging.Logger:
    """Set up logging for experiment.
    
    Args:
        log_dir: Directory to save logs
        scenario: Scenario name ('tajikistan', 'kyrgyzstan', or 'combined')
        
    Returns:
        Configured logger
    """
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"data_sharing_{scenario}")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{scenario}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Seed for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    model: BaseLitModel,
    datamodule: HydroDataModule,
    scenario: str,
    model_type: str,
    run_idx: int,
    experiment_config: Any,
    logger: logging.Logger
) -> Tuple[pl.Trainer, BaseLitModel]:
    """Train a model for a specific scenario and model type.
    
    Args:
        model: Model to train
        datamodule: DataModule with data
        scenario: Scenario name ('tajikistan', 'kyrgyzstan', or 'combined')
        model_type: Type of model ('tide', 'tsmixer', 'tft', 'ealstm')
        run_idx: Index of the current run
        experiment_config: Experiment configuration
        logger: Logger for output
        
    Returns:
        Tuple of (trainer, trained_model)
    """
    # Get directories for this scenario
    scenario_dirs = experiment_config.get_scenario_dirs(scenario)
    
    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(scenario_dirs["logs"]),
        name=model_type,
        version=f"run_{run_idx}"
    )
    
    # Set up model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(scenario_dirs["checkpoints"] / model_type / f"run_{run_idx}"),
        filename=f"{model_type}_{scenario}_run{run_idx}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    # Set up early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=experiment_config.patience,
        min_delta=experiment_config.min_delta,
        mode="min",
    )
    
    # Set up learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=experiment_config.max_epochs,
        accelerator=experiment_config.accelerator,
        devices=experiment_config.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        precision=experiment_config.precision,
        enable_progress_bar=experiment_config.verbose,
        enable_checkpointing=True,
    )
    
    # Train model
    logger.info(f"Starting training for {model_type} model on {scenario} data (run {run_idx})")
    trainer.fit(model, datamodule)
    
    # Log best validation loss
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else float('inf')
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Test model
    logger.info(f"Testing {model_type} model on {scenario} data (run {run_idx})")
    trainer.test(model, datamodule=datamodule)
    
    return trainer, model

