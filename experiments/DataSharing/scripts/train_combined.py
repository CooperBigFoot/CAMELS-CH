"""Train models on combined Tajikistan and Kyrgyzstan data."""
import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

from experiments.DataSharing.configs.experiment_config import DataSharingExperimentConfig
from experiments.DataSharing.configs.data_config import DataSharingDataConfig
from experiments.DataSharing.configs.model_configs.tide_config import TiDEModelConfig
from experiments.DataSharing.configs.model_configs.tsmixer_config import TSMixerModelConfig
from experiments.DataSharing.configs.model_configs.tft_config import TFTModelConfig
from experiments.DataSharing.configs.model_configs.ealstm_config import EALSTMModelConfig
from experiments.DataSharing.scripts.common import (
    setup_logging,
    set_seed,
    train_model
)

SCENARIO = "combined"


def train_combined_models(
    experiment_config: DataSharingExperimentConfig,
    data_config: DataSharingDataConfig,
    model_types: Optional[List[str]] = None,
    runs: Optional[List[int]] = None
) -> Dict[str, Dict[int, float]]:
    """Train models on combined Tajikistan and Kyrgyzstan data.
    
    Args:
        experiment_config: Experiment configuration
        data_config: Data configuration
        model_types: List of model types to train (default: all defined in experiment_config)
        runs: List of run indices to execute (default: all runs in experiment_config)
        
    Returns:
        Dictionary of best validation losses by model type and run
    """
    # Set up logging
    logger = setup_logging(experiment_config.logs_dir, SCENARIO)
    logger.info(f"Starting {SCENARIO} training experiments")
    
    # Create experiment directories
    experiment_config.create_experiment_dirs()
    
    # Determine which models to train
    model_types = model_types or experiment_config.model_types
    runs = runs or list(range(experiment_config.num_runs))
    
    # Load data
    logger.info("Loading CARAVAN data")
    caravan, kgz_ids, tjk_ids = data_config.load_data()
    
    # Combine IDs from both countries
    combined_ids = kgz_ids + tjk_ids
    
    # Store best validation losses
    best_val_losses = {model_type: {} for model_type in model_types}
    
    # Train each model type with multiple runs
    for model_type in model_types:
        logger.info(f"Processing model type: {model_type}")
        
        # Prepare model-specific datamodule for combined data
        logger.info(f"Preparing {model_type}-specific datamodule for {SCENARIO} with {len(combined_ids)} basins")
        datamodule = data_config.prepare_model_datamodule(
            caravan=caravan, 
            basin_ids=combined_ids, 
            model_type=model_type,
            domain_id="CA"
        )
        
        for run_idx in runs:
            logger.info(f"Starting run {run_idx} for {model_type}")
            
            # Set seed for reproducibility
            seed = experiment_config.get_run_seed(run_idx)
            set_seed(seed)
            logger.info(f"Using seed {seed} for run {run_idx}")
            
            # Create model based on type
            input_size = len(data_config.forcing_features) + 1  # +1 for target
            static_size = len(data_config.static_features) - 1  # -1 for gauge_id
            future_input_size = len(data_config.forcing_features)
            
            # Select and create appropriate model
            model = None
            if model_type == "tide":
                config = TiDEModelConfig()
                model = config.create_model(
                    input_len=data_config.input_length,
                    output_len=data_config.output_length,
                    input_size=input_size,
                    static_size=static_size,
                    future_input_size=future_input_size
                )
            elif model_type == "tsmixer":
                config = TSMixerModelConfig()
                model = config.create_model(
                    input_len=data_config.input_length,
                    output_len=data_config.output_length,
                    input_size=input_size,
                    static_size=static_size,
                    future_input_size=future_input_size
                )
            elif model_type == "tft":
                config = TFTModelConfig()
                model = config.create_model(
                    input_len=data_config.input_length,
                    output_len=data_config.output_length,
                    input_size=input_size,
                    static_size=static_size,
                    future_input_size=future_input_size
                )
            elif model_type == "ealstm":
                config = EALSTMModelConfig()
                model = config.create_model(
                    input_len=data_config.input_length,
                    output_len=data_config.output_length,
                    input_size=input_size,
                    static_size=static_size,
                    future_input_size=future_input_size
                )
            else:
                logger.error(f"Unknown model type: {model_type}")
                continue
                
            # Train model
            start_time = time.time()
            try:
                trainer, trained_model = train_model(
                    model=model,
                    datamodule=datamodule,
                    scenario=SCENARIO,
                    model_type=model_type,
                    run_idx=run_idx,
                    experiment_config=experiment_config,
                    logger=logger
                )
                
                # Store best validation loss
                if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback.best_model_score:
                    best_val_losses[model_type][run_idx] = trainer.checkpoint_callback.best_model_score.item()
                else:
                    best_val_losses[model_type][run_idx] = float('inf')
                    
            except Exception as e:
                logger.error(f"Error training {model_type} model (run {run_idx}): {e}")
                best_val_losses[model_type][run_idx] = float('inf')
                
            # Log training time
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Training time for {model_type} (run {run_idx}): {training_time:.2f} seconds")
    
    # Log summary of results
    logger.info("Training completed. Summary of best validation losses:")
    for model_type in model_types:
        losses = best_val_losses[model_type]
        average_loss = sum(losses.values()) / len(losses) if losses else float('inf')
        logger.info(f"  {model_type}: {average_loss:.4f} (average of {len(losses)} runs)")
    
    return best_val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on combined Tajikistan and Kyrgyzstan data")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["tide", "tsmixer", "tft", "ealstm", "all"],
        default=["all"], 
        help="List of models to train"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=None, 
        help="Number of runs (default: use value from experiment config)"
    )
    args = parser.parse_args()
    
    # Initialize configurations
    experiment_config = DataSharingExperimentConfig()
    data_config = DataSharingDataConfig()
    
    # Handle 'all' in models argument
    model_types = None
    if args.models != ["all"]:
        model_types = args.models
    
    # Determine number of runs
    runs = None
    if args.runs is not None:
        runs = list(range(args.runs))
    
    # Run experiments
    train_combined_models(
        experiment_config=experiment_config,
        data_config=data_config,
        model_types=model_types,
        runs=runs
    )
