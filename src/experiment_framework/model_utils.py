"""Model creation and loading utilities for hydrological forecasting experiments."""

import os
from typing import Dict, Any, Optional, List
import logging
import torch
import pytorch_lightning as pl

# Configure logging
logger = logging.getLogger(__name__)


def create_model(
    model_type: str,
    model_config: Any,
    **kwargs
) -> pl.LightningModule:
    """Create a new model instance of the specified type.
    
    Args:
        model_type: Type of model to create (tide, tsmixer, ealstm, tft)
        model_config: Configuration for the model
        **kwargs: Additional keyword arguments to pass to the model constructor
        
    Returns:
        New PyTorch Lightning model instance
        
    Raises:
        ValueError: If the model type is not supported
        ImportError: If the model module cannot be imported
    """
    logger.info(f"Creating model of type: {model_type}")
    
    model_type = model_type.lower()
    
    # Create model based on type
    try:
        if model_type == "tide":
            from src.models.tide import LitTiDE
            return LitTiDE(model_config, **kwargs)
            
        elif model_type == "tsmixer":
            from src.models.tsmixer import LitTSMixer
            return LitTSMixer(model_config, **kwargs)
            
        elif model_type == "ealstm":
            from src.models.ealstm import LitEALSTM
            return LitEALSTM(model_config, **kwargs)
            
        elif model_type == "tft":
            from src.models.tft import LitTFT
            return LitTFT(model_config, **kwargs)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except ImportError as e:
        raise ImportError(f"Could not import model type {model_type}: {e}")


def load_pretrained_model(
    checkpoint_path: str,
    model_type: str,
    model_config: Any,
    finetune: bool = False,
    lr_factor: float = 10.0,
    reset_optimizer: bool = False,
    **kwargs
) -> pl.LightningModule:
    """Load a pre-trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model to create
        model_config: Configuration for the model
        finetune: Whether to prepare the model for fine-tuning
        lr_factor: Factor to reduce learning rate by when fine-tuning
        reset_optimizer: Whether to reset optimizer state
        **kwargs: Additional keyword arguments to pass to the model constructor
        
    Returns:
        Pre-trained PyTorch Lightning model
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist
        RuntimeError: If the checkpoint file cannot be loaded
    """
    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create new model
    model = create_model(model_type, model_config, **kwargs)
    
    # Load checkpoint
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # PyTorch Lightning checkpoint
            model.load_state_dict(checkpoint["state_dict"])
            
            # If requested, preserve optimizer state
            if not reset_optimizer and "optimizer_states" in checkpoint:
                # Store optimizer state for later use
                model.optimizer_states = checkpoint["optimizer_states"]
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        logger.info("Successfully loaded model weights from checkpoint")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Adjust learning rate for fine-tuning if needed
    if finetune:
        original_lr = model.hparams.learning_rate
        new_lr = original_lr / lr_factor
        
        # Update learning rate in all relevant places
        model.hparams.learning_rate = new_lr
        
        # Update in config if it exists
        if hasattr(model, "config"):
            model.config.learning_rate = new_lr
            
        # If learning_rate is directly accessible
        if hasattr(model, "learning_rate"):
            model.learning_rate = new_lr
        
        # Create a reference to original configure_optimizers
        original_configure_optimizers = model.configure_optimizers
        
        def new_configure_optimizers():
            """Override configure_optimizers to use reduced learning rate."""
            # Get original optimizer and scheduler
            result = original_configure_optimizers()
            
            # If the result is just an optimizer
            if isinstance(result, torch.optim.Optimizer):
                # Update learning rate for all parameter groups
                for param_group in result.param_groups:
                    param_group['lr'] = new_lr
                return result
                
            # If result is a tuple or list
            elif isinstance(result, (tuple, list)):
                optimizer = result[0]
                
                # Handle case where optimizer is a list
                if isinstance(optimizer, list):
                    for opt in optimizer:
                        for param_group in opt.param_groups:
                            param_group['lr'] = new_lr
                else:
                    # Single optimizer
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                        
                return result
                
            # If we can't modify the result, return it as is
            return result
        
        # Replace the original method with our custom one
        model.configure_optimizers = new_configure_optimizers
        
        # Store original and new learning rates
        model.original_lr = original_lr
        model.fine_tuned_lr = new_lr
        model.is_fine_tuned = True
        
        logger.info(f"Adjusted learning rate from {original_lr:.6f} to {new_lr:.6f} for fine-tuning")
        
    return model


def load_model_configs_from_yaml(
    yaml_paths: Dict[str, str],
    model_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load model configurations from pre-tuned YAML files.
    
    Args:
        yaml_paths: Dictionary mapping model types to their YAML paths
        model_types: Optional list of model types to load (defaults to all keys in yaml_paths)
        
    Returns:
        Dictionary mapping model types to their configurations
    """
    try:
        from src.model_evaluation.hp_from_yaml import load_model_config
    except ImportError:
        raise ImportError(
            "load_model_config not found. Make sure src.model_evaluation.hp_from_yaml is available."
        )
    
    # Filter yaml_paths to requested model types if specified
    if model_types:
        yaml_paths = {
            k: v for k, v in yaml_paths.items() if k in model_types
        }
    
    # Initialize configurations dictionary
    configs = {}
    
    # Process each model type
    for model_type, yaml_path in yaml_paths.items():
        if not yaml_path or not os.path.exists(yaml_path):
            logger.warning(f"YAML path for {model_type} not found: {yaml_path}")
            continue
            
        try:
            # Load hyperparameters from YAML
            model_hp = load_model_config(model_type, yaml_path)
            
            # Create model configuration
            if model_type == "tide":
                from src.models.tide import TiDEConfig
                model_config = TiDEConfig(**model_hp)
                
            elif model_type == "tsmixer":
                from src.models.tsmixer import TSMixerConfig
                model_config = TSMixerConfig(**model_hp)
                
            elif model_type == "ealstm":
                from src.models.ealstm import EALSTMConfig
                model_config = EALSTMConfig(**model_hp)
                
            elif model_type == "tft":
                from src.models.tft import TFTConfig
                model_config = TFTConfig(**model_hp)
                
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                continue
                
            # Store the configuration
            configs[model_type] = model_config
            logger.info(f"Loaded configuration for {model_type} from {yaml_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration for {model_type}: {str(e)}")
            continue
            
    return configs


def load_model_datamodules(
    time_series_data: Any,
    static_data: Any,
    config: Any,
    model_configs: Dict[str, Any]
) -> Dict[str, Any]:
    """Create DataModules for each model type with model-specific parameters.
    
    Args:
        time_series_data: Time series data frame
        static_data: Static attributes data frame
        config: Experiment configuration
        model_configs: Dictionary mapping model types to their configurations
        
    Returns:
        Dictionary mapping model types to their DataModules
    """
    from src.experiment_framework.data_utils import create_datamodule
    
    # Initialize data modules dictionary
    data_modules = {}
    
    # Process each model type
    for model_type, model_config in model_configs.items():
        try:
            # Extract model-specific parameters
            input_length = getattr(model_config, "input_len", None)
            output_length = getattr(model_config, "output_len", None)
            batch_size = getattr(model_config, "batch_size", config.batch_size)
            
            if input_length is None or output_length is None:
                logger.warning(f"Model {model_type} is missing input_len or output_len parameters")
                continue
                
            # Create DataModule with model-specific parameters
            data_modules[model_type] = create_datamodule(
                time_series_df=time_series_data,
                static_df=static_data,
                config=config,
                input_length=input_length,
                output_length=output_length,
                batch_size=batch_size
            )
            
            logger.info(
                f"Created DataModule for {model_type} with input_length={input_length}, "
                f"output_length={output_length}, batch_size={batch_size}"
            )
            
        except Exception as e:
            logger.error(f"Error creating DataModule for {model_type}: {str(e)}")
            continue
            
    return data_modules
