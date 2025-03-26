import sys
from pathlib import Path

from typing import Any, Dict, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model_evaluation.hp_from_yaml import hp_from_yaml


def create_model(model_type: str, yaml_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a model instance from a YAML file.

    Args:
        model_type: Type of model to create ('tide', 'tsmixer', etc.)
        yaml_path: Path to model hyperparameter YAML file

    Returns:
        Tuple containing:
        - Model instance
        - Dictionary of model hyperparameters
    """
    # Load hyperparameters from YAML
    model_hp = hp_from_yaml(model_type, yaml_path)

    # Create appropriate model configuration
    if model_type == "tide":
        from src.models.tide import TiDEConfig, LitTiDE

        model_config = TiDEConfig(**model_hp)
        model = LitTiDE(config=model_config)
    elif model_type == "tsmixer":
        from src.models.tsmixer import TSMixerConfig, LitTSMixer

        model_config = TSMixerConfig(**model_hp)
        model = LitTSMixer(config=model_config)
    elif model_type == "ealstm":
        from src.models.ealstm import EALSTMConfig, LitEALSTM

        model_config = EALSTMConfig(**model_hp)
        model = LitEALSTM(config=model_config)
    elif model_type == "tft":
        from src.models.tft import TFTConfig, LitTFT

        model_config = TFTConfig(**model_hp)
        model = LitTFT(config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, model_hp


def load_pretrained_model(
    model_type: str,
    yaml_path: str,
    checkpoint_path: str,
    finetune: bool = False,
    lr_factor: float = 10.0,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a pretrained model from a checkpoint.

    Args:
        model_type: Type of model to load
        yaml_path: Path to model hyperparameter YAML file
        checkpoint_path: Path to model checkpoint
        finetune: Whether to prepare model for fine-tuning
        lr_factor: Factor to reduce learning rate by for fine-tuning

    Returns:
        Tuple containing:
        - Loaded model instance
        - Dictionary of model hyperparameters
    """
    # Create model config
    model_hp = hp_from_yaml(model_type, yaml_path)

    if model_type == "tide":
        from src.models.tide import TiDEConfig, LitTiDE

        model_config = TiDEConfig(**model_hp)
        model = LitTiDE.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "tsmixer":
        from src.models.tsmixer import TSMixerConfig, LitTSMixer

        model_config = TSMixerConfig(**model_hp)
        model = LitTSMixer.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "ealstm":
        from src.models.ealstm import EALSTMConfig, LitEALSTM

        model_config = EALSTMConfig(**model_hp)
        model = LitEALSTM.load_from_checkpoint(checkpoint_path, config=model_config)
    elif model_type == "tft":
        from src.models.tft import TFTConfig, LitTFT

        model_config = TFTConfig(**model_hp)
        model = LitTFT.load_from_checkpoint(checkpoint_path, config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Adjust learning rate for fine-tuning if needed
    if finetune:
        original_lr = model.hparams.learning_rate
        model.hparams.learning_rate = original_lr / lr_factor

        # Store original learning rate for reference
        model.original_lr = original_lr
        model.fine_tuned_lr = original_lr / lr_factor

    return model, model_hp
