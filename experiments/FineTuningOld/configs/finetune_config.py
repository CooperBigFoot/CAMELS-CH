"""Configuration for fine-tuning pre-trained models on Central Asian data."""

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from experiments.DataSharingOld.configs.base_config import BaseDataSharingConfig


@dataclass
class FineTuningConfig(BaseDataSharingConfig):
    """Configuration for fine-tuning pre-trained hydrological models.

    This class extends the base data sharing configuration with fine-tuning
    specific parameters for adapting pre-trained models to Central Asian basins.
    """

    # Fine-tuning specific parameters
    CHECKPOINT_PATH: str = ""  # Path to pre-trained model checkpoint
    MODEL_TYPE: str = ""  # Type of model (tide, tsmixer, ealstm, tft)
    TARGET_COUNTRY: str = (
        ""  # Country to fine-tune on (Tajikistan, Kyrgyzstan, Combined)
    )
    YAML_PATH: str = ""  # Path to model hyperparameter YAML file

    # Fine-tuning settings
    OUTPUT_DIR: str = "experiments/FineTuning/results"
    LR_FACTOR: float = 10.0  # Factor to reduce learning rate by
    MAX_EPOCHS: int = 100

    # Early stopping settings
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_MIN_DELTA: float = 0.0001

    # Checkpointing settings
    SAVE_TOP_K: int = 1
    SAVE_LAST: bool = True

    def get_checkpoint_dir(self) -> Path:
        """Get directory to save fine-tuned model checkpoints."""
        return (
            Path(self.OUTPUT_DIR)
            / "checkpoints"
            / self.TARGET_COUNTRY.lower()
            / self.MODEL_TYPE
        )

    def get_logs_dir(self) -> Path:
        """Get directory to save fine-tuning logs."""
        return (
            Path(self.OUTPUT_DIR)
            / "logs"
            / self.TARGET_COUNTRY.lower()
            / self.MODEL_TYPE
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.CHECKPOINT_PATH:
            raise ValueError("Checkpoint path must be provided")

        if not self.MODEL_TYPE:
            raise ValueError("Model type must be provided")

        if self.MODEL_TYPE.lower() not in ["tide", "tsmixer", "ealstm", "tft"]:
            raise ValueError(f"Unsupported model type: {self.MODEL_TYPE}")

        if not self.TARGET_COUNTRY:
            raise ValueError("Target country must be provided")

        if self.TARGET_COUNTRY.lower() not in ["tajikistan", "kyrgyzstan", "combined"]:
            raise ValueError(f"Unsupported country: {self.TARGET_COUNTRY}")

        if not self.YAML_PATH or not Path(self.YAML_PATH).exists():
            raise ValueError(f"Invalid YAML path: {self.YAML_PATH}")

        if self.LR_FACTOR <= 0:
            raise ValueError(
                f"Learning rate factor must be positive, got {self.LR_FACTOR}"
            )
