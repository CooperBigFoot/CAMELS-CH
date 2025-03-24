"""Configuration file for TiDE models in the Data Sharing experiment."""
import sys
from pathlib import Path
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[4]))

from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.model_evaluation.hp_from_yaml import hp_from_yaml
from src.models.tide import TiDEConfig, LitTiDE


@dataclass
class TiDEModelConfig:
    """Configuration for TiDE models in the data sharing experiment."""
    
    # Model identifier
    MODEL_TYPE: str = "tide"
    
    # Path to hyperparameters YAML file
    hyperparams_path: str = os.path.join(
        Path(__file__).resolve().parents[2], 
        "hyperparams", 
        "tide_best.yaml"
    )
    
    # Optional overrides for specific parameters
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    input_size: Optional[int] = None
    static_size: Optional[int] = None
    future_input_size: Optional[int] = None
    learning_rate: Optional[float] = None
    
    def get_model_config(self, 
                         input_len: int,
                         output_len: int, 
                         input_size: int, 
                         static_size: int,
                         future_input_size: int) -> TiDEConfig:
        """Create a TiDEConfig with hyperparameters from YAML.
        
        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence
            input_size: Number of input features
            static_size: Number of static features
            future_input_size: Number of future forcing features
            
        Returns:
            TiDEConfig object configured for the experiment
        """
        # Load hyperparameters from YAML file
        hp_dict = hp_from_yaml(self.MODEL_TYPE, self.hyperparams_path)
        
        # Override with any explicitly provided parameters
        hp_dict.update({
            "input_len": self.input_len or input_len,
            "output_len": self.output_len or output_len,
            "input_size": self.input_size or input_size,
            "static_size": self.static_size or static_size,
            "future_input_size": self.future_input_size or future_input_size,
        })
        
        if self.learning_rate is not None:
            hp_dict["learning_rate"] = self.learning_rate
            
        # Create and return the configuration
        return TiDEConfig(**hp_dict)
    
    def create_model(self, 
                    input_len: int,
                    output_len: int, 
                    input_size: int, 
                    static_size: int,
                    future_input_size: int) -> LitTiDE:
        """Create a LitTiDE model instance with proper configuration.
        
        Args:
            input_len: Length of input sequence
            output_len: Length of output sequence
            input_size: Number of input features
            static_size: Number of static features
            future_input_size: Number of future forcing features
            
        Returns:
            Configured LitTiDE model
        """
        config = self.get_model_config(
            input_len=input_len,
            output_len=output_len,
            input_size=input_size,
            static_size=static_size,
            future_input_size=future_input_size
        )
        
        return LitTiDE(config)
