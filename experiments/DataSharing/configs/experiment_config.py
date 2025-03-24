"""Configuration for the data sharing experiment."""
import os
import sys
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[3]))

@dataclass
class DataSharingExperimentConfig:
    """Main configuration for the data sharing experiment."""
    
    # Experiment metadata
    experiment_name: str = "data_sharing_experiment"
    description: str = "Evaluating the impact of data sharing between Tajikistan and Kyrgyzstan"
    
    # Experiment structure
    scenarios: List[str] = field(default_factory=lambda: ["tajikistan", "kyrgyzstan", "combined"])
    model_types: List[str] = field(default_factory=lambda: ["tide", "tsmixer", "tft", "ealstm"])
    
    # Training parameters
    num_runs: int = 3  # Number of times to repeat with different seeds
    base_seed: int = 42  # Base seed for reproducibility
    max_epochs: int = 50
    
    # Early stopping configuration
    patience: int = 5
    min_delta: float = 0.001
    
    # Paths for outputs
    base_dir: Path = Path(__file__).resolve().parents[1]
    results_dir: Path = field(default_factory=lambda: Path(os.path.join(
        Path(__file__).resolve().parents[1], "results"
    )))
    logs_dir: Path = field(default_factory=lambda: Path(os.path.join(
        Path(__file__).resolve().parents[1], "results", "logs"
    )))
    checkpoints_dir: Path = field(default_factory=lambda: Path(os.path.join(
        Path(__file__).resolve().parents[1], "results", "checkpoints"
    )))
    
    # Hardware configuration
    accelerator: str = "cuda" if torch.cuda.is_available() else "cpu"
    devices: int = 1 if torch.cuda.is_available() else 1
    precision: str = "32-true"  # or "16-mixed" for mixed precision
    
    # Verbose output
    verbose: bool = True
    
    def get_scenario_dirs(self, scenario: str) -> Dict[str, Path]:
        """Get directories for a specific scenario.
        
        Args:
            scenario: One of 'tajikistan', 'kyrgyzstan', or 'combined'
            
        Returns:
            Dictionary with paths for logs and checkpoints
        """
        scenario_logs = self.logs_dir / scenario
        scenario_checkpoints = self.checkpoints_dir / scenario
        
        # Ensure directories exist
        scenario_logs.mkdir(parents=True, exist_ok=True)
        scenario_checkpoints.mkdir(parents=True, exist_ok=True)
        
        return {
            "logs": scenario_logs,
            "checkpoints": scenario_checkpoints
        }
    
    def get_run_seed(self, run_idx: int) -> int:
        """Get seed for a specific run.
        
        Args:
            run_idx: Index of the run (0-based)
            
        Returns:
            Seed for the run
        """
        return self.base_seed + run_idx
    
    def create_experiment_dirs(self) -> None:
        """Create all necessary directories for the experiment."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scenario-specific directories
        for scenario in self.scenarios:
            logs_dir = self.logs_dir / scenario
            checkpoints_dir = self.checkpoints_dir / scenario
            
            logs_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
