"""Main script for running the hyperparameter tuning experiment."""

import argparse
import torch
import pytorch_lightning as pl
from datetime import datetime

from config import ExperimentConfig
from utils import setup_dirs
from tuner import HyperparameterTuner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning Experiment for Hydrological Models"
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        choices=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to tune",
    )
    
    # Country selection
    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=["Tajikistan", "Kyrgyzstan", "Combined"],
        help="Countries to tune for",
    )
    
    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/HyperparameterTune/output",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials per model/country combination",
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Base random seed for reproducibility"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=50, 
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for optimization in seconds (None for no timeout)",
    )
    
    return parser.parse_args()


def main():
    """Run the hyperparameter tuning experiment."""
    # Parse command line arguments
    args = parse_args()
    
    # Set global seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    
    # Load configuration
    config = ExperimentConfig()
    config.model_types = args.models
    config.countries = args.countries
    config.output_dir = args.output_dir
    config.n_trials = args.n_trials
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.timeout = args.timeout
    
    # Validate configuration
    config.validate()
    
    # Setup directories
    dirs = setup_dirs(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run optimization for each combination of country and model
    for country in args.countries:
        print(f"\n{'=' * 60}")
        print(f"Processing {country} scenario")
        print(f"{'=' * 60}\n")
        
        for model_type in args.models:
            print(f"\n{'-' * 50}")
            print(f"Tuning {model_type.upper()} for {country}")
            print(f"{'-' * 50}\n")
            
            try:
                # Create study name with timestamp
                study_name = f"{model_type}_{country.lower()}_{timestamp}"
                
                # Create and run tuner
                tuner = HyperparameterTuner(
                    config=config,
                    model_type=model_type,
                    country=country,
                    study_name=study_name,
                    dirs=dirs
                )
                
                # Set new seed for each model/country combination
                config.set_seed(args.seed + hash(f"{country}_{model_type}") % 10000)
                
                # Load data (done here to catch any data loading errors early)
                tuner.load_data()
                
                # Run optimization
                study = tuner.run_optimization()
                
                # Clean up
                tuner.cleanup()
                
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error tuning {model_type} for {country}: {str(e)}")
                continue
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
