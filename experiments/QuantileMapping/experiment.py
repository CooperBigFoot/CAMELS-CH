"""
Main script for running the quantile mapping experiment.

This experiment evaluates whether using quantile mapped meteorological forcing data
improves hydrological model performance compared to the original forcing data. Both
data sources use the same reduced feature set containing only temperature_2m_mean and
total_precipitation_sum.
"""

import sys
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import QuantileMappingConfig
from data_loader import load_data
from utils import (
    setup_dirs,
    train_and_save_model,
    save_experiment_results,
    load_model_configs_and_datamodules,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantile Mapping Experiment")

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tide", "tsmixer", "ealstm", "tft"],
        help="Model types to evaluate",
    )

    # YAML paths for hyperparameters
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="experiments/QuantileMapping/yaml_files",
        help="Directory containing model YAML files",
    )

    # Data source selection
    parser.add_argument(
        "--data-sources",
        type=str,
        nargs="+",
        default=["original", "quantile_mapped"],
        help="Data sources to use (original and/or quantile_mapped)",
    )

    parser.add_argument(
        "--quantile-mapped-folder",
        type=str,
        required=False,
        help="Path to folder containing quantile mapped timeseries data (required if using 'quantile_mapped')",
    )

    # Experiment settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/QuantileMapping/output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=1, 
        help="Number of runs for each model"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Base random seed for reproducibility"
    )

    # Optional feature override
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="Override default forcing features (defaults to temperature_2m_mean and total_precipitation_sum)",
    )
    
    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training (overrides config)",
    )

    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=100, 
        help="Maximum number of training epochs"
    )

    return parser.parse_args()


def main():
    """Run the quantile mapping experiment."""
    # Parse command line arguments
    args = parse_args()

    # Set global seed for reproducibility
    pl.seed_everything(args.seed)

    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Load configuration
    config = QuantileMappingConfig()
    config.data_sources = args.data_sources
    config.quantile_mapped_folder = args.quantile_mapped_folder
    config.output_dir = args.output_dir
    config.num_runs = args.num_runs
    config.batch_size = args.batch_size
    config.max_epochs = args.max_epochs
    config.model_types = args.models
    
    # Override forcing features if specified
    if args.features:
        config.forcing_features = args.features

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Setup directories
    setup_dirs(config.output_dir)
    
    # Prepare YAML paths for models
    yaml_dir = Path(args.yaml_dir)
    yaml_paths = {
        model_type: str(yaml_dir / f"{model_type}.yaml") for model_type in args.models
    }
    
    # Check that YAML files exist
    for model_type, yaml_path in yaml_paths.items():
        if not Path(yaml_path).exists():
            print(f"YAML file for {model_type} not found: {yaml_path}")
            sys.exit(1)

    # Process each data source
    all_results = {}
    
    for data_source in args.data_sources:
        print(f"\n{'=' * 50}")
        print(f"Processing {data_source} data")
        print(f"{'=' * 50}\n")

        try:
            # Load data for this source
            data = load_data(
                config=config,
                data_source=data_source,
            )
            
            # Load model configurations and create DataModules
            model_configs, data_modules = load_model_configs_and_datamodules(
                time_series_data=data["time_series"],
                static_data=data["static"],
                config=config,
                yaml_paths=yaml_paths,
            )
            
            # Initialize results for this data source
            data_source_results = []
            
            # Train each model
            for model_type in args.models:
                if model_type not in model_configs:
                    print(f"Skipping unknown model type: {model_type}")
                    continue
                
                print(f"\nTraining {model_type.upper()} model on {data_source} data")
                
                # Get model-specific configuration and DataModule
                model_config = model_configs[model_type]
                data_module = data_modules[model_type]
                
                # Train the model
                model_results = train_and_save_model(
                    model_type=model_type,
                    model_config=model_config,
                    data_module=data_module,
                    data_source=data_source,
                    output_dir=config.output_dir,
                    num_runs=config.num_runs,
                    early_stopping_patience=config.early_stopping_patience,
                    save_top_k=config.save_top_k,
                    save_last=config.save_last,
                )
                
                # Add results
                data_source_results.extend(model_results)
                
                # Clean up CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Store results for this data source
            all_results[data_source] = data_source_results
            
            # Save results for this data source
            save_experiment_results(data_source_results, config.output_dir, data_source)
            
        except Exception as e:
            print(f"Error processing {data_source} data: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
