"""Main script to run all data sharing experiments.

This script orchestrates the complete experiment, sequentially running training on:
1. Tajikistan data only
2. Kyrgyzstan data only
3. Combined data from both countries

The script provides a comprehensive interface for controlling which models and scenarios
to run, with appropriate parameter validation and progress tracking.
"""
import os
import sys
from pathlib import Path
import argparse
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.DataSharing.configs.experiment_config import DataSharingExperimentConfig
from experiments.DataSharing.configs.data_config import DataSharingDataConfig
from experiments.DataSharing.scripts.train_tajikistan import train_tajikistan_models
from experiments.DataSharing.scripts.train_kyrgyzstan import train_kyrgyzstan_models
from experiments.DataSharing.scripts.train_combined import train_combined_models


def setup_main_logger() -> logging.Logger:
    """Set up the main logger for the experiment runner.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("data_sharing_experiment")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"experiment_run_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def run_experiments(
    scenarios: List[str],
    model_types: Optional[List[str]] = None,
    runs: Optional[int] = None,
    data_config_overrides: Optional[Dict[str, Any]] = None,
    experiment_config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Run the data sharing experiments for specified scenarios and models.
    
    Args:
        scenarios: List of scenarios to run ('tajikistan', 'kyrgyzstan', 'combined')
        model_types: List of model types to train (default: all available models)
        runs: Number of runs to execute (default: from experiment config)
        data_config_overrides: Optional overrides for data configuration parameters
        experiment_config_overrides: Optional overrides for experiment configuration parameters
        
    Returns:
        Nested dictionary of results with structure {scenario: {model_type: {run_idx: val_loss}}}
    """
    # Set up logger
    logger = setup_main_logger()
    logger.info("Starting Data Sharing Experiments")
    
    # Initialize configurations
    experiment_config = DataSharingExperimentConfig()
    data_config = DataSharingDataConfig()
    
    # Apply overrides if provided
    if experiment_config_overrides:
        for key, value in experiment_config_overrides.items():
            if hasattr(experiment_config, key):
                setattr(experiment_config, key, value)
                logger.info(f"Overriding experiment config: {key} = {value}")
            else:
                logger.warning(f"Unknown experiment config parameter: {key}")
    
    if data_config_overrides:
        for key, value in data_config_overrides.items():
            if hasattr(data_config, key):
                setattr(data_config, key, value)
                logger.info(f"Overriding data config: {key} = {value}")
            else:
                logger.warning(f"Unknown data config parameter: {key}")
    
    # Validate scenarios
    valid_scenarios = ["tajikistan", "kyrgyzstan", "combined"]
    scenarios = [s.lower() for s in scenarios]
    invalid_scenarios = [s for s in scenarios if s not in valid_scenarios]
    if invalid_scenarios:
        raise ValueError(f"Invalid scenarios: {', '.join(invalid_scenarios)}. " 
                        f"Must be one or more of: {', '.join(valid_scenarios)}")
    
    # Process model types
    if model_types is None:
        model_types = experiment_config.model_types
    elif "all" in model_types:
        model_types = experiment_config.model_types
    
    # Process runs
    run_indices = None
    if runs is not None:
        run_indices = list(range(runs))
    
    # Create experiment directories
    experiment_config.create_experiment_dirs()
    
    # Track results
    all_results = {}
    
    # Helper function to log experiment progress
    def log_scenario_progress(scenario: str, status: str):
        separator = "=" * 50
        logger.info(separator)
        logger.info(f"{status} {scenario.upper()} EXPERIMENT")
        logger.info(separator)
    
    # Run each scenario in sequence
    total_scenarios = len(scenarios)
    
    for idx, scenario in enumerate(scenarios):
        scenario_start_time = time.time()
        log_scenario_progress(scenario, "STARTING")
        logger.info(f"Progress: Scenario {idx + 1}/{total_scenarios}")
        
        try:
            # Run appropriate training function based on scenario
            if scenario == "tajikistan":
                results = train_tajikistan_models(
                    experiment_config=experiment_config,
                    data_config=data_config,
                    model_types=model_types,
                    runs=run_indices
                )
            elif scenario == "kyrgyzstan":
                results = train_kyrgyzstan_models(
                    experiment_config=experiment_config,
                    data_config=data_config,
                    model_types=model_types,
                    runs=run_indices
                )
            elif scenario == "combined":
                results = train_combined_models(
                    experiment_config=experiment_config,
                    data_config=data_config,
                    model_types=model_types,
                    runs=run_indices
                )
            else:
                logger.error(f"Unrecognized scenario: {scenario}")
                continue
                
            # Store results
            all_results[scenario] = results
            
            # Log completion
            scenario_end_time = time.time()
            scenario_duration = scenario_end_time - scenario_start_time
            log_scenario_progress(scenario, "COMPLETED")
            logger.info(f"Duration: {scenario_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in {scenario} experiment: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            log_scenario_progress(scenario, "FAILED")
    
    # Print final summary
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    
    for scenario in scenarios:
        if scenario in all_results:
            logger.info(f"Results for {scenario.upper()}:")
            for model_type, run_results in all_results[scenario].items():
                average_loss = sum(run_results.values()) / len(run_results) if run_results else float('inf')
                logger.info(f"  {model_type}: {average_loss:.4f} (average of {len(run_results)} runs)")
        else:
            logger.info(f"No results for {scenario.upper()} (execution failed)")
    
    return all_results


def compare_results(all_results: Dict[str, Dict[str, Dict[int, float]]]) -> None:
    """Compare results across different scenarios.
    
    Args:
        all_results: Nested dictionary of results with structure {scenario: {model_type: {run_idx: val_loss}}}
    """
    logger = logging.getLogger("data_sharing_experiment")
    
    # Only compare if we have both individual and combined results
    scenarios = ["tajikistan", "kyrgyzstan", "combined"]
    if not all(scenario in all_results for scenario in scenarios):
        logger.warning("Cannot compare results: not all scenarios completed successfully")
        return
    
    logger.info("=" * 50)
    logger.info("COMPARING RESULTS ACROSS SCENARIOS")
    logger.info("=" * 50)
    
    # For each model type, compare average validation loss across scenarios
    model_types = set()
    for scenario_results in all_results.values():
        model_types.update(scenario_results.keys())
    
    for model_type in sorted(model_types):
        logger.info(f"Model: {model_type.upper()}")
        
        scenario_averages = {}
        for scenario in scenarios:
            if model_type in all_results[scenario]:
                run_results = all_results[scenario][model_type]
                average_loss = sum(run_results.values()) / len(run_results) if run_results else float('inf')
                scenario_averages[scenario] = average_loss
        
        # Determine if combined data improves or worsens results
        for country in ["tajikistan", "kyrgyzstan"]:
            if country in scenario_averages and "combined" in scenario_averages:
                country_loss = scenario_averages[country]
                combined_loss = scenario_averages["combined"]
                
                if combined_loss < country_loss:
                    improvement = ((country_loss - combined_loss) / country_loss) * 100
                    logger.info(f"  {country.title()} benefits from combined data: " 
                               f"{improvement:.2f}% improvement")
                else:
                    worsening = ((combined_loss - country_loss) / country_loss) * 100
                    logger.info(f"  {country.title()} performs better with only its own data: "
                               f"{worsening:.2f}% worse with combined data")
        
        # Print actual values for reference
        for scenario, avg_loss in scenario_averages.items():
            logger.info(f"  {scenario.title()}: {avg_loss:.4f}")
        
        logger.info("")  # Add blank line between models
    
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete data sharing experiment suite")
    
    # Scenarios to run
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["tajikistan", "kyrgyzstan", "combined", "all"],
        default=["all"],
        help="List of scenarios to run"
    )
    
    # Models to train
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tide", "tsmixer", "tft", "ealstm", "all"],
        default=["all"],
        help="List of models to train"
    )
    
    # Number of runs
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of runs (default: use value from experiment config)"
    )
    
    # Data configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Input sequence length (lookback window)"
    )
    
    parser.add_argument(
        "--output-length",
        type=int,
        default=None,
        help="Output sequence length (forecast horizon)"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process scenarios argument
    scenarios = args.scenarios
    if "all" in scenarios:
        scenarios = ["tajikistan", "kyrgyzstan", "combined"]
    
    # Process models argument
    models = args.models if "all" not in args.models else None
    
    # Collect data config overrides
    data_config_overrides = {}
    if args.batch_size is not None:
        data_config_overrides["batch_size"] = args.batch_size
    if args.input_length is not None:
        data_config_overrides["input_length"] = args.input_length
    if args.output_length is not None:
        data_config_overrides["output_length"] = args.output_length
    
    # Collect experiment config overrides
    experiment_config_overrides = {}
    if args.max_epochs is not None:
        experiment_config_overrides["max_epochs"] = args.max_epochs
    if args.patience is not None:
        experiment_config_overrides["patience"] = args.patience
    
    # Run the experiments
    try:
        start_time = time.time()
        results = run_experiments(
            scenarios=scenarios,
            model_types=models,
            runs=args.runs,
            data_config_overrides=data_config_overrides or None,
            experiment_config_overrides=experiment_config_overrides or None
        )
        
        # Compare results across scenarios if appropriate
        if len(scenarios) > 1:
            compare_results(results)
            
        # Log total execution time
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger = logging.getLogger("data_sharing_experiment")
        logger.info(f"Total experiment duration: {total_duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
