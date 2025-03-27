"""Utility functions for hyperparameter tuning."""

from pathlib import Path
from typing import Dict, Any
import optuna
import yaml
import pandas as pd
from datetime import datetime


def setup_dirs(config: Any) -> Dict[str, Dict[str, Path]]:
    """Create and return necessary directories for experiment outputs.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Nested dictionary of Path objects for different directories by country and model type
    """
    base_dir = Path(config.output_dir)
    
    # Define directory structure
    dirs = {
        country.lower(): {
            "checkpoints": {},
            "logs": {},
            "results": {},
        }
        for country in config.countries
    }

    # Create directories for each country and model type
    for country in config.countries:
        country_lower = country.lower()
        
        for model_type in config.model_types:
            # Checkpoint directories
            checkpoint_dir = base_dir / "checkpoints" / country_lower / model_type
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            dirs[country_lower]["checkpoints"][model_type] = checkpoint_dir
            
            # Log directories
            log_dir = base_dir / "logs" / country_lower / model_type
            log_dir.mkdir(parents=True, exist_ok=True)
            dirs[country_lower]["logs"][model_type] = log_dir
            
            # Results directories
            results_dir = base_dir / "results" / country_lower / model_type
            results_dir.mkdir(parents=True, exist_ok=True)
            dirs[country_lower]["results"][model_type] = results_dir
    
    return dirs


def save_study_results(study: optuna.Study, model_type: str, country: str, output_dir: Path) -> None:
    """Save study results to CSV and generate a report.
    
    Args:
        study: Completed Optuna study
        model_type: Type of model that was tuned
        country: Country for which tuning was performed
        output_dir: Directory to save results
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results dataframe
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "best_epoch": trial.user_attrs.get("best_epoch", None),
                # Include dataset sizes in results
                "train_size": trial.user_attrs.get("train_size", None),
                "val_size": trial.user_attrs.get("val_size", None),
                "test_size": trial.user_attrs.get("test_size", None),
                **trial.params,  # Includes all hyperparameters
            }
            results.append(trial_data)
    
    if not results:
        print("No completed trials to save.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = output_dir / f"{model_type}_optimization_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save best parameters separately to yaml
    best_params = study.best_trial.params
    best_value = study.best_trial.value
    best_params_yaml = {
        "best_value": best_value,
        "country": country,
        "model_type": model_type,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": best_params
    }
    
    yaml_path = output_dir / f"{model_type}_best_parameters.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(best_params_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"Best parameters saved to {yaml_path}")
    
    # Generate optimization report
    generate_optimization_report(study, model_type, country, output_dir)


def generate_optimization_report(
    study: optuna.Study, model_type: str, country: str, output_dir: Path
) -> None:
    """Generate a comprehensive optimization report in Markdown format.
    
    Args:
        study: Completed Optuna study
        model_type: Type of model that was tuned
        country: Country for which tuning was performed
        output_dir: Directory to save the report
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare report content
    report = [
        f"# Hyperparameter Optimization Report for {model_type.upper()} - {country}",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of trials: {len(study.trials)}",
        f"Best trial: #{study.best_trial.number}",
        "",
        "## Best Parameters",
        "",
        "| Parameter | Value |",
        "| --------- | ----- |",
    ]
    
    # Add best parameters
    for param_name, param_value in study.best_trial.params.items():
        report.append(f"| {param_name} | {param_value} |")
    
    # Add best value
    report.append("")
    report.append(f"**Best validation loss**: {study.best_trial.value:.6f}")
    
    # Add information about the best trial
    report.append("")
    report.append("## Best Trial Details")
    report.append("")
    
    # Add user attributes if available
    if hasattr(study.best_trial, 'user_attrs') and study.best_trial.user_attrs:
        report.append("| Attribute | Value |")
        report.append("| --------- | ----- |")
        for attr_name, attr_value in study.best_trial.user_attrs.items():
            report.append(f"| {attr_name} | {attr_value} |")
    
    # Add parameter importance if available
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            report.append("")
            report.append("## Parameter Importance")
            report.append("")
            report.append("| Parameter | Importance |")
            report.append("| --------- | ---------- |")
            for param_name, importance in importances.items():
                report.append(f"| {param_name} | {importance:.4f} |")
    except Exception as e:
        report.append("")
        report.append("Parameter importance calculation failed.")
        report.append(f"Error: {str(e)}")
    
    # Add country-specific information
    report.append("")
    report.append("## Dataset Information")
    report.append("")
    report.append(f"- **Country**: {country}")
    
    if hasattr(study.best_trial, 'user_attrs'):
        basin_count = study.best_trial.user_attrs.get("basin_count", "N/A")
        train_size = study.best_trial.user_attrs.get("train_size", "N/A")
        val_size = study.best_trial.user_attrs.get("val_size", "N/A")
        test_size = study.best_trial.user_attrs.get("test_size", "N/A")
        
        report.append(f"- **Number of basins**: {basin_count}")
        report.append(f"- **Training samples**: {train_size}")
        report.append(f"- **Validation samples**: {val_size}")
        report.append(f"- **Testing samples**: {test_size}")
    
    # Write report to file
    report_path = output_dir / f"{model_type}_optimization_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    
    print(f"Report saved to {report_path}")
