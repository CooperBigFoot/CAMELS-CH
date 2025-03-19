"""Utility functions for hyperparameter tuning."""

import os
from pathlib import Path
from typing import Dict, Any
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour


def setup_dirs(model_type: str) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.
    
    Args:
        model_type: Type of model being tuned
        
    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path("experiments/HyperparameterTune")
    dirs = {
        "results": base_dir / "results" / model_type,
        "logs": base_dir / "logs" / model_type,
        "checkpoints": base_dir / "checkpoints" / model_type,
        "visualizations": base_dir / "visualizations" / model_type,
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs


def save_visualizations(study: optuna.Study, model_type: str, output_dir: Path) -> None:
    """Generate and save visualization plots for optimization results.
    
    Args:
        study: Completed Optuna study object
        model_type: Type of model that was tuned
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(str(output_dir / f"{model_type}_optimization_history.png"))
        
        # Plot parameter importance
        fig2 = plot_param_importances(study)
        fig2.write_image(str(output_dir / f"{model_type}_param_importances.png"))
        
        # Plot contour plots for top parameters if at least 2 parameters exist
        if len(study.best_trial.params) >= 2:
            fig3 = plot_contour(study)
            fig3.write_image(str(output_dir / f"{model_type}_param_contours.png"))
            
        # Generate and save parameter correlation heatmap
        # First make sure matplotlib is using a non-interactive backend
        plt.switch_backend('agg')
        
        # Create correlation heatmap using matplotlib
        params_df = study.trials_dataframe()
        # Filter params columns and remove NaNs
        params_cols = [c for c in params_df.columns if c.startswith('params_')]
        corr_df = params_df[params_cols + ['value']].dropna()
        
        if len(corr_df) > 5:  # Only create heatmap if we have enough data
            # Calculate correlation
            corr_matrix = corr_df.corr()
            
            plt.figure(figsize=(12, 10))
            plt.matshow(corr_matrix, fignum=1)
            plt.colorbar()
            
            # Add correlation values in the cells
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > 0.3:  # Only show significant correlations
                        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                                ha="center", va="center", size=8)
            
            # Add parameter names as ticks
            tick_labels = [col.replace('params_', '') for col in corr_matrix.columns]
            plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha='left')
            plt.yticks(range(len(tick_labels)), tick_labels)
            plt.title(f'{model_type} Parameter Correlation Heatmap')
            
            # Save the heatmap
            plt.tight_layout()
            plt.savefig(str(output_dir / f"{model_type}_param_correlation.png"))
            plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")


def save_study(study: optuna.Study, filepath: str) -> None:
    """Save an Optuna study to disk.
    
    Args:
        study: The study to save
        filepath: Path where to save the study
    """
    import joblib
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save study
    joblib.dump(study, filepath)


def load_study(filepath: str) -> optuna.Study:
    """Load an Optuna study from disk.
    
    Args:
        filepath: Path to the saved study
        
    Returns:
        Loaded Optuna study object
        
    Raises:
        FileNotFoundError: If the study file doesn't exist
    """
    import joblib
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Study file not found: {filepath}")
    
    return joblib.load(filepath)


def generate_optimization_report(study: optuna.Study, model_type: str, output_dir: Path) -> None:
    """Generate a comprehensive optimization report in Markdown format.
    
    Args:
        study: Completed Optuna study
        model_type: Type of model that was tuned
        output_dir: Directory to save the report
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare report content
    report = [
        f"# Hyperparameter Optimization Report for {model_type.upper()}",
        "",
        f"Date: {study.user_attrs.get('date', 'N/A')}",
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
    
    # Add visualization references
    report.append("")
    report.append("## Visualizations")
    report.append("")
    report.append(f"- [Optimization History](./{model_type}_optimization_history.png)")
    report.append(f"- [Parameter Importance](./{model_type}_param_importances.png)")
    report.append(f"- [Parameter Contours](./{model_type}_param_contours.png)")
    report.append(f"- [Parameter Correlation](./{model_type}_param_correlation.png)")
    
    # Write report to file
    report_path = output_dir / f"{model_type}_optimization_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {report_path}")
