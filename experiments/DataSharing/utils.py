"""Utility functions for the Central Asian data sharing experiment."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_models.datamodule import HydroDataModule
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.model_evaluation.hp_from_yaml import hp_from_yaml


def setup_dirs(output_dir: str) -> Dict[str, Path]:
    """Create and return necessary directories for experiment outputs.

    Args:
        output_dir: Base output directory path

    Returns:
        Dictionary of Path objects for different directories
    """
    base_dir = Path(output_dir)

    # Define directory structure
    dirs = {
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }

    # Create directories for each country
    countries = ["tajikistan", "kyrgyzstan", "combined"]
    model_types = ["tide", "tsmixer", "ealstm", "tft"]

    for country in countries:
        for model_type in model_types:
            (dirs["checkpoints"] / country / model_type).mkdir(
                parents=True, exist_ok=True
            )
            (dirs["logs"] / country / model_type).mkdir(parents=True, exist_ok=True)
            (dirs["results"] / country / model_type).mkdir(parents=True, exist_ok=True)

    return dirs


def load_country_data(config: Any, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and prepare datasets from Central Asia with optional country filtering.

    Args:
        config: Configuration object with dataset paths
        country: Optional country filter ('Tajikistan', 'Kyrgyzstan', or None)
                If None or 'Combined', no country filtering is applied

    Returns:
        Dictionary containing time series and static data frames
    """
    # Configure Caravan dataset
    ca_config = CaravanifyConfig(
        attributes_dir=config.CA_CONFIG["ATTRIBUTE_DIR"],
        timeseries_dir=config.CA_CONFIG["TIMESERIES_DIR"],
        gauge_id_prefix=config.CA_CONFIG["GAUGE_ID_PREFIX"],
        human_influence_path=config.CA_CONFIG["HUMAN_INFLUENCE_PATH"],
        use_hydroatlas_attributes=True,
        use_caravan_attributes=True,
        use_other_attributes=True,
    )

    # Initialize and load all basins
    ca_caravan = Caravanify(ca_config)
    ca_basins = ca_caravan.get_all_gauge_ids()

    # Apply human influence filtering
    print(f"Found {len(ca_basins)} total CA basins")
    ca_basins, discarded_ca = ca_caravan.filter_gauge_ids_by_human_influence(
        ca_basins, ["Low", "Medium"]
    )
    print(f"Loading {len(ca_basins)} CA basins after human influence filtering")
    print(f"Discarded {len(discarded_ca)} CA basins with high human influence")

    ca_caravan.load_stations(ca_basins)

    # Prepare columns
    ts_columns = config.FORCING_FEATURES + [config.TARGET]
    static_columns = config.STATIC_FEATURES
    ts_columns_with_date = ts_columns + ["date"] + [config.GROUP_IDENTIFIER]

    # Get all data
    all_ts_data = ca_caravan.get_time_series()[ts_columns_with_date]
    all_static_data = ca_caravan.get_static_attributes()[static_columns]

    # Apply country filtering if specified
    if country and country.lower() != "combined":
        print(f"Filtering data for country: {country}")

        # Get gauge IDs for specified country
        country_gauge_ids = all_static_data[all_static_data["country"] == country][
            config.GROUP_IDENTIFIER
        ].unique()

        if len(country_gauge_ids) == 0:
            raise ValueError(f"No basins found for country: {country}")

        # Filter time series and static data
        ts_data = all_ts_data[
            all_ts_data[config.GROUP_IDENTIFIER].isin(country_gauge_ids)
        ]
        static_data = all_static_data[
            all_static_data[config.GROUP_IDENTIFIER].isin(country_gauge_ids)
        ]

        print(f"Selected {len(country_gauge_ids)} basins in {country}")
    else:
        # Use all data
        ts_data = all_ts_data
        static_data = all_static_data
        print(f"Using all {len(ca_basins)} basins (Combined dataset)")

    # Return the filtered data
    return {
        "time_series": ts_data,
        "static": static_data,
        "basin_count": len(static_data[config.GROUP_IDENTIFIER].unique()),
    }


def load_model_configs_and_datamodules(
    time_series_data: pd.DataFrame,
    static_data: pd.DataFrame,
    config: Any,
    yaml_paths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, HydroDataModule]]:
    """
    Load model configurations from pre-tuned YAML files and create model-specific DataModules.

    Args:
        time_series_data: Time series data for the DataModule
        static_data: Static data for the DataModule
        config: Base configuration with common parameters
        yaml_paths: Dictionary mapping model types to their YAML paths

    Returns:
        Tuple containing:
        - Dictionary mapping model types to their configurations
        - Dictionary mapping model types to their DataModules
    """
    # Initialize dictionaries
    configs = {}
    data_modules = {}

    # Process each model type
    for model_type, yaml_path in yaml_paths.items():
        if not yaml_path or not os.path.exists(yaml_path):
            print(f"Warning: YAML path for {model_type} not found: {yaml_path}")
            continue

        try:
            # Load hyperparameters from YAML
            model_hp = hp_from_yaml(model_type, yaml_path)

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
                print(f"Warning: Unsupported model type: {model_type}")
                continue

            # Store the configuration
            configs[model_type] = model_config

            # Extract model-specific parameters for DataModule
            input_length = model_config.input_len
            output_length = model_config.output_len

            # Use config's batch size if model_config doesn't have it
            batch_size = getattr(model_config, "batch_size", config.BATCH_SIZE)

            # Get preprocessing config
            preprocessing_config = config.get_preprocessing_config()

            # Create DataModule with model-specific parameters
            data_modules[model_type] = HydroDataModule(
                time_series_df=time_series_data,
                static_df=static_data,
                group_identifier=config.GROUP_IDENTIFIER,
                preprocessing_config=preprocessing_config,
                input_length=input_length,
                output_length=output_length,
                batch_size=batch_size,
                num_workers=min(config.MAX_WORKERS, os.cpu_count()),
                features=config.FORCING_FEATURES + [config.TARGET],
                static_features=[f for f in config.STATIC_FEATURES if f != "country"],
                target=config.TARGET,
                use_proportional_split=config.USE_PROPORTIONAL_SPLIT,
                train_prop=config.TRAIN_PROP,
                val_prop=config.VAL_PROP,
                test_prop=config.TEST_PROP,
                min_train_years=config.CA_CONFIG.get("MIN_TRAIN_YEARS", 5),
            )

            print(
                f"Created DataModule for {model_type} with input_length={input_length}, "
                f"output_length={output_length}, batch_size={batch_size}"
            )

        except Exception as e:
            print(f"Error creating configuration for {model_type}: {str(e)}")
            continue

    return configs, data_modules


def prepare_country_scenario(
    country: str, config: Any, yaml_paths: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare model configs and DataModules for a specific country scenario.

    Args:
        country: Country to filter data for ('Tajikistan', 'Kyrgyzstan', or 'Combined')
        config: Base configuration
        yaml_paths: Dictionary mapping model types to their YAML paths

    Returns:
        Dictionary with model configurations and data modules for each model type
    """
    # Load country-specific data
    data = load_country_data(config, country)
    time_series_data = data["time_series"]
    static_data = data["static"]

    # Load model configurations and create DataModules
    model_configs, data_modules = load_model_configs_and_datamodules(
        time_series_data=time_series_data,
        static_data=static_data,
        config=config,
        yaml_paths=yaml_paths,
    )

    return {"model_configs": model_configs, "data_modules": data_modules, "data": data}


def train_and_save_model(
    model_type: str,
    model_config: Any,
    data_module: HydroDataModule,
    country: str,
    output_dir: str,
    num_runs: int = 3,
    early_stopping_patience: int = 5,
    save_top_k: int = 1,
    save_last: bool = True,
) -> Dict[str, Any]:
    """
    Train a model for a specific country scenario and save checkpoints.

    Args:
        model_type: Type of model to train ('tide', 'tsmixer', etc.)
        model_config: Configuration for the model
        data_module: DataModule for the model
        country: Country scenario ('Tajikistan', 'Kyrgyzstan', or 'Combined')
        output_dir: Base output directory
        num_runs: Number of training runs with different seeds
        early_stopping_patience: Patience for early stopping
        save_top_k: Number of best models to save
        save_last: Whether to save the last model checkpoint

    Returns:
        Dictionary with training results
    """
    from experiments.DataSharing.model_factory import ModelFactory

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Create output directories
    checkpoint_dir = Path(output_dir) / "checkpoints" / country.lower() / model_type
    logs_dir = Path(output_dir) / "logs" / country.lower() / model_type

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Results dictionary
    results = {"runs": [], "best_val_loss": float("inf"), "best_run": None}

    # Train for multiple runs
    for run in range(num_runs):
        print(f"\nStarting run {run + 1}/{num_runs} for {model_type} on {country} data")

        # Set seed for reproducibility
        seed = 42 + run
        pl.seed_everything(seed)

        # Create model
        model = ModelFactory.create_model(model_config, model_type)

        # Setup logger
        logger = TensorBoardLogger(
            save_dir=str(logs_dir), name=f"{country}_{model_type}", version=f"run_{run}"
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, mode="min"
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=str(checkpoint_dir / f"run_{run}"),
                filename=f"{country}_{model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=save_top_k,
                save_last=save_last,
            ),
        ]

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=50,  # Use a reasonable default, should be in config
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=True,
        )

        # Train model
        try:
            trainer.fit(model, data_module)

            # Get the best validation loss
            best_val_loss = trainer.callback_metrics.get(
                "val_loss", torch.tensor(float("inf"))
            ).item()

            # Store run results
            run_results = {
                "run": run,
                "seed": seed,
                "best_val_loss": best_val_loss,
                "best_epoch": trainer.current_epoch,
                "checkpoint_path": str(
                    checkpoint_dir
                    / f"run_{run}"
                    / f"{country}_{model_type}_epoch={trainer.current_epoch:02d}_val_loss={best_val_loss:.4f}.ckpt"
                ),
            }

            results["runs"].append(run_results)

            # Update best run if needed
            if best_val_loss < results["best_val_loss"]:
                results["best_val_loss"] = best_val_loss
                results["best_run"] = run_results

            print(
                f"Run {run + 1} completed with best val_loss: {best_val_loss:.4f} at epoch {trainer.current_epoch}"
            )

        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Log the error and continue with the next run
            results["runs"].append({"run": run, "seed": seed, "error": str(e)})

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return results summary
    return results


def save_experiment_results(
    results: Dict[str, Any], output_dir: str, country: str
) -> None:
    """
    Save experiment results to CSV files.

    Args:
        results: Dictionary with results for different model types
        output_dir: Base output directory
        country: Country scenario
    """
    results_dir = Path(output_dir) / "results" / country.lower()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create summary dataframe
    summary_rows = []

    for model_type, model_results in results.items():
        if "best_run" in model_results and model_results["best_run"]:
            best_run = model_results["best_run"]
            summary_rows.append(
                {
                    "model_type": model_type,
                    "country": country,
                    "best_val_loss": best_run["best_val_loss"],
                    "best_epoch": best_run["best_epoch"],
                    "checkpoint_path": best_run["checkpoint_path"],
                }
            )

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(results_dir / "summary.csv", index=False)

    # Save detailed results for each model
    for model_type, model_results in results.items():
        if "runs" in model_results and model_results["runs"]:
            runs_df = pd.DataFrame(model_results["runs"])
            runs_df.to_csv(results_dir / f"{model_type}_runs.csv", index=False)
