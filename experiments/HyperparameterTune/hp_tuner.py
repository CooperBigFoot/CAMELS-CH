"""Main script for hyperparameter tuning of hydrological forecasting models."""

import os
import sys
from pathlib import Path
import argparse
import time
import json
import pandas as pd
import numpy as np
import torch
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.data_models.datamodule import HydroDataModule
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from experiments.HyperparameterTune.model_factory import ModelFactory
from experiments.HyperparameterTune.utils import save_visualizations, setup_dirs


class HyperparameterTuner:
    """Hyperparameter tuning framework for hydrological forecasting models."""

    def __init__(self, config, model_type, study_name=None, n_trials=20):
        """Initialize the hyperparameter tuner.

        Args:
            config: Configuration object for tuning
            model_type: Type of model to tune ('tide', 'tsmixer', etc.)
            study_name: Optional name for the Optuna study
            n_trials: Number of optimization trials to run
        """
        self.config = config
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.study_name = study_name or f"{self.model_type}_optimization"

        # Setup directories
        self.dirs = setup_dirs(model_type)

        # Configure optuna study
        self.sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="minimize", study_name=self.study_name, sampler=self.sampler
        )

        # Flag to track if data has been loaded
        self._data_loaded = False

    def load_data(self):
        """Load and prepare datasets from Central Asia with human influence filtering."""
        # CA Dataset
        print("CONFIGURING CA DATASET")
        ca_config = CaravanifyConfig(
            attributes_dir=self.config.CA_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.CA_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.CA_CONFIG["GAUGE_ID_PREFIX"],
            human_influence_path=self.config.CA_CONFIG["HUMAN_INFLUENCE_PATH"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        self.ca_caravan = Caravanify(ca_config)
        ca_basins = self.ca_caravan.get_all_gauge_ids()

        # Filter basins by human influence
        print(f"Found {len(ca_basins)} total CA basins")
        ca_basins, discarded_ca = self.ca_caravan.filter_gauge_ids_by_human_influence(
            ca_basins, ["Low", "Medium"]
        )
        print(f"Loading {len(ca_basins)} CA basins after human influence filtering")
        print(f"Discarded {len(discarded_ca)} CA basins with high human influence")
        self.ca_caravan.load_stations(ca_basins)

        # Prepare data frames
        ts_columns = self.config.FORCING_FEATURES + [self.config.TARGET]
        static_columns = self.config.STATIC_FEATURES

        # Add date column required for data splitting
        ts_columns_with_date = ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]

        # CA data
        self.ts_data = self.ca_caravan.get_time_series()[ts_columns_with_date]
        self.static_data = self.ca_caravan.get_static_attributes()[static_columns]

        self._data_loaded = True
        print(f"Data loaded successfully with {len(ca_basins)} basins.")

    def sample_hyperparameters(self, trial):
        """Sample hyperparameters based on model type and search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameters
        """
        hyperparameters = {}

        # Get hyperparameter search space from config
        search_space = self.config.HYPERPARAMETER_SPACE

        # Sample common hyperparameters
        for param_name, param_config in search_space["common"].items():
            param_type = param_config["type"]

            if param_type == "int":
                step = param_config.get("step", 1)
                hyperparameters[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], step=step
                )
            elif param_type == "float":
                log = param_config.get("log", False)
                hyperparameters[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=log
                )
            elif param_type == "categorical":
                hyperparameters[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        # Sample model-specific hyperparameters
        if "model_specific" in search_space:
            for param_name, param_config in search_space["model_specific"].items():
                param_type = param_config["type"]

                if param_type == "int":
                    hyperparameters[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_type == "float":
                    log = param_config.get("log", False)
                    hyperparameters[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], log=log
                    )
                elif param_type == "categorical":
                    hyperparameters[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

        return hyperparameters

    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation loss achieved
        """
        # Make sure data is loaded
        if not self._data_loaded:
            self.load_data()

        # Sample hyperparameters
        hyperparameters = self.sample_hyperparameters(trial)

        # Update config with trial parameters
        for key, value in hyperparameters.items():
            setattr(self.config, key.upper(), value)

        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config()

        # Create data module
        data_module = HydroDataModule(
            time_series_df=self.ts_data,
            static_df=self.static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            # Use proportional splitting
            use_proportional_split=self.config.USE_PROPORTIONAL_SPLIT,
            train_prop=self.config.TRAIN_PROP,
            val_prop=self.config.VAL_PROP,
            test_prop=self.config.TEST_PROP,
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
        )

        # Prepare data
        data_module.prepare_data()
        data_module.setup()

        # Log dataset sizes
        train_size = (
            len(data_module.train_dataset)
            if hasattr(data_module, "train_dataset")
            else 0
        )
        val_size = (
            len(data_module.val_dataset) if hasattr(data_module, "val_dataset") else 0
        )
        test_size = (
            len(data_module.test_dataset) if hasattr(data_module, "test_dataset") else 0
        )

        print(
            f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}"
        )

        # Store for later logging
        trial.set_user_attr("train_size", train_size)
        trial.set_user_attr("val_size", val_size)
        trial.set_user_attr("test_size", test_size)

        # Create model with trial hyperparameters
        try:
            model = ModelFactory.create_model(self.config)
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            raise optuna.exceptions.TrialPruned()

        # Set up TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.dirs["logs"]),
            name=self.model_type,
            version=f"trial_{trial.number}",
        )

        # Configure callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=str(self.dirs["checkpoints"] / f"trial_{trial.number}"),
                filename="model-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ]

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            logger=tb_logger,
            callbacks=callbacks,
            enable_progress_bar=True,
            # Add checkpointing
            default_root_dir=str(self.dirs["checkpoints"]),
        )

        # Train and get best validation loss
        try:
            trainer.fit(model, data_module)

            # Get the best validation loss
            best_val_loss = trainer.callback_metrics["val_loss"].item()

            # Log additional information
            trial.set_user_attr("best_epoch", trainer.current_epoch)

            return best_val_loss

        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

    def run_optimization(self):
        """Run the hyperparameter optimization study."""
        print(
            f"\n===== Starting hyperparameter optimization for {self.model_type} ====="
        )
        print(f"Running {self.n_trials} trials...")

        # Run optimization
        try:
            self.study.optimize(self.objective, n_trials=self.n_trials)

            # Save results
            self.save_study_results()

            print("\nBest trial:")
            print(f"  Value: {self.study.best_trial.value:.5f}")
            print("  Params: ")
            for key, value in self.study.best_trial.params.items():
                print(f"    {key}: {value}")

            return self.study

        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

            # Still save partial results if available
            if self.study.trials:
                self.save_study_results()

    def save_study_results(self):
        """Save optimization results to CSV and visualizations."""
        # Create results dataframe
        results = []
        for trial in self.study.trials:
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

        # Save to CSV
        results_df.to_csv(
            self.dirs["results"] / f"{self.model_type}_optimization_results.csv",
            index=False,
        )

        # Save best parameters separately
        best_params = self.study.best_trial.params
        best_value = self.study.best_trial.value
        best_results = {"best_value": best_value, **best_params}

        pd.DataFrame([best_results]).to_csv(
            self.dirs["results"] / f"{self.model_type}_best_parameters.csv", index=False
        )

        # Save optimization visualization
        try:
            save_visualizations(
                self.study, self.model_type, self.dirs["visualizations"]
            )
        except Exception as e:
            print(f"Could not create visualizations: {str(e)}")

    def cleanup(self):
        """Clean up resources to free memory."""
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for hydrological models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tide", "tsmixer", "ealstm", "tft"],
        help="Model type to tune",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of optimization trials"
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Name for the Optuna study"
    )
    args = parser.parse_args()

    # Set CUDA precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Load appropriate config based on model type
    if args.model == "tide":
        from experiments.HyperparameterTune.configs.tide_config import TiDETuneConfig

        config = TiDETuneConfig()
    elif args.model == "tsmixer":
        from experiments.HyperparameterTune.configs.tsmixer_config import (
            TSMixerTuneConfig,
        )

        config = TSMixerTuneConfig()
    elif args.model == "ealstm":
        from experiments.HyperparameterTune.configs.ealstm_config import (
            EALSTMTuneConfig,
        )

        config = EALSTMTuneConfig()
    elif args.model == "tft":
        from experiments.HyperparameterTune.configs.tft_config import TFTTuneConfig

        config = TFTTuneConfig()
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # Add current date to study for reporting
    from datetime import datetime

    study_name = (
        args.study_name
        or f"{args.model}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Run optimization
    tuner = HyperparameterTuner(
        config=config,
        model_type=args.model,
        study_name=study_name,
        n_trials=args.n_trials,
    )

    try:
        tuner.load_data()
        study = tuner.run_optimization()

        # Add date for reporting
        study.set_user_attr("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Generate report
        if study and study.trials:
            from experiments.HyperparameterTune.utils import (
                generate_optimization_report,
            )

            generate_optimization_report(
                study, args.model, tuner.dirs["visualizations"]
            )
    finally:
        tuner.cleanup()
