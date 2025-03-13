import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.TSMixer import LitTSMixer
from src.data_models.datamodule import HydroDataModule
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from experiments.HyperparameterTune.configHT import ExperimentConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
import pandas as pd
import optuna
import os


class BenchmarkTuner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path("experiments/HyperparameterTune/results/benchmark")
        self.logs_dir = Path("experiments/HyperparameterTune/logs/benchmark")

        for directory in [self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load CA dataset with human influence filtering."""
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
        
        # Add required date column for data splitting
        ts_columns_with_date = ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]

        self.ca_ts_data = self.ca_caravan.get_time_series()[ts_columns_with_date]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        input_length = trial.suggest_int("input_length", 30, 365)
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 2, 15)
        static_embedding_size = trial.suggest_int("static_embedding_size", 5, 20)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        fusion_method = trial.suggest_categorical("fusion_method", ["add", "concat"])

        # Update config with trial parameters
        self.config.INPUT_LENGTH = input_length
        self.config.HIDDEN_SIZE = hidden_size
        self.config.NUM_LAYERS = num_layers
        self.config.STATIC_EMBEDDING_SIZE = static_embedding_size
        self.config.LEARNING_RATE = learning_rate
        self.config.DROPOUT = dropout
        self.config.FUSION_METHOD = fusion_method

        # Create data module with trial hyperparameters
        preprocessing_configs = self.config.get_preprocessing_config()
        data_module = HydroDataModule(
            time_series_df=self.ca_ts_data,
            static_df=self.ca_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=input_length,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=self.config.MAX_WORKERS,
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            # Use proportional splitting
            use_proportional_split=self.config.USE_PROPORTIONAL_SPLIT,
            train_prop=self.config.TRAIN_PROP,
            val_prop=self.config.VAL_PROP,
            test_prop=self.config.TEST_PROP,
            # Legacy parameters (used only if use_proportional_split=False)
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        # Log data split information
        if self.config.USE_PROPORTIONAL_SPLIT:
            print(f"\nUsing proportional splitting with:")
            print(f"  - Training: {self.config.TRAIN_PROP*100:.1f}% of data")
            print(f"  - Validation: {self.config.VAL_PROP*100:.1f}% of data")
            print(f"  - Testing: {self.config.TEST_PROP*100:.1f}% of data")
        else:
            print("\nUsing fixed-year splitting with:")
            print(f"  - Min train years: {self.config.CA_CONFIG['MIN_TRAIN_YEARS']}")
            print(f"  - Validation years: {self.config.CA_CONFIG['VAL_YEARS']}")
            print(f"  - Test years: {self.config.CA_CONFIG['TEST_YEARS']}")

        # Prepare data
        data_module.prepare_data()
        data_module.setup(stage="fit")

        # Create model with trial hyperparameters
        tsmixer_config = self.config.get_tsmixer_config()
        model = LitTSMixer(config=tsmixer_config)

        # Set up TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name="benchmark_optimization",
            version=f"trial_{trial.number}",
        )

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            logger=tb_logger,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            enable_progress_bar=True,
        )

        # Train and get best validation loss
        trainer.fit(model, data_module)

        # Get the best validation loss
        best_val_loss = trainer.callback_metrics["val_loss"].item()

        # Log additional information
        trial.set_user_attr("best_epoch", trainer.current_epoch)
        
        # Log dataset sizes
        train_size = len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') else 0
        val_size = len(data_module.val_dataset) if hasattr(data_module, 'val_dataset') else 0
        test_size = len(data_module.test_dataset) if hasattr(data_module, 'test_dataset') else 0
        trial.set_user_attr("train_size", train_size)
        trial.set_user_attr("val_size", val_size)
        trial.set_user_attr("test_size", test_size)

        return best_val_loss

    def run_optimization(self, n_trials: int = 12):
        """Run the hyperparameter optimization study."""
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name="tsmixer_benchmark_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)

        # Save results
        self.save_study_results(study)

        return study

    def save_study_results(self, study: optuna.Study):
        """Save optimization results to CSV."""
        # Create results dataframe
        results = []
        for trial in study.trials:
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

        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(
            self.results_dir / "benchmark_optimization_results.csv", index=False
        )

        # Save best parameters separately
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        best_results = {"best_value": best_value, **best_params}

        pd.DataFrame([best_results]).to_csv(
            self.results_dir / "benchmark_best_parameters.csv", index=False
        )

        # Save optimization visualization if plotly is available
        try:
            import optuna.visualization as vis
            import matplotlib.pyplot as plt

            # Plot optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image(
                str(self.results_dir / "benchmark_optimization_history.png")
            )

            # Plot parameter importance
            fig2 = vis.plot_param_importances(study)
            fig2.write_image(str(self.results_dir / "benchmark_param_importances.png"))

            # Plot contour plots for top parameters
            fig3 = vis.plot_contour(study)
            fig3.write_image(str(self.results_dir / "benchmark_param_contours.png"))

        except Exception as e:
            print(f"Could not create visualization: {e}")


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()
    
    # Log split configuration
    print("\nEXPERIMENT CONFIGURATION:")
    if config.USE_PROPORTIONAL_SPLIT:
        print(f"Using proportional data splitting: {config.TRAIN_PROP:.1f}/{config.VAL_PROP:.1f}/{config.TEST_PROP:.1f}")
    else:
        print("Using fixed-year data splitting")

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run optimization
    tuner = BenchmarkTuner(config)
    tuner.load_data()
    study = tuner.run_optimization(n_trials=12)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.5f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
