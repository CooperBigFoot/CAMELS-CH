import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.TSMixer import LitTSMixer, TSMixerConfig
from src.data_models.datamodule import HydroDataModule
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from experiments.HyperparameterTune.configHT import ExperimentConfig
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
import torch
import pandas as pd
import optuna


class BenchmarkTuner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path("experiments/HyperparameterTune/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load CA dataset once for all trials."""
        print("CONFIGURING CA DATASET")
        ca_config = CaravanifyConfig(
            attributes_dir=self.config.CA_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.CA_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.CA_CONFIG["GAUGE_ID_PREFIX"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        self.ca_caravan = Caravanify(ca_config)
        ca_basins = self.ca_caravan.get_all_gauge_ids()
        print(f"Loading {len(ca_basins)} CA basins")
        self.ca_caravan.load_stations(ca_basins)

        # Prepare data frames
        ts_columns = self.config.FORCING_FEATURES + [self.config.TARGET]
        static_columns = self.config.STATIC_FEATURES

        self.ca_ts_data = self.ca_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[
            static_columns]

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        input_length = trial.suggest_int("input_length", 30, 120)
        hidden_size = trial.suggest_int("hidden_size", 32, 128
                                        )
        num_layers = trial.suggest_int("num_layers", 2, 10)
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Create data module with trial hyperparameters
        preprocessing_configs = self.config.get_preprocessing_config("CA")
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
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        # Create TSMixerConfig
        tsmixer_config = TSMixerConfig(
            input_len=input_length,
            input_size=len(self.config.FORCING_FEATURES) +
            1,  # Add 1 for target
            output_len=self.config.OUTPUT_LENGTH,
            static_size=len(self.config.STATIC_FEATURES) -
            1,  # Subtract 1 for gauge_id
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            group_identifier=self.config.GROUP_IDENTIFIER,
            lr_scheduler_patience=self.config.LR_SCHEDULER_PATIENCE,
            lr_scheduler_factor=self.config.LR_SCHEDULER_FACTOR,
        )

        # Create model with TSMixerConfig
        model = LitTSMixer(config=tsmixer_config)

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
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

        return best_val_loss

    def run_optimization(self, n_trials: int = 50):
        """Run the hyperparameter optimization study."""
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name="tsmixer_optimization",
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
                **trial.params,  # Includes all hyperparameters
            }
            results.append(trial_data)

        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(
            self.results_dir / "tsmixer_optimization_results.csv", index=False
        )

        # Save best parameters separately
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        best_results = {"best_value": best_value, **best_params}

        pd.DataFrame([best_results]).to_csv(
            self.results_dir / "tsmixer_best_parameters.csv", index=False
        )

        # Save optimization visualization
        try:
            import matplotlib.pyplot as plt

            # Plot optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(str(self.results_dir / "optimization_history.png"))

            # Plot parameter importance
            param_importance = optuna.visualization.plot_param_importances(
                study)
            param_importance.write_image(
                str(self.results_dir / "param_importances.png"))

            # Plot contour plots for top parameters
            contour = optuna.visualization.plot_contour(study)
            contour.write_image(str(self.results_dir / "param_contours.png"))

        except (ImportError, AttributeError) as e:
            print(f"Could not create visualization: {e}")


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run optimization
    tuner = BenchmarkTuner(config)
    tuner.load_data()
    # Adjust number of trials as needed
    study = tuner.run_optimization(n_trials=50)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.5f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
