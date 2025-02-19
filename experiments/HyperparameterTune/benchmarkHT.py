import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


import optuna
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import numpy as np
from experiments.HyperparameterTune.configHT import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer

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
        batch_size = trial.suggest_int("batch_size", 16, 256)
        input_length = trial.suggest_int("input_length", 30, 365)
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Create data module with trial hyperparameters
        preprocessing_configs = self.config.get_preprocessing_config("CA")
        data_module = HydroDataModule(
            time_series_df=self.ca_ts_data,
            static_df=self.ca_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=batch_size,
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

        # Create model with trial hyperparameters
        model = LitTSMixer(
            input_len=input_length,
            output_len=self.config.OUTPUT_LENGTH,
            input_size=len(self.config.FORCING_FEATURES) + 1,
            static_size=len(self.config.STATIC_FEATURES) - 1,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=self.config.PRETRAIN_LR,
        )

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            enable_progress_bar=True,  # Reduce output clutter
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
            study_name="benchmark_optimization",
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
            self.results_dir / "benchmark_optimization_results.csv", index=False
        )

        # Save best parameters separately
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        best_results = {"best_value": best_value, **best_params}

        pd.DataFrame([best_results]).to_csv(
            self.results_dir / "benchmark_best_parameters.csv", index=False
        )


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
    study = tuner.run_optimization(n_trials=30)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.5f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
