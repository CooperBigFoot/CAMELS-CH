import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.TSMixer import LitTSMixer
from src.data_models.datamodule import HydroDataModule, HydroTransferDataModule
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from experiments.HyperparameterTune.configHT import ExperimentConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
import pandas as pd
import optuna
import multiprocessing


class ChallengerTuner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path("experiments/HyperparameterTune/results/challenger")
        self.logs_dir = Path("experiments/HyperparameterTune/logs/challenger")

        for directory in [self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load and prepare datasets from all regions with human influence filtering."""
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

        # CH Dataset
        print("CONFIGURING CH DATASET")
        ch_config = CaravanifyConfig(
            attributes_dir=self.config.CH_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.CH_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.CH_CONFIG["GAUGE_ID_PREFIX"],
            human_influence_path=self.config.CH_CONFIG["HUMAN_INFLUENCE_PATH"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        self.ch_caravan = Caravanify(ch_config)
        ch_basins = self.ch_caravan.get_all_gauge_ids()
        # Filter basins by human influence
        print(f"Found {len(ch_basins)} total CH basins")
        ch_basins, discarded_ch = self.ch_caravan.filter_gauge_ids_by_human_influence(
            ch_basins, ["Low", "Medium"]
        )
        print(f"Loading {len(ch_basins)} CH basins after human influence filtering")
        print(f"Discarded {len(discarded_ch)} CH basins with high human influence")
        self.ch_caravan.load_stations(ch_basins)

        # CL Dataset
        print("CONFIGURING CL DATASET")
        cl_config = CaravanifyConfig(
            attributes_dir=self.config.CL_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.CL_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.CL_CONFIG["GAUGE_ID_PREFIX"],
            human_influence_path=self.config.CL_CONFIG["HUMAN_INFLUENCE_PATH"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        self.cl_caravan = Caravanify(cl_config)
        cl_basins = self.cl_caravan.get_all_gauge_ids()
        # Filter basins by human influence
        print(f"Found {len(cl_basins)} total CL basins")
        cl_basins, discarded_cl = self.cl_caravan.filter_gauge_ids_by_human_influence(
            cl_basins, ["Low", "Medium"]
        )
        print(f"Loading {len(cl_basins)} CL basins after human influence filtering")
        print(f"Discarded {len(discarded_cl)} CL basins with high human influence")
        self.cl_caravan.load_stations(cl_basins)

        # USA Dataset
        print("CONFIGURING USA DATASET")
        usa_config = CaravanifyConfig(
            attributes_dir=self.config.USA_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.USA_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.USA_CONFIG["GAUGE_ID_PREFIX"],
            human_influence_path=self.config.USA_CONFIG["HUMAN_INFLUENCE_PATH"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )
        self.usa_caravan = Caravanify(usa_config)
        usa_basins = self.usa_caravan.get_all_gauge_ids()
        # Filter basins by human influence
        print(f"Found {len(usa_basins)} total USA basins")
        usa_basins, discarded_usa = (
            self.usa_caravan.filter_gauge_ids_by_human_influence(
                usa_basins, ["Low", "Medium"]
            )
        )
        print(f"Loading {len(usa_basins)} USA basins after human influence filtering")
        print(f"Discarded {len(discarded_usa)} USA basins with high human influence")
        self.usa_caravan.load_stations(usa_basins)

        # Prepare data frames
        ts_columns = self.config.FORCING_FEATURES + [self.config.TARGET]
        static_columns = self.config.STATIC_FEATURES

        # Add date column required for data splitting
        ts_columns_with_date = ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        
        # CA data
        self.ca_ts_data = self.ca_caravan.get_time_series()[ts_columns_with_date]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

        # CH data
        self.ch_ts_data = self.ch_caravan.get_time_series()[ts_columns_with_date]
        self.ch_static_data = self.ch_caravan.get_static_attributes()[static_columns]

        # CL data
        self.cl_ts_data = self.cl_caravan.get_time_series()[ts_columns_with_date]
        self.cl_static_data = self.cl_caravan.get_static_attributes()[static_columns]

        # USA data
        self.usa_ts_data = self.usa_caravan.get_time_series()[ts_columns_with_date]
        self.usa_static_data = self.usa_caravan.get_static_attributes()[static_columns]

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        input_length = trial.suggest_int("input_length", 30, 365)
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 2, 15)
        static_embedding_size = trial.suggest_int("static_embedding_size", 5, 20)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
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

        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config()

        # Add domain identifier to data frames to avoid duplicate gauge IDs
        self.ca_ts_data["domain"] = "CA"
        self.ch_ts_data["domain"] = "CH"
        self.cl_ts_data["domain"] = "CL"
        self.usa_ts_data["domain"] = "USA"
        self.ca_static_data["domain"] = "CA"
        self.ch_static_data["domain"] = "CH"
        self.cl_static_data["domain"] = "CL"
        self.usa_static_data["domain"] = "USA"

        # Create merged DataFrames using lists of DataFrames
        merged_ts_data = [
            self.ca_ts_data,
            self.ch_ts_data,
            self.cl_ts_data,
            self.usa_ts_data,
        ]
        merged_static_data = [
            self.ca_static_data,
            self.ch_static_data,
            self.cl_static_data,
            self.usa_static_data,
        ]

        # Log splitting configuration
        if self.config.USE_PROPORTIONAL_SPLIT:
            print(f"\nUsing proportional splitting with:")
            print(f"  - Training: {self.config.TRAIN_PROP*100:.2f}% of data")
            print(f"  - Validation: {self.config.VAL_PROP*100:.2f}% of data")
            print(f"  - Testing: {self.config.TEST_PROP*100:.2f}% of data")
        else:
            print("\nUsing fixed-year splitting with:")
            print(f"  - Min train years: {self.config.CA_CONFIG['MIN_TRAIN_YEARS']}")
            print(f"  - Validation years: {self.config.CA_CONFIG['VAL_YEARS']}")
            print(f"  - Test years: {self.config.CA_CONFIG['TEST_YEARS']}")

        # Create merged data module using CA training parameters
        print(f"\n=== CREATING MERGED DATASET FOR TRIAL {trial.number} ===")
        merged_data_module = HydroDataModule(
            time_series_df=merged_ts_data,  # Can be a list of DataFrames
            static_df=merged_static_data,  # Can be a list of DataFrames
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=input_length,
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
            # Legacy parameters (used only if use_proportional_split=False)
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        # Prepare data
        merged_data_module.prepare_data()
        merged_data_module.setup()
        
        # Log dataset sizes
        train_size = len(merged_data_module.train_dataset) if hasattr(merged_data_module, 'train_dataset') else 0
        val_size = len(merged_data_module.val_dataset) if hasattr(merged_data_module, 'val_dataset') else 0
        test_size = len(merged_data_module.test_dataset) if hasattr(merged_data_module, 'test_dataset') else 0
        print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        
        # Store for later logging
        trial.set_user_attr("train_size", train_size)
        trial.set_user_attr("val_size", val_size)
        trial.set_user_attr("test_size", test_size)

        # Create model with trial hyperparameters
        tsmixer_config = self.config.get_tsmixer_config()
        model = LitTSMixer(config=tsmixer_config)
        
        # Log future forcing info
        print(f"Trial {trial.number} using fusion method: {fusion_method}")
        print(f"Future forcing size: {len(self.config.FORCING_FEATURES)}")

        # Set up TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name="challenger_optimization",
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
        trainer.fit(model, merged_data_module)

        # Get the best validation loss
        best_val_loss = trainer.callback_metrics["val_loss"].item()

        # Log additional information
        trial.set_user_attr("best_epoch", trainer.current_epoch)

        return best_val_loss

    def run_optimization(self, n_trials: int = 12):
        """Run the hyperparameter optimization study."""
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name="tsmixer_challenger_optimization",
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
            self.results_dir / "challenger_optimization_results.csv", index=False
        )

        # Save best parameters separately
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        best_results = {"best_value": best_value, **best_params}

        pd.DataFrame([best_results]).to_csv(
            self.results_dir / "challenger_best_parameters.csv", index=False
        )

        # Save optimization visualization if plotly is available
        try:
            import optuna.visualization as vis
            import matplotlib.pyplot as plt

            # Plot optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image(
                str(self.results_dir / "challenger_optimization_history.png")
            )

            # Plot parameter importance
            fig2 = vis.plot_param_importances(study)
            fig2.write_image(str(self.results_dir / "challenger_param_importances.png"))

            # Plot contour plots for top parameters
            fig3 = vis.plot_contour(study)
            fig3.write_image(str(self.results_dir / "challenger_param_contours.png"))

        except Exception as e:
            print(f"Could not create visualization: {e}")

    def cleanup(self):
        """Clean up resources to free memory."""
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()
    
    # Log split configuration
    print("\nEXPERIMENT CONFIGURATION:")
    if config.USE_PROPORTIONAL_SPLIT:
        print(f"Using proportional data splitting: {config.TRAIN_PROP:.2f}/{config.VAL_PROP:.2f}/{config.TEST_PROP:.2f}")
    else:
        print("Using fixed-year data splitting")

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run optimization
    tuner = ChallengerTuner(config)
    tuner.load_data()
    study = tuner.run_optimization(n_trials=12)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value:.5f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
