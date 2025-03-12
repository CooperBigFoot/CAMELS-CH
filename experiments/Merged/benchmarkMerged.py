import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import matplotlib.pyplot as plt
from experiments.Merged.configMerged import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
import multiprocessing


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/Merged/results/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.model_dir = Path(
            f"experiments/Merged/saved_models/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.checkpoint_dir = Path(
            f"experiments/Merged/checkpoints/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.logs_dir = Path(
            f"experiments/Merged/logs/{self.config.EXPERIMENT_NAME}_benchmark"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load Central Asia dataset with human influence filtering."""
        # CA Dataset
        print("CONFIGURING CA DATASET (BASELINE)")
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
        ca_basins = self.ca_caravan.get_all_gauge_ids()[:3]
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

        self.ca_ts_data = self.ca_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

    def run_experiment(self):
        """Run the complete experiment with multiple runs."""
        for run in range(self.config.NUM_RUNS):
            try:
                print(f"\nStarting run {run}...")
                self.config.set_seed(run)
                self.run_single_experiment(run)
                print(f"Successfully completed run {run}")

                # Clean up after each run
                self.cleanup()

            except Exception as e:
                print(f"Error in run {run}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    def run_single_experiment(self, run: int):
        """Run a single experiment by training only on Central Asia data."""
        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config()

        # Add domain prefix to gauge_ids for consistency with challenger
        self.ca_ts_data["domain"] = "CA"
        self.ca_static_data["domain"] = "CA"

        # Create data module for CA only
        ca_data_module = self.create_data_module(
            self.ca_ts_data,
            self.ca_static_data,
            preprocessing_configs,
        )

        # Train model on CA data
        print("\n=== TRAINING BENCHMARK MODEL ON CA DATA ONLY ===")
        self.train_model(ca_data_module, run)

    def create_data_module(
        self,
        ts_data,
        static_data,
        preprocessing_configs,
    ):
        """Create a data module."""
        dm = HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.BENCHMARK_INPUT_LENGTH,  # Use benchmark input length
            output_length=self.config.BENCHMARK_OUTPUT_LENGTH,  # Use benchmark output length
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        # Explicitly prepare and set up the data module
        dm.prepare_data()
        dm.setup(stage="fit")

        return dm

    def train_model(self, data_module, run):
        """Train TSMixer on given data."""
        print("SETTING UP MODEL FOR TRAINING")

        # Create a TSMixer model with benchmark hyperparameters
        model = LitTSMixer(self.config.get_benchmark_tsmixer_config())

        # Train the model
        trainer = self.create_trainer("train", run)
        trainer.fit(model, data_module)

        # Save full Lightning checkpoint (with global_step and all metadata)
        save_path = self.model_dir / f"tsmixer_benchmark_{run}.ckpt"
        trainer.save_checkpoint(save_path)

        return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        # Create a TensorBoardLogger instance
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name=f"{self.config.EXPERIMENT_NAME}_benchmark",
            version=run,
        )

        return pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            logger=tb_logger,  # Add the logger
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=self.checkpoint_dir / stage,
                    filename=f"{stage}-checkpoint-run{run}",
                    save_top_k=1,
                    mode="min",
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.LR_SCHEDULER_PATIENCE,
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )

    def cleanup(self):
        """Clean up resources after each run."""
        # Delete explicit references to large objects
        if hasattr(self, "model"):
            del self.model

        # Force Python garbage collection
        import gc

        gc.collect()

        # Only clear CUDA cache if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = BenchmarkRunner(config)
    runner.load_data()
    runner.run_experiment()