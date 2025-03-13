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
from experiments.GroupBased.configGroupBased import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
import multiprocessing
from datetime import datetime


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/GroupBased/results/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.model_dir = Path(
            f"experiments/GroupBased/saved_models/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.checkpoint_dir = Path(
            f"experiments/GroupBased/checkpoints/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.viz_dir = Path(
            f"experiments/GroupBased/visualizations/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.logs_dir = Path(
            f"experiments/GroupBased/logs/{self.config.EXPERIMENT_NAME}_benchmark"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.viz_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load Central Asia dataset."""
        print("CONFIGURING CA DATASET (BASELINE)")
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
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

    def run_experiment(self):
        """Run the experiment with multiple runs, training and saving the model."""
        for run in range(self.config.NUM_RUNS):
            try:
                print(f"\nStarting run {run}...")
                self.config.set_seed(run)
                self.run_single_experiment(run)
                print(f"Successfully completed run {run}")
                self.cleanup()
            except Exception as e:
                print(f"Error in run {run}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue
        print("Training complete for all runs.")

    def run_single_experiment(self, run: int):
        """Run a single experiment by training only on Central Asia data."""
        preprocessing_configs = self.config.get_preprocessing_config()

        # Add domain prefix to gauge_ids for consistency
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
        trained_model = self.train_model(ca_data_module, run)
        return trained_model

    def create_data_module(self, ts_data, static_data, preprocessing_configs):
        """Create a data module."""
        dm = HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.BENCHMARK_INPUT_LENGTH,
            output_length=self.config.BENCHMARK_OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            # Use proportional splitting
            use_proportional_split=self.config.USE_PROPORTIONAL_SPLIT,
            train_prop=self.config.TRAIN_PROP,
            val_prop=self.config.VAL_PROP,
            test_prop=self.config.TEST_PROP,
            # Legacy parameters (used when not using proportional split)
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        dm.prepare_data()
        dm.setup()
        
        # Log data splitting method
        if self.config.USE_PROPORTIONAL_SPLIT:
            print(f"\nUsing proportional splitting with:")
            print(f"  - Training: {self.config.TRAIN_PROP*100:.2f}% of data")
            print(f"  - Validation: {self.config.VAL_PROP*100:.2f}% of data")
            print(f"  - Testing: {self.config.TEST_PROP*100:.2f}% of data")
            print(f"  - Train dataset size: {len(dm.train_dataset)}")
            print(f"  - Validation dataset size: {len(dm.val_dataset)}")
            print(f"  - Test dataset size: {len(dm.test_dataset)}")
        else:
            print("\nUsing fixed-year splitting with:")
            print(f"  - Min train years: {self.config.CA_CONFIG['MIN_TRAIN_YEARS']}")
            print(f"  - Validation years: {self.config.CA_CONFIG['VAL_YEARS']}")
            print(f"  - Test years: {self.config.CA_CONFIG['TEST_YEARS']}")
            
        return dm

    def train_model(self, data_module, run):
        """Train TSMixer on given data."""
        print("SETTING UP MODEL FOR TRAINING")
        
        # Log information about future forcing features
        print(f"Using {len(self.config.FORCING_FEATURES)} features as future forcing inputs")
        print(f"Future forcing fusion method: {self.config.FUSION_METHOD}")
        
        model = LitTSMixer(self.config.get_benchmark_tsmixer_config())
        trainer = self.create_trainer("train", run)
        trainer.fit(model, data_module)

        # Save full Lightning checkpoint (with global_step and all metadata)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_dir / f"tsmixer_benchmark_run{run}_{timestamp}.ckpt"
        trainer.save_checkpoint(save_path)
        
        print(f"Saved complete model checkpoint to {save_path}")
        return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        if "_" in stage:
            stage_type, group_key = stage.split("_", 1)
            checkpoint_path = self.checkpoint_dir / group_key
            checkpoint_path.mkdir(exist_ok=True)
        else:
            stage_type = stage
            checkpoint_path = self.checkpoint_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger = TensorBoardLogger(
            save_dir="experiments/GroupBased/logs",  # Base directory
            name=f"{self.config.EXPERIMENT_NAME}_benchmark",  # Experiment name
            version=f"run_{run}_{timestamp}",  # Version includes run and timestamp
        )

        return pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=checkpoint_path,
                    filename=f"{stage}-checkpoint-run{run}-{{val_loss:.4f}}",
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
            logger=logger,
        )

    def cleanup(self):
        """Clean up resources after each run."""
        if hasattr(self, "model"):
            del self.model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()
    
    # Log split configuration
    print("\nEXPERIMENT CONFIGURATION:")
    if config.USE_PROPORTIONAL_SPLIT:
        print(f"Using proportional data splitting: {config.TRAIN_PROP:.2f}/{config.VAL_PROP:.2f}/{config.TEST_PROP:.2f}")
    else:
        print("Using fixed-year data splitting")

    # Set CUDA precision if applicable
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run training experiment
    runner = BenchmarkRunner(config)
    runner.load_data()
    runner.run_experiment()
