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
from experiments.Merged.configMerged import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
import multiprocessing


class ChallengerRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/Merged/results/{self.config.EXPERIMENT_NAME}_challenger"
        )
        self.model_dir = Path(
            f"experiments/Merged/saved_models/{self.config.EXPERIMENT_NAME}_challenger"
        )
        self.checkpoint_dir = Path(
            f"experiments/Merged/checkpoints/{self.config.EXPERIMENT_NAME}_challenger"
        )
        self.logs_dir = Path(
            f"experiments/Merged/logs/{self.config.EXPERIMENT_NAME}_challenger"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.logs_dir,
        ]:
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

        # CA data
        self.ca_ts_data = self.ca_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

        # CH data
        self.ch_ts_data = self.ch_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ch_static_data = self.ch_caravan.get_static_attributes()[static_columns]

        # CL data
        self.cl_ts_data = self.cl_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.cl_static_data = self.cl_caravan.get_static_attributes()[static_columns]

        # USA data
        self.usa_ts_data = self.usa_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.usa_static_data = self.usa_caravan.get_static_attributes()[static_columns]

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
        """Run a single experiment by training on merged CA+CH+CL+USA data."""
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

        # Create merged data module using CA training parameters
        print("\n=== CREATING MERGED DATASET (CA + CH + CL + USA) ===")
        merged_data_module = self.create_data_module(
            merged_ts_data,
            merged_static_data,
            preprocessing_configs,
        )

        # Train model on merged data
        print("\n=== TRAINING CHALLENGER MODEL ON MERGED DATA ===")
        self.train_model(merged_data_module, run)

    def create_data_module(
        self,
        ts_data,
        static_data,
        preprocessing_configs,
    ):
        """Create a data module with merged data sources."""
        dm = HydroDataModule(
            time_series_df=ts_data,  # Can be a list of DataFrames
            static_df=static_data,  # Can be a list of DataFrames
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.CHALLENGER_INPUT_LENGTH,  # Use challenger input length
            output_length=self.config.CHALLENGER_OUTPUT_LENGTH,  # Use challenger output length
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

        # Explicitly prepare and set up the data module
        dm.prepare_data()
        dm.setup()
        
        # Log data splitting method
        if self.config.USE_PROPORTIONAL_SPLIT:
            print("\nUsing proportional splitting with:")
            print(f"  - Training: {self.config.TRAIN_PROP*100:.1f}% of data")
            print(f"  - Validation: {self.config.VAL_PROP*100:.1f}% of data")
            print(f"  - Testing: {self.config.TEST_PROP*100:.1f}% of data")
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
        """Train TSMixer on the given data."""
        print("SETTING UP MODEL FOR TRAINING")
        
        # Log information about future forcing features
        print(f"Using {len(self.config.FORCING_FEATURES)} features as future forcing inputs")
        print(f"Future forcing fusion method: {self.config.FUSION_METHOD}")

        # Create a TSMixer model with challenger hyperparameters
        model = LitTSMixer(self.config.get_challenger_tsmixer_config())

        # Train the model
        trainer = self.create_trainer("train", run)
        trainer.fit(model, data_module)

        # Save full Lightning checkpoint (with global_step and all metadata)
        save_path = self.model_dir / f"tsmixer_challenger_{run}.ckpt"
        trainer.save_checkpoint(save_path)

        return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        # Create a TensorBoardLogger instance
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name=f"{self.config.EXPERIMENT_NAME}_challenger",
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
    
    # Log split configuration
    print("\nEXPERIMENT CONFIGURATION:")
    if config.USE_PROPORTIONAL_SPLIT:
        print(f"Using proportional data splitting: {config.TRAIN_PROP:.1f}/{config.VAL_PROP:.1f}/{config.TEST_PROP:.1f}")
    else:
        print("Using fixed-year data splitting")

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = ChallengerRunner(config)
    runner.load_data()
    runner.run_experiment()