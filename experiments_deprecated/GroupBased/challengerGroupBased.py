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
from experiments.GroupBased.configGroupBased import ExperimentConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
import multiprocessing


class GroupBasedRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.group_models = {}
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/GroupBased/results/{self.config.EXPERIMENT_NAME}"
        )
        self.model_dir = Path(
            f"experiments/GroupBased/saved_models/{self.config.EXPERIMENT_NAME}"
        )
        self.checkpoint_dir = Path(
            f"experiments/GroupBased/checkpoints/{self.config.EXPERIMENT_NAME}"
        )
        self.viz_dir = Path(
            f"experiments/GroupBased/visualizations/{self.config.EXPERIMENT_NAME}"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.viz_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each group
        for group_key in self.config.GROUP_MAPPINGS.keys():
            group_dir = self.model_dir / group_key
            group_dir.mkdir(exist_ok=True)

            group_results_dir = self.results_dir / group_key
            group_results_dir.mkdir(exist_ok=True)

            group_viz_dir = self.viz_dir / group_key
            group_viz_dir.mkdir(exist_ok=True)

    def run_experiment(self):
        """Run the complete experiment with multiple runs, training and saving models."""
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
        """Run a single experiment by training a model for each group."""
        # Extract training data mapping
        training_data = self.config.extract_source_basins_for_training()

        # Train a model for each group
        for group_key in training_data.keys():
            print(f"\n=== PROCESSING GROUP: {group_key} ===")

            # Load data for this group
            group_data = self.config.load_data_for_group(group_key)

            # Prepare data module for this group
            data_module = self.prepare_data_module(group_key, group_data)

            # Train model for this group
            model = self.train_model(data_module, group_key, run)

            # Store model for potential future use
            self.group_models[group_key] = model

    def prepare_data_module(self, group_key, group_data):
        """Create a data module for a specific group."""
        preprocessing_configs = self.config.get_preprocessing_config()

        # Extract required columns
        ts_columns = self.config.FORCING_FEATURES + [
            self.config.TARGET,
            "date",
            self.config.GROUP_IDENTIFIER,
        ]
        static_columns = self.config.STATIC_FEATURES

        # Filter CA data
        ca_ts_data = group_data["ca_ts_data"][ts_columns]
        ca_static_data = group_data["ca_static_data"][static_columns]

        # Filter source data
        source_ts_data = []
        source_static_data = []

        for i, ts_df in enumerate(group_data["source_ts_data"]):
            static_df = group_data["source_static_data"][i]
            filtered_ts = ts_df[ts_columns]
            filtered_static = static_df[static_columns]
            source_ts_data.append(filtered_ts)
            source_static_data.append(filtered_static)

        # Combine source and target data
        all_ts_data = [ca_ts_data] + source_ts_data
        all_static_data = [ca_static_data] + source_static_data

        # Create data module
        data_module = HydroDataModule(
            time_series_df=all_ts_data,
            static_df=all_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.CHALLENGER_INPUT_LENGTH,
            output_length=self.config.CHALLENGER_OUTPUT_LENGTH,
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
            domain_id=f"group_{group_key}",
        )

        data_module.prepare_data()
        data_module.setup()

        # Log data splitting method
        if self.config.USE_PROPORTIONAL_SPLIT:
            print(f"\nUsing proportional splitting for group {group_key} with:")
            print(f"  - Training: {self.config.TRAIN_PROP * 100:.1f}% of data")
            print(f"  - Validation: {self.config.VAL_PROP * 100:.1f}% of data")
            print(f"  - Testing: {self.config.TEST_PROP * 100:.1f}% of data")
            print(f"  - Train dataset size: {len(data_module.train_dataset)}")
            print(f"  - Validation dataset size: {len(data_module.val_dataset)}")
            print(f"  - Test dataset size: {len(data_module.test_dataset)}")
        else:
            print(f"\nUsing fixed-year splitting for group {group_key} with:")
            print(f"  - Min train years: {self.config.CA_CONFIG['MIN_TRAIN_YEARS']}")
            print(f"  - Validation years: {self.config.CA_CONFIG['VAL_YEARS']}")
            print(f"  - Test years: {self.config.CA_CONFIG['TEST_YEARS']}")

        return data_module

    def train_model(self, data_module, group_key, run):
        """Train a model for a specific group."""
        print(f"SETTING UP MODEL FOR GROUP {group_key} TRAINING")

        # Log information about future forcing features
        print(
            f"Using {len(self.config.FORCING_FEATURES)} features as future forcing inputs"
        )
        print(f"Future forcing fusion method: {self.config.FUSION_METHOD}")

        # Create TSMixer model
        model = LitTSMixer(self.config.get_challenger_tsmixer_config())

        # Train the model
        trainer = self.create_trainer(f"train_{group_key}", run)
        trainer.fit(model, data_module)

        # Save full Lightning checkpoint (with global_step and all metadata)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            self.model_dir
            / group_key
            / f"tsmixer_{group_key}_run{run}_{timestamp}.ckpt"
        )
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
            group_key = "all"  # Default if no group is specified

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger = TensorBoardLogger(
            save_dir="experiments/GroupBased/logs",  # Base directory
            name=f"{self.config.EXPERIMENT_NAME}",  # Experiment name
            version=f"group_{group_key}_run{run}_{timestamp}",  # Version includes group, run, timestamp
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
        self.group_models = {}
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
        print(
            f"Using proportional data splitting: {config.TRAIN_PROP:.1f}/{config.VAL_PROP:.1f}/{config.TEST_PROP:.1f}"
        )
    else:
        print("Using fixed-year data splitting")

    # Set CUDA precision if applicable
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run training experiment
    runner = GroupBasedRunner(config)
    runner.run_experiment()
