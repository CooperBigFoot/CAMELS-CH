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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiments.GroupBased.configGroupBased import ExperimentConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
from src.models.evaluators import TSForecastEvaluator
import multiprocessing
from datetime import datetime


class GroupBasedFineTuneRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.group_models = {}
        self.group_finetuned_models = {}
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        base_name = f"{self.config.EXPERIMENT_NAME}_finetune"
        self.results_dir = Path(f"experiments/GroupBased/results/{base_name}")
        self.model_dir = Path(f"experiments/GroupBased/saved_models/{base_name}")
        self.checkpoint_dir = Path(f"experiments/GroupBased/checkpoints/{base_name}")
        self.viz_dir = Path(f"experiments/GroupBased/visualizations/{base_name}")
        self.logs_dir = Path(f"experiments/GroupBased/logs/{base_name}")

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.viz_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        for group_key in self.config.GROUP_MAPPINGS.keys():
            (self.model_dir / group_key).mkdir(exist_ok=True)
            (self.results_dir / group_key).mkdir(exist_ok=True)
            (self.viz_dir / group_key).mkdir(exist_ok=True)

    def run_experiment(self):
        """Run the complete experiment with multiple runs."""
        all_results = []

        for run in range(self.config.NUM_RUNS):
            try:
                print(f"\nStarting run {run}...")
                self.config.set_seed(run)
                run_results = self.run_single_experiment(run)
                if run_results is not None:
                    all_results.append(run_results)
                    print(f"Successfully completed run {run}")
                else:
                    print(f"Run {run} failed to produce results")

                # Clean up after each run
                self.cleanup()

            except Exception as e:
                print(f"Error in run {run}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate results across runs
        self.save_aggregated_results(all_results)

    def run_single_experiment(self, run: int):
        """Run a single experiment by training a model for each group and fine-tuning."""
        training_data = self.config.extract_source_basins_for_training()
        group_results = {}

        for group_key in training_data.keys():
            print(f"\n=== PROCESSING GROUP: {group_key} ===")

            # Load data for this group
            group_data = self.config.load_data_for_group(group_key)

            # Prepare data modules for this group
            data_module = self.prepare_data_module(group_key, group_data)

            # Train initial model
            initial_model = self.train_model(data_module, group_key, run, stage="initial")
            self.group_models[group_key] = initial_model

            # Fine-tune model on CA data
            finetuned_model = self.fine_tune_model(data_module, group_key, run)
            self.group_finetuned_models[group_key] = finetuned_model

            # Evaluate fine-tuned model
            results = self.evaluate_model(finetuned_model, data_module, group_key, run, stage="finetuned")
            group_results[group_key] = results

        # Evaluate cross-group performance
        self.evaluate_cross_group_performance(run)

        # Combine results from all groups
        combined_results = self.combine_group_results(group_results)

        return combined_results

    def prepare_data_module(self, group_key, group_data):
        # Same as in previous implementation
        preprocessing_configs = self.config.get_preprocessing_config()
        
        ts_columns = self.config.FORCING_FEATURES + [
            self.config.TARGET,
            "date",
            self.config.GROUP_IDENTIFIER,
        ]
        static_columns = self.config.STATIC_FEATURES

        ca_ts_data = group_data["ca_ts_data"][ts_columns]
        ca_static_data = group_data["ca_static_data"][static_columns]

        source_ts_data = []
        source_static_data = []

        for i, ts_df in enumerate(group_data["source_ts_data"]):
            static_df = group_data["source_static_data"][i]

            filtered_ts = ts_df[ts_columns]
            filtered_static = static_df[static_columns]

            source_ts_data.append(filtered_ts)
            source_static_data.append(filtered_static)

        all_ts_data = [ca_ts_data] + source_ts_data
        all_static_data = [ca_static_data] + source_static_data

        data_module = HydroDataModule(
            time_series_df=all_ts_data,
            static_df=all_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
            domain_id=f"group_{group_key}",
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")

        return data_module

    def train_model(self, data_module, group_key, run, stage="initial"):
        """Train a model for a specific group."""
        config = self.config.get_tsmixer_config()
        
        # Use standard config for initial training
        model = LitTSMixer(config)
        trainer = self.create_trainer(f"train_{group_key}_{stage}", run)
        trainer.fit(model, data_module)

        # Save best model
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint:
            best_model = LitTSMixer.load_from_checkpoint(best_checkpoint, config=config)
            
            val_loss = trainer.checkpoint_callback.best_model_score.item()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_filename = f"tsmixer_{group_key}_{stage}_run{run}_valloss{val_loss:.4f}_{timestamp}.pt"
            save_path = self.model_dir / group_key / model_filename

            torch.save(
                {
                    "state_dict": best_model.state_dict(),
                    "config": config.to_dict(),
                    "val_loss": val_loss,
                    "group_key": group_key,
                    "run": run,
                    "stage": stage,
                    "timestamp": timestamp,
                },
                save_path,
            )
            return best_model
        
        return model

    def fine_tune_model(self, data_module, group_key, run):
        """Fine-tune model only on CA data with reduced learning rate."""
        # Load initial model from previous training stage
        initial_model = self.group_models[group_key]

        # Create a new model config with reduced learning rate
        finetune_config = self.config.get_tsmixer_config()
        finetune_config.learning_rate /= 10  # Reduce learning rate by 10x

        # Create a new model and load initial weights
        finetuned_model = LitTSMixer(finetune_config)
        
        # Freeze backbone
        for param in finetuned_model.model.backbone.parameters():
            param.requires_grad = False

        # Load initial model weights
        initial_state_dict = initial_model.state_dict()
        finetuned_model.load_state_dict(initial_state_dict, strict=False)

        # Train only on CA data (first dataset in the module)
        ca_ts_data = data_module.train_dataset.df_sorted[
            data_module.train_dataset.df_sorted[self.config.GROUP_IDENTIFIER].str.startswith('CA_')
        ]
        ca_data_module = HydroDataModule(
            time_series_df=ca_ts_data,
            static_df=data_module.processed_static,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=data_module.preprocessing_config,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )
        ca_data_module.prepare_data()
        ca_data_module.setup(stage="fit")

        # Train fine-tuned model
        trainer = self.create_trainer(f"finetune_{group_key}", run)
        trainer.fit(finetuned_model, ca_data_module)

        # Save fine-tuned model
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint:
            best_model = LitTSMixer.load_from_checkpoint(best_checkpoint, config=finetune_config)
            
            val_loss = trainer.checkpoint_callback.best_model_score.item()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_filename = f"tsmixer_{group_key}_finetuned_run{run}_valloss{val_loss:.4f}_{timestamp}.pt"
            save_path = self.model_dir / group_key / model_filename

            torch.save(
                {
                    "state_dict": best_model.state_dict(),
                    "config": finetune_config.to_dict(),
                    "val_loss": val_loss,
                    "group_key": group_key,
                    "run": run,
                    "timestamp": timestamp,
                },
                save_path,
            )
            return best_model
        
        return finetuned_model

    # Other methods remain the same as in challengerGroupBased.py

if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = GroupBasedFineTuneRunner(config)
    runner.run_experiment()