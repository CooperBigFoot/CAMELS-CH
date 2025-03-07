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

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger = pl.loggers.CSVLogger(
            save_dir=self.logs_dir,
            name=f"{stage}_run{run}_{timestamp}",
            flush_logs_every_n_steps=100,
        )

        return pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=self.checkpoint_dir,
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

    def save_aggregated_results(self, all_results):
        """Save aggregated results across all runs."""
        if not all_results:
            print("Warning: No results to aggregate")
            return

        try:
            overall_metrics_df = pd.concat(
                [
                    pd.DataFrame(run["overall_metrics"]).assign(run=i)
                    for i, run in enumerate(all_results)
                    if run is not None and "overall_metrics" in run
                ]
            )

            if overall_metrics_df.empty:
                print("Warning: No valid metrics to aggregate")
                return

            summary_stats = overall_metrics_df.groupby(level=0).agg(
                ["mean", "std", "min", "max"]
            )
            summary_stats.to_csv(self.results_dir / "aggregate_metrics.csv")

            print(f"Successfully saved aggregate metrics for {len(all_results)} runs")

        except Exception as e:
            print(f"Error while saving aggregated results: {str(e)}")

    def evaluate_cross_group_performance(self, run):
        """Evaluate models on all CA groups."""
        print("\n=== CROSS-GROUP EVALUATION ===")
        cross_group_results = {}
        training_data = self.config.extract_source_basins_for_training()

        for model_key, model in self.group_finetuned_models.items():
            cross_group_results[model_key] = {}

            for target_key in training_data.keys():
                if model_key == target_key:
                    continue

                group_data = self.config.load_data_for_group(target_key)
                ca_ts_data = group_data["ca_ts_data"]

                test_dm = HydroDataModule(
                    time_series_df=ca_ts_data,
                    static_df=group_data["ca_static_data"],
                    group_identifier=self.config.GROUP_IDENTIFIER,
                    preprocessing_config=self.config.get_preprocessing_config(),
                    batch_size=self.config.BATCH_SIZE,
                    input_length=self.config.INPUT_LENGTH,
                    output_length=self.config.OUTPUT_LENGTH,
                    features=self.config.FORCING_FEATURES + [self.config.TARGET],
                    static_features=self.config.STATIC_FEATURES,
                    target=self.config.TARGET,
                )
                test_dm.prepare_data()
                test_dm.setup(stage="test")

                trainer = self.create_trainer(f"cross_{model_key}_on_{target_key}", run)
                trainer.test(model, test_dm)

                evaluator = TSForecastEvaluator(
                    test_dm, horizons=list(range(1, self.config.OUTPUT_LENGTH + 1))
                )
                _, overall_metrics, _ = evaluator.evaluate(model.test_results)

                cross_group_results[model_key][target_key] = overall_metrics

        if cross_group_results:
            cross_df_rows = []
            for model_group, target_groups in cross_group_results.items():
                for target_group, metrics in target_groups.items():
                    for horizon, horizon_metrics in metrics.items():
                        row = {
                            "model_group": model_group,
                            "target_group": target_group,
                            "horizon": horizon,
                            **horizon_metrics
                        }
                        cross_df_rows.append(row)

            cross_df = pd.DataFrame(cross_df_rows)
            cross_df.to_csv(self.results_dir / f"cross_group_evaluation_run{run}.csv", index=False)

    def combine_group_results(self, group_results):
        """Combine results from all groups."""
        combined = {
            "overall_metrics": {},
            "basin_metrics": {},
        }

        # Pool basin metrics
        for group_key, results in group_results.items():
            for basin_id, basin_data in results["basin_metrics"].items():
                combined["basin_metrics"][f"{group_key}_{basin_id}"] = basin_data

        # Calculate combined overall metrics
        horizon_metrics = {}
        basin_counts = {group_key: len(results["basin_metrics"]) for group_key, results in group_results.items()}
        total_basins = sum(basin_counts.values())

        for horizon in range(1, self.config.OUTPUT_LENGTH + 1):
            horizon_metrics[horizon] = {}
            first_group = next(iter(group_results.values()))
            metric_names = first_group["overall_metrics"][horizon].keys()

            for metric in metric_names:
                weighted_sum = 0
                for group_key, results in group_results.items():
                    group_weight = basin_counts[group_key] / total_basins
                    weighted_sum += results["overall_metrics"][horizon][metric] * group_weight

                horizon_metrics[horizon][metric] = weighted_sum

        combined["overall_metrics"] = horizon_metrics

        return combined

    def evaluate_model(self, model, data_module, group_key, run, stage="finetuned"):
        """Evaluate the model and save results."""
        trainer = self.create_trainer(f"evaluate_{group_key}_{stage}", run)
        trainer.test(model, data_module)

        evaluator = TSForecastEvaluator(
            data_module, horizons=list(range(1, self.config.OUTPUT_LENGTH + 1))
        )

        results_df, overall_metrics, basin_metrics = evaluator.evaluate(model.test_results)

        # Save results
        results_path = self.results_dir / group_key / f"{stage}_detailed_results_{run}.csv"
        results_df.to_csv(results_path, index=True)

        overall_summary = evaluator.summarize_metrics(overall_metrics)
        overall_path = self.results_dir / group_key / f"{stage}_overall_metrics_{run}.csv"
        overall_summary.to_csv(overall_path, index=True)

        basin_summary = evaluator.summarize_metrics(basin_metrics, per_basin=True)
        basin_path = self.results_dir / group_key / f"{stage}_basin_metrics_{run}.csv"
        basin_summary.to_csv(basin_path, index=True)

        return {
            "overall_metrics": overall_metrics,
            "basin_metrics": basin_metrics,
            "results_df": results_df,
        }

    def cleanup(self):
        """Clean up resources after each run."""
        import gc
        
        self.group_models.clear()
        self.group_finetuned_models.clear()
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = GroupBasedFineTuneRunner(config)
    runner.run_experiment()