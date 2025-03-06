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
        """Run a single experiment by training a model for each group."""
        # Extract training data mapping
        training_data = self.config.extract_source_basins_for_training()

        # Dictionary to hold results by group
        group_results = {}

        # Train and evaluate a model for each group
        for group_key in training_data.keys():
            print(f"\n=== PROCESSING GROUP: {group_key} ===")

            # Load data for this group
            group_data = self.config.load_data_for_group(group_key)

            # Prepare data modules for this group
            data_module = self.prepare_data_module(group_key, group_data)

            # Train model for this group
            model = self.train_model(data_module, group_key, run)

            # Store model for later evaluation
            self.group_models[group_key] = model

            # Evaluate model on this group's data
            results = self.evaluate_model(model, data_module, group_key, run)

            group_results[group_key] = results

        # Evaluate each model on all groups to compare generalization
        self.evaluate_cross_group_performance(run)

        # Combine results from all groups
        combined_results = self.combine_group_results(group_results)

        return combined_results

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

        # Prepare data
        data_module.prepare_data()
        data_module.setup(stage="fit")

        return data_module

    def train_model(self, data_module, group_key, run):
        """Train a model for a specific group."""
        print(f"SETTING UP MODEL FOR GROUP {group_key} TRAINING")

        # Create TSMixer model
        model = LitTSMixer(self.config.get_tsmixer_config())

        # Train the model
        trainer = self.create_trainer(f"train_{group_key}", run)
        trainer.fit(model, data_module)

        # Save best model from checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path

        if best_checkpoint:
            # Load best model for both saving and returning
            best_model = LitTSMixer.load_from_checkpoint(best_checkpoint)

            # Save full model (architecture + weights)
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            val_loss = trainer.checkpoint_callback.best_model_score.item()

            # Create filename with validation loss
            model_filename = (
                f"tsmixer_{group_key}_run{run}_valloss{val_loss:.4f}_{timestamp}.pt"
            )
            save_path = self.model_dir / group_key / model_filename

            # Save model states
            torch.save(
                {
                    "state_dict": best_model.state_dict(),
                    "config": self.config.get_tsmixer_config().to_dict(),
                    "val_loss": val_loss,
                    "group_key": group_key,
                    "run": run,
                    "timestamp": timestamp,
                    "epoch": trainer.checkpoint_callback.best_model_path.split("=")[
                        -1
                    ].split(".")[0],
                },
                save_path,
            )

            print(f"Saved best model to {save_path} (val_loss: {val_loss:.4f})")
            return best_model
        else:
            # If no checkpoint was saved, save current model
            save_path = self.model_dir / group_key / f"tsmixer_{group_key}_run{run}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": self.config.get_tsmixer_config().to_dict(),
                    "group_key": group_key,
                    "run": run,
                },
                save_path,
            )
            print(f"No best checkpoint found, saved current model to {save_path}")
            return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        # Extract group key from stage if present
        if "_" in stage:
            stage_type, group_key = stage.split("_", 1)
            checkpoint_path = self.checkpoint_dir / group_key
            checkpoint_path.mkdir(exist_ok=True)

            # Define logs directory for CSVLogger
            logs_path = Path(
                f"experiments/GroupBased/logs/{self.config.EXPERIMENT_NAME}/{group_key}"
            )
            logs_path.mkdir(parents=True, exist_ok=True)
        else:
            stage_type = stage
            checkpoint_path = self.checkpoint_dir

            # Define logs directory for CSVLogger
            logs_path = Path(
                f"experiments/GroupBased/logs/{self.config.EXPERIMENT_NAME}"
            )
            logs_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp for unique logging
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the CSVLogger instance separately
        logger = pl.loggers.CSVLogger(
            save_dir=logs_path,
            name=f"{stage_type}_run{run}_{timestamp}",
            flush_logs_every_n_steps=100,
        )

        # Return Trainer with callbacks and logger set separately
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

    def evaluate_model(self, model, data_module, group_key, run):
        """Evaluate a model on its group's data."""
        print(f"EVALUATING MODEL FOR GROUP {group_key}")
        trainer = self.create_trainer(f"evaluate_{group_key}", run)
        trainer.test(model, data_module)

        evaluator = TSForecastEvaluator(
            data_module, horizons=list(range(1, self.config.OUTPUT_LENGTH + 1))
        )

        results_df, overall_metrics, basin_metrics = evaluator.evaluate(
            model.test_results
        )

        # Save results
        results_df.to_csv(
            self.results_dir / group_key / f"detailed_results_{run}.csv", index=True
        )

        overall_summary = evaluator.summarize_metrics(overall_metrics)
        overall_summary.to_csv(
            self.results_dir / group_key / f"overall_metrics_{run}.csv", index=True
        )

        basin_summary = evaluator.summarize_metrics(basin_metrics, per_basin=True)
        basin_summary.to_csv(
            self.results_dir / group_key / f"basin_metrics_{run}.csv", index=True
        )

        # Plot sample forecasts
        self.plot_sample_forecasts(model, data_module, evaluator, group_key, run)

        return {
            "overall_metrics": overall_metrics,
            "basin_metrics": basin_metrics,
            "results_df": results_df,
        }

    def evaluate_cross_group_performance(self, run):
        """Evaluate each group model on all other groups to test generalization."""
        print("\n=== CROSS-GROUP EVALUATION ===")

        training_data = self.config.extract_source_basins_for_training()

        # Create a data module for each group that contains only CA test data
        ca_test_modules = {}

        for group_key in training_data.keys():
            # Load only CA data for this group
            group_data = self.config.load_data_for_group(group_key)

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

            # Create CA-only data module
            preprocessing_configs = self.config.get_preprocessing_config()
            ca_test_modules[group_key] = HydroDataModule(
                time_series_df=ca_ts_data,
                static_df=ca_static_data,
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
                domain_id=f"ca_only_{group_key}",
            )

            ca_test_modules[group_key].prepare_data()
            ca_test_modules[group_key].setup(stage="test")

        # Evaluate each model on all CA test sets
        cross_group_results = {}

        for model_group in self.group_models:
            model = self.group_models[model_group]
            cross_group_results[model_group] = {}

            for target_group in ca_test_modules:
                print(f"Evaluating model from {model_group} on {target_group} data...")

                # Skip if same group (already evaluated)
                if model_group == target_group:
                    continue

                # Test model on this group's CA data
                test_datamodule = ca_test_modules[target_group]

                trainer = self.create_trainer(f"cross_eval", run)
                trainer.test(model, test_datamodule)

                evaluator = TSForecastEvaluator(
                    test_datamodule,
                    horizons=list(range(1, self.config.OUTPUT_LENGTH + 1)),
                )

                results_df, overall_metrics, basin_metrics = evaluator.evaluate(
                    model.test_results
                )

                # Save cross-evaluation results
                results_df.to_csv(
                    self.results_dir
                    / f"cross_{model_group}_on_{target_group}_run{run}.csv",
                    index=True,
                )

                overall_summary = evaluator.summarize_metrics(overall_metrics)
                overall_summary.to_csv(
                    self.results_dir
                    / f"cross_{model_group}_on_{target_group}_metrics_run{run}.csv",
                    index=True,
                )

                cross_group_results[model_group][target_group] = overall_metrics

        # Save combined cross-group evaluation
        cross_df_rows = []

        for model_group, target_groups in cross_group_results.items():
            for target_group, metrics in target_groups.items():
                for horizon, horizon_metrics in metrics.items():
                    row = {
                        "model_group": model_group,
                        "target_group": target_group,
                        "horizon": horizon,
                        **horizon_metrics,
                    }
                    cross_df_rows.append(row)

        if cross_df_rows:
            cross_df = pd.DataFrame(cross_df_rows)
            cross_df.to_csv(
                self.results_dir / f"cross_group_evaluation_run{run}.csv", index=False
            )

    def combine_group_results(self, group_results):
        """Combine results from all groups into a single structure."""
        combined = {
            "overall_metrics": {},
            "basin_metrics": {},
        }

        # Pool basin metrics from all groups
        for group_key, results in group_results.items():
            # Add basin metrics with group prefix to ensure uniqueness
            for basin_id, basin_data in results["basin_metrics"].items():
                combined["basin_metrics"][f"{group_key}_{basin_id}"] = basin_data

        # Calculate combined overall metrics (weighted by number of basins)
        horizon_metrics = {}
        basin_counts = {
            group_key: len(results["basin_metrics"])
            for group_key, results in group_results.items()
        }
        total_basins = sum(basin_counts.values())

        # For each horizon, calculate weighted average of metrics
        for horizon in range(1, self.config.OUTPUT_LENGTH + 1):
            horizon_metrics[horizon] = {}

            # Get metric names from first group
            first_group = next(iter(group_results.values()))
            metric_names = first_group["overall_metrics"][horizon].keys()

            for metric in metric_names:
                weighted_sum = 0
                for group_key, results in group_results.items():
                    if horizon in results["overall_metrics"]:
                        group_weight = basin_counts[group_key] / total_basins
                        weighted_sum += (
                            results["overall_metrics"][horizon][metric] * group_weight
                        )

                horizon_metrics[horizon][metric] = weighted_sum

        combined["overall_metrics"] = horizon_metrics

        return combined

    def plot_sample_forecasts(
        self, model, data_module, evaluator, group_key, run, num_basins=3
    ):
        """Plot sample forecasts for a group."""
        # Get unique basin IDs from test data
        basin_ids = data_module.test_dataset.gauge_ids

        # Filter to get only CA basins
        ca_basins = [bid for bid in basin_ids if str(bid).startswith("CA_")]

        # Select a subset of basins to visualize
        if len(ca_basins) > num_basins:
            sample_basins = ca_basins[:num_basins]
        else:
            sample_basins = ca_basins

        for basin_id in sample_basins:
            for horizon in [1, 5, 10]:  # Plot for different horizons
                try:
                    fig, ax = evaluator.plot_rolling_forecast(
                        horizon=horizon,
                        group_identifier=basin_id,
                        datamodule=data_module,
                        title=f"Basin {basin_id} (Group {group_key}): {horizon}-day Forecast",
                    )

                    # Save the plot
                    save_path = (
                        self.viz_dir
                        / group_key
                        / f"forecast_basin_{basin_id}_h{horizon}_run{run}.png"
                    )
                    fig.savefig(save_path, dpi=self.config.VIZ_DPI)
                    plt.close(fig)
                except Exception as e:
                    print(
                        f"Error plotting forecast for basin {basin_id}, horizon {horizon}: {e}"
                    )

    def cleanup(self):
        """Clean up resources after each run."""
        # Delete explicit references to large objects
        self.group_models = {}

        # Force Python garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_aggregated_results(self, all_results):
        """Save aggregated results across all runs."""
        if not all_results:
            print("Warning: No results to aggregate - all runs failed")
            return

        try:
            # Get group keys from first run
            if all_results and "overall_metrics" in all_results[0]:
                # Aggregate overall metrics across runs
                overall_metrics_df = pd.DataFrame()

                for i, run_result in enumerate(all_results):
                    if run_result and "overall_metrics" in run_result:
                        run_metrics = []
                        for horizon, metrics in run_result["overall_metrics"].items():
                            metrics_row = {"horizon": horizon, "run": i, **metrics}
                            run_metrics.append(metrics_row)

                        if run_metrics:
                            run_df = pd.DataFrame(run_metrics)
                            overall_metrics_df = pd.concat([overall_metrics_df, run_df])

                if not overall_metrics_df.empty:
                    # Calculate and save summary statistics
                    summary_stats = overall_metrics_df.groupby("horizon").agg(
                        ["mean", "std", "min", "max"]
                    )
                    summary_stats.to_csv(self.results_dir / "aggregate_metrics.csv")

                    print(
                        f"Successfully saved aggregate metrics for {len(all_results)} runs"
                    )
                else:
                    print("Warning: No valid metrics to aggregate")
            else:
                print("Warning: No valid results format for aggregation")

        except Exception as e:
            print(f"Error while saving aggregated results: {str(e)}")


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = GroupBasedRunner(config)
    runner.run_experiment()
