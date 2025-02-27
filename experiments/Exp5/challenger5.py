import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import pandas as pd
import numpy as np
from experiments.Exp5.config5 import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
from src.models.evaluators import TSForecastEvaluator
import multiprocessing


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path("experiments/Exp5/results")
        self.model_dir = Path("experiments/Exp5/saved_models")
        self.checkpoint_dir = Path("experiments/Exp5/checkpoints")

        for directory in [self.results_dir, self.model_dir, self.checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load and prepare datasets for both domains."""
        # CH Dataset
        print("CONFIGURING CH DATASET FOR PRETRAINING")
        ch_config = CaravanifyConfig(
            attributes_dir=self.config.CH_CONFIG["ATTRIBUTE_DIR"],
            timeseries_dir=self.config.CH_CONFIG["TIMESERIES_DIR"],
            gauge_id_prefix=self.config.CH_CONFIG["GAUGE_ID_PREFIX"],
            use_hydroatlas_attributes=True,
            use_caravan_attributes=True,
            use_other_attributes=True,
        )

        self.ch_caravan = Caravanify(ch_config)
        ch_basins = self.ch_caravan.get_all_gauge_ids()
        print(f"Loading {len(ch_basins)} CH basins")
        self.ch_caravan.load_stations(ch_basins)

        # CA Dataset
        print("CONFIGURING CA DATASET FOR FINE-TUNING")
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

        self.ch_ts_data = self.ch_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ch_static_data = self.ch_caravan.get_static_attributes()[static_columns]

        self.ca_ts_data = self.ca_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

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
        """Run a single experiment including pretraining, fine-tuning and evaluation."""
        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config("CH")

        # Create data modules
        ch_data_module = self.create_data_module(
            self.ch_ts_data, self.ch_static_data, preprocessing_configs, is_source=True
        )

        ca_data_module = self.create_data_module(
            self.ca_ts_data, self.ca_static_data, preprocessing_configs, is_source=False
        )

        # Pretrain
        pretrain_model = self.run_pretraining(ch_data_module, run)

        # Fine-tune
        finetune_model = self.run_finetuning(pretrain_model, ca_data_module, run)

        # Evaluate
        return self.evaluate_model(finetune_model, ca_data_module, run)

    def create_data_module(
        self, ts_data, static_data, preprocessing_configs, is_source: bool
    ):
        """Create a data module with appropriate configuration."""
        # Get the appropriate domain config using dot notation
        domain_config = self.config.CH_CONFIG if is_source else self.config.CA_CONFIG

        return HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.CH_CONFIG["BATCH_SIZE"],
            input_length=self.config.CH_CONFIG["INPUT_LENGTH"],
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            min_train_years=domain_config["MIN_TRAIN_YEARS"],
            val_years=domain_config["VAL_YEARS"],
            test_years=domain_config["TEST_YEARS"],
            max_missing_pct=domain_config["MAX_MISSING_PCT"],
        )

    def run_pretraining(self, data_module, run):
        """Run pretraining phase."""
        print("SETTING UP MODEL FOR PRETRAINING")
        model = LitTSMixer(
            input_len=self.config.CH_CONFIG["INPUT_LENGTH"],
            output_len=self.config.OUTPUT_LENGTH,
            input_size=len(self.config.FORCING_FEATURES) + 1,  # +1 for target
            static_size=len(self.config.STATIC_FEATURES) - 1,  # -1 for gauge_id
            hidden_size=self.config.CH_CONFIG["HIDDEN_SIZE"],
            learning_rate=self.config.PRETRAIN_LR,
            dropout=self.config.CH_CONFIG["DROPOUT"],
        )

        trainer = self.create_trainer("pretrain", run)
        trainer.fit(model, data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_challenger_pretrained_{run}.pt"
        torch.save(model.state_dict(), save_path)

        return model

    def run_finetuning(self, pretrain_model, data_module, run):
        """Run fine-tuning phase."""
        print("CONFIGURING MODEL FOR FINE-TUNING")
        model = LitTSMixer(
            input_len=self.config.CH_CONFIG["INPUT_LENGTH"],
            output_len=self.config.OUTPUT_LENGTH,
            input_size=len(self.config.FORCING_FEATURES) + 1,
            static_size=len(self.config.STATIC_FEATURES) - 1,
            hidden_size=self.config.CH_CONFIG["HIDDEN_SIZE"],
            learning_rate=self.config.BENCHMARK_LR,
            dropout=self.config.CH_CONFIG["DROPOUT"],
        )

        # Load pretrained weights
        model.load_state_dict(pretrain_model.state_dict())

        trainer = self.create_trainer("finetune", run)
        trainer.fit(model, data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_challenger_{run}.pt"
        torch.save(model.state_dict(), save_path)

        return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        return pl.Trainer(
            max_epochs=self.config.MAX_EPOCHS,
            accelerator=self.config.ACCELERATOR,
            devices=1,
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

    def evaluate_model(self, model, data_module, run):
        """Evaluate the model and save results."""
        print("STARTING MODEL EVALUATION")
        trainer = self.create_trainer("evaluate", run)
        trainer.test(model, data_module)

        evaluator = TSForecastEvaluator(
            data_module, horizons=list(range(1, model.config.pred_len + 1))
        )

        results_df, overall_metrics, basin_metrics = evaluator.evaluate(
            model.test_results
        )

        # Save results
        results_df.to_csv(
            self.results_dir / f"challenger_detailed_results_{run}.csv", index=True
        )

        overall_summary = evaluator.summarize_metrics(overall_metrics)
        overall_summary.to_csv(
            self.results_dir / f"challenger_overall_metrics_{run}.csv", index=True
        )

        basin_summary = evaluator.summarize_metrics(basin_metrics, per_basin=True)
        basin_summary.to_csv(
            self.results_dir / f"challenger_basin_metrics_{run}.csv", index=True
        )

        return {
            "overall_metrics": overall_metrics,
            "basin_metrics": basin_metrics,
            "results_df": results_df,
        }

    def cleanup(self):
        """Clean up resources after each run."""
        torch.cuda.empty_cache()
        gc.collect()

    def save_aggregated_results(self, all_results):
        """Save aggregated results across all runs."""
        if not all_results:
            print("Warning: No results to aggregate - all runs failed")
            return

        try:
            # Combine overall metrics across runs
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

            # Calculate and save summary statistics
            summary_stats = overall_metrics_df.groupby(level=0).agg(
                ["mean", "std", "min", "max"]
            )
            summary_stats.to_csv(self.results_dir / "challenger_aggregate_metrics.csv")

            print(f"Successfully saved aggregate metrics for {len(all_results)} runs")

        except Exception as e:
            print(f"Error while saving aggregated results: {str(e)}")


if __name__ == "__main__":
    # Initialize config
    config = ExperimentConfig()

    # Set CUDA precision
    if config.ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Run experiment
    runner = ExperimentRunner(config)
    runner.load_data()
    runner.run_experiment()
