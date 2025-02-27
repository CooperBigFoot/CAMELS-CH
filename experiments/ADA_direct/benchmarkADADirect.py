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
import matplotlib.pyplot as plt
from experiments.ADA_direct.configADADirect import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
from src.models.TSMixerDomainAdaptation import LitTSMixerDomainAdaptation
from src.models.evaluators import TSForecastEvaluator
import multiprocessing


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/ADA_direct/results/{self.config.EXPERIMENT_NAME}"
        )
        self.model_dir = Path(
            f"experiments/ADA_direct/saved_models/{self.config.EXPERIMENT_NAME}"
        )
        self.checkpoint_dir = Path(
            f"experiments/ADA_direct/checkpoints/{self.config.EXPERIMENT_NAME}"
        )
        self.viz_dir = Path(
            f"experiments/ADA_direct/visualizations/{self.config.EXPERIMENT_NAME}"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.viz_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load both CA and CH datasets for visualization later."""
        # Load CA dataset (target domain)
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

        # Load CH dataset (source domain) for visualization
        print("CONFIGURING CH DATASET")
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

        # Prepare data frames
        ts_columns = self.config.FORCING_FEATURES + [self.config.TARGET]
        static_columns = self.config.STATIC_FEATURES

        self.ca_ts_data = self.ca_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ca_static_data = self.ca_caravan.get_static_attributes()[static_columns]

        self.ch_ts_data = self.ch_caravan.get_time_series()[
            ts_columns + ["date"] + [self.config.GROUP_IDENTIFIER]
        ]
        self.ch_static_data = self.ch_caravan.get_static_attributes()[static_columns]

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
        """Run a single experiment including training and evaluation."""
        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config("CA")

        # Create CA data module for training
        ca_data_module = HydroDataModule(
            time_series_df=self.ca_ts_data,
            static_df=self.ca_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            domain_id="target",  # Mark as target domain
            min_train_years=self.config.CA_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CA_CONFIG["VAL_YEARS"],
            test_years=self.config.CA_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CA_CONFIG["MAX_MISSING_PCT"],
        )

        # Call prepare_data and setup for CA explicitly
        ca_data_module.prepare_data()
        ca_data_module.setup()

        # Train and evaluate
        model = self.train_model(ca_data_module, run)
        results = self.evaluate_model(model, ca_data_module, run)

        # Create CH data module for visualization only
        # Using the same preprocessing config trained on CA data
        ch_data_module = HydroDataModule(
            time_series_df=self.ch_ts_data,
            static_df=self.ch_static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            domain_id="source",  # Mark as source domain
            min_train_years=self.config.CH_CONFIG["MIN_TRAIN_YEARS"],
            val_years=self.config.CH_CONFIG["VAL_YEARS"],
            test_years=self.config.CH_CONFIG["TEST_YEARS"],
            max_missing_pct=self.config.CH_CONFIG["MAX_MISSING_PCT"],
        )

        # Call prepare_data and setup for CH explicitly
        ch_data_module.prepare_data()
        ch_data_module.setup(stage="fit")  # Explicitly setup for training

        # Visualize domain adaptation
        self.visualize_domain_adaptation(model, ch_data_module, ca_data_module, run)

        return results

    def train_model(self, data_module, run):
        """Train the model."""
        print("SETTING UP MODEL FOR TRAINING")

        # Create the model using configuration object
        model = LitTSMixer(self.config.get_tsmixer_config())

        trainer = self.create_trainer("train", run)
        trainer.fit(model, data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_benchmark_{run}.pt"
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
            data_module, horizons=list(range(1, model.config.output_len + 1))
        )

        results_df, overall_metrics, basin_metrics = evaluator.evaluate(
            model.test_results
        )

        # Save results
        results_df.to_csv(
            self.results_dir / f"benchmark_detailed_results_{run}.csv", index=True
        )

        overall_summary = evaluator.summarize_metrics(overall_metrics)
        overall_summary.to_csv(
            self.results_dir / f"benchmark_overall_metrics_{run}.csv", index=True
        )

        basin_summary = evaluator.summarize_metrics(basin_metrics, per_basin=True)
        basin_summary.to_csv(
            self.results_dir / f"benchmark_basin_metrics_{run}.csv", index=True
        )

        return {
            "overall_metrics": overall_metrics,
            "basin_metrics": basin_metrics,
            "results_df": results_df,
        }

    def visualize_domain_adaptation(self, trained_model, source_dm, target_dm, run):
        """Create and save visualizations of domain adaptation using a trained model."""
        print("CREATING DOMAIN ADAPTATION VISUALIZATION")

        # Create a domain adaptation model and load weights from the trained TSMixer
        da_model = LitTSMixerDomainAdaptation(
            self.config.get_domain_adaptation_config()
        )

        # Transfer weights from TSMixer to DA model's backbone
        tsmixer_state_dict = trained_model.state_dict()
        da_state_dict = {}

        for key, value in tsmixer_state_dict.items():
            if key.startswith("model."):
                new_key = key  # Keep the key as is
            else:
                new_key = f"model.{key}"  # Add 'model.' prefix

            da_state_dict[new_key] = value

        # Load weights, ignoring missing keys (domain discriminator)
        missing_keys, unexpected_keys = da_model.load_state_dict(
            da_state_dict, strict=False
        )
        print(f"Missing keys when loading: {missing_keys}")
        print(f"Unexpected keys when loading: {unexpected_keys}")

        # Create dataloaders for visualization
        source_dl = source_dm.train_dataloader()
        target_dl = target_dm.train_dataloader()

        # Generate visualization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        da_model = da_model.to(device)

        fig = da_model.visualize_domain_adaptation(
            source_dl,
            target_dl,
            max_samples=self.config.VIZ_MAX_SAMPLES,
            device=device,
            perplexity=self.config.VIZ_PERPLEXITY,
            figsize=(10, 8),
            title=f"Benchmark Domain Adaptation (Run {run})",
        )

        # Save the figure
        if fig:
            save_path = self.viz_dir / f"benchmark_domain_adaptation_run{run}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved domain adaptation visualization to {save_path}")
        else:
            print("Warning: No visualization generated")

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
            summary_stats.to_csv(self.results_dir / f"benchmark_aggregate_metrics.csv")

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
    runner = BenchmarkRunner(config)
    runner.load_data()
    runner.run_experiment()
