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
from src.data_models.datamodule import HydroDataModule, HydroTransferDataModule
from src.models.TSMixerDomainAdaptation import LitTSMixerDomainAdaptation
from src.models.evaluators import TSForecastEvaluator
import multiprocessing


class ChallengerRunner:
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
        """Load and prepare datasets for both domains."""
        # CH Dataset (source domain)
        print("CONFIGURING CH DATASET (SOURCE DOMAIN)")
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

        # CA Dataset (target domain)
        print("CONFIGURING CA DATASET (TARGET DOMAIN)")
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
        """Run a single experiment with direct domain adaptation training."""
        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config("CH")

        # Create source and target data modules
        ch_data_module = self.create_data_module(
            self.ch_ts_data,
            self.ch_static_data,
            preprocessing_configs,
            is_source=True,
            domain_id=0,  # Source domain = 0
        )

        ca_data_module = self.create_data_module(
            self.ca_ts_data,
            self.ca_static_data,
            preprocessing_configs,
            is_source=False,
            domain_id=1,  # Target domain = 1
        )

        # Create transfer data module for domain adaptation
        transfer_data_module = HydroTransferDataModule(
            source_datamodule=ch_data_module,
            target_datamodule=ca_data_module,
            num_workers=self.config.MAX_WORKERS,
        )

        # Train with domain adaptation
        print("\n=== DIRECT DOMAIN ADAPTATION TRAINING ===")
        model = self.train_domain_adaptation_model(transfer_data_module, run)

        # Evaluate on target domain
        results = self.evaluate_model(model, ca_data_module, run)

        # Visualize domain adaptation
        self.visualize_domain_adaptation(model, ch_data_module, ca_data_module, run)

        return results

    def create_data_module(
        self,
        ts_data,
        static_data,
        preprocessing_configs,
        is_source: bool,
        domain_id=None,
    ):
        """Create a data module with appropriate configuration."""
        # Get the appropriate domain config
        domain_config = self.config.CH_CONFIG if is_source else self.config.CA_CONFIG

        dm = HydroDataModule(
            time_series_df=ts_data,
            static_df=static_data,
            group_identifier=self.config.GROUP_IDENTIFIER,
            preprocessing_config=preprocessing_configs,
            batch_size=self.config.BATCH_SIZE,
            input_length=self.config.INPUT_LENGTH,
            output_length=self.config.OUTPUT_LENGTH,
            num_workers=min(self.config.MAX_WORKERS, multiprocessing.cpu_count()),
            features=self.config.FORCING_FEATURES + [self.config.TARGET],
            static_features=self.config.STATIC_FEATURES,
            target=self.config.TARGET,
            min_train_years=domain_config["MIN_TRAIN_YEARS"],
            val_years=domain_config["VAL_YEARS"],
            test_years=domain_config["TEST_YEARS"],
            max_missing_pct=domain_config["MAX_MISSING_PCT"],
            domain_id="source" if is_source else "target",  # Add domain identifier
        )

        # Explicitly prepare and set up the data module
        dm.prepare_data()
        dm.setup(stage="fit")

        return dm

    def train_domain_adaptation_model(self, transfer_data_module, run):
        """Train domain adaptation model."""
        print("CONFIGURING MODEL FOR DOMAIN ADAPTATION")

        # Create domain adaptation model using the config
        model = LitTSMixerDomainAdaptation(self.config.get_domain_adaptation_config())

        # Train with domain adaptation
        trainer = self.create_trainer("domain_adapt", run)
        trainer.fit(model, transfer_data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_challenger_{run}.pt"
        torch.save(model.state_dict(), save_path)

        return model

    def create_trainer(self, stage, run):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        max_epochs = self.config.MAX_EPOCHS

        return pl.Trainer(
            max_epochs=max_epochs,
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

    def visualize_domain_adaptation(self, model, source_dm, target_dm, run):
        """Create and save visualizations of domain adaptation."""
        print("CREATING DOMAIN ADAPTATION VISUALIZATION")

        # Create dataloaders for visualization
        source_dl = source_dm.train_dataloader()
        target_dl = target_dm.train_dataloader()

        # Generate visualization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        fig = model.visualize_domain_adaptation(
            source_dl,
            target_dl,
            max_samples=self.config.VIZ_MAX_SAMPLES,
            device=device,
            perplexity=self.config.VIZ_PERPLEXITY,
            figsize=(10, 8),
            title=f"Challenger Domain Adaptation (Run {run})",
        )

        # Save the figure
        if fig:
            save_path = self.viz_dir / f"challenger_domain_adaptation_run{run}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved domain adaptation visualization to {save_path}")

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
    runner = ChallengerRunner(config)
    runner.load_data()
    runner.run_experiment()
