import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

# import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import pandas as pd
import matplotlib.pyplot as plt
from experiments.adversarial_finetune.configAdversarialFinetune import ExperimentConfig
from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
from src.models.evaluators import TSForecastEvaluator
import multiprocessing


class BenchmarkRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/adversarial_finetune/results/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.model_dir = Path(
            f"experiments/adversarial_finetune/saved_models/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.checkpoint_dir = Path(
            f"experiments/adversarial_finetune/checkpoints/{self.config.EXPERIMENT_NAME}_benchmark"
        )
        self.viz_dir = Path(
            f"experiments/adversarial_finetune/visualizations/{self.config.EXPERIMENT_NAME}_benchmark"
        )

        for directory in [
            self.results_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.viz_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load and prepare dataset for only the target domain (CA)."""
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
        """Run a single experiment by training directly on target data."""
        # Get preprocessing configs for CA
        preprocessing_configs = self.config.get_preprocessing_config("CA")

        # Create target data module
        ca_data_module = self.create_data_module(
            self.ca_ts_data,
            self.ca_static_data,
            preprocessing_configs,
        )

        # Train model directly on target data
        print("\n=== TRAINING BENCHMARK MODEL ON TARGET DATA ===")
        trained_model = self.train_model(ca_data_module, run)

        # Evaluate model
        results = self.evaluate_model(trained_model, ca_data_module, run)

        # Visualize feature space
        self.plot_feature_space(trained_model, ca_data_module, run)

        return results

    def create_data_module(
        self,
        ts_data,
        static_data,
        preprocessing_configs,
    ):
        """Create a data module for target domain."""
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
        """Train TSMixer directly on target data."""
        print("SETTING UP MODEL FOR TRAINING")

        # Create a TSMixer model with pretrain learning rate
        model = LitTSMixer(self.config.get_tsmixer_config())

        # Train on target data
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
            data_module, horizons=list(range(1, self.config.OUTPUT_LENGTH + 1))
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

    def plot_feature_space(self, model, data_module, run):
        """Plot the feature space of the benchmark model using t-SNE."""
        from sklearn.manifold import TSNE
        import seaborn as sns

        print("CREATING FEATURE SPACE VISUALIZATION")

        # Get a batch of data
        dataloader = data_module.train_dataloader()
        batch = next(iter(dataloader))

        # Extract features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            x = batch["X"].to(device)
            static = batch["static"].to(device)
            # Extract features from the backbone
            features = model.model.backbone(x, static)
            # Flatten features
            flattened_features = features.reshape(features.size(0), -1).cpu().numpy()

        # Color by basin ID (convert to numeric)
        basin_ids = batch[self.config.GROUP_IDENTIFIER]
        unique_basins = list(set(basin_ids))
        basin_to_idx = {basin: i for i, basin in enumerate(unique_basins)}
        basin_indices = [basin_to_idx[basin] for basin in basin_ids]

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(flattened_features) - 1),
            random_state=42,
        )
        features_2d = tsne.fit_transform(flattened_features)

        # Create plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            hue=basin_indices,
            palette="viridis",
            s=50,
            alpha=0.8,
        )

        plt.title(f"Feature Space Visualization - Benchmark Model (Run {run})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title="Basin", loc="best", bbox_to_anchor=(1.05, 1), ncol=1)
        plt.tight_layout()

        # Save the figure
        save_path = self.viz_dir / f"benchmark_feature_space_run{run}.png"
        plt.savefig(save_path, dpi=self.config.VIZ_DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved feature space visualization to {save_path}")

    def cleanup(self):
        """Clean up resources after each run."""
        # Delete explicit references to large objects
        if hasattr(self, "adapted_model"):
            del self.adapted_model
        if hasattr(self, "fine_tuned_model"):
            del self.fine_tuned_model

        # Only clear CUDA cache if needed (check GPU memory usage first)
        if (
            torch.cuda.is_available() and torch.cuda.memory_allocated() > 1e9
        ):  # Only if >1GB used
            torch.cuda.empty_cache()

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
            summary_stats.to_csv(self.results_dir / "benchmark_aggregate_metrics.csv")

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
