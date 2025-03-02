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
from src.data_models.datamodule import HydroDataModule, HydroTransferDataModule
from src.models.TSMixer import LitTSMixer
from src.models.TSMixerDomainAdaptation import LitTSMixerDomainAdaptation
from src.models.evaluators import TSForecastEvaluator
import multiprocessing


class AdversarialFinetuneRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for experiment outputs."""
        self.results_dir = Path(
            f"experiments/adversarial_finetune/results/{self.config.EXPERIMENT_NAME}"
        )
        self.model_dir = Path(
            f"experiments/adversarial_finetune/saved_models/{self.config.EXPERIMENT_NAME}"
        )
        self.checkpoint_dir = Path(
            f"experiments/adversarial_finetune/checkpoints/{self.config.EXPERIMENT_NAME}"
        )
        self.viz_dir = Path(
            f"experiments/adversarial_finetune/visualizations/{self.config.EXPERIMENT_NAME}"
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
        """Run a single experiment with two phases: adversarial training and fine-tuning."""
        # Get preprocessing configs based on CH data
        preprocessing_configs = self.config.get_preprocessing_config("CH")

        # Create source and target data modules
        ch_data_module = self.create_data_module(
            self.ch_ts_data,
            self.ch_static_data,
            preprocessing_configs,
            is_source=True,
            domain_id="source",
        )

        ca_data_module = self.create_data_module(
            self.ca_ts_data,
            self.ca_static_data,
            preprocessing_configs,
            is_source=False,
            domain_id="target",
        )

        # Create transfer data module for domain adaptation
        transfer_data_module = HydroTransferDataModule(
            source_datamodule=ch_data_module,
            target_datamodule=ca_data_module,
            num_workers=self.config.MAX_WORKERS,
        )

        # Phase 1: Adversarial domain adaptation training
        print("\n=== PHASE 1: ADVERSARIAL DOMAIN ADAPTATION ===")
        adapted_model = self.run_adversarial_training(transfer_data_module, run)

        # Visualize domain adaptation after Phase 1
        self.visualize_domain_adaptation(
            adapted_model, ch_data_module, ca_data_module, run, phase="phase1"
        )

        # Phase 2: Fine-tune on target data with frozen backbone
        print("\n=== PHASE 2: FINE-TUNING ON TARGET DATA ===")
        fine_tuned_model = self.run_fine_tuning(adapted_model, ca_data_module, run)

        # Evaluate final model
        results = self.evaluate_model(fine_tuned_model, ca_data_module, run)

        # Visualize final domain adaptation
        self.visualize_domain_adaptation(
            adapted_model,  # Use the adapted model for visualization
            ch_data_module,
            ca_data_module,
            run,
            phase="final",
        )

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
            domain_id=domain_id,
        )

        # Explicitly prepare and set up the data module
        dm.prepare_data()
        dm.setup(stage="fit")

        return dm

    def run_adversarial_training(self, transfer_data_module, run):
        """Run adversarial domain adaptation training."""
        print("CONFIGURING MODEL FOR ADVERSARIAL DOMAIN ADAPTATION")

        # Create domain adaptation model using the config
        model = LitTSMixerDomainAdaptation(self.config.get_domain_adaptation_config())

        # Train with domain adaptation
        trainer = self.create_trainer(
            "adversarial", run, max_epochs=self.config.MAX_EPOCHS
        )
        trainer.fit(model, transfer_data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_adversarial_{run}.pt"
        torch.save(model.state_dict(), save_path)

        return model

    def run_fine_tuning(self, adapted_model, target_data_module, run):
        """Fine-tune on target data with frozen backbone."""
        print("CONFIGURING MODEL FOR FINE-TUNING")

        # Create a standard LitTSMixer model with fine-tuning configuration
        fine_tune_model = LitTSMixer(self.config.get_finetune_config())

        # Extract TSMixer weights from the domain-adapted model
        adapted_tsmixer_state_dict = {}
        for key, value in adapted_model.state_dict().items():
            if key.startswith("model."):
                new_key = key.replace("model.", "")
                adapted_tsmixer_state_dict[new_key] = value

        # Transfer weights to the fine-tuning model
        missing_keys, unexpected_keys = fine_tune_model.model.load_state_dict(
            adapted_tsmixer_state_dict, strict=False
        )

        if missing_keys:
            print(f"Warning: Missing keys when transferring weights: {missing_keys}")
        if unexpected_keys:
            print(
                f"Warning: Unexpected keys when transferring weights: {unexpected_keys}"
            )

        print("Transferred weights from domain-adapted model to fine-tuning model")

        # Freeze backbone if configured
        if self.config.FREEZE_BACKBONE:
            fine_tune_model.freeze_backbone()
            print("Backbone frozen for fine-tuning")

        # Train with fine-tuning
        trainer = self.create_trainer(
            "finetune", run, max_epochs=self.config.MAX_EPOCHS // 2
        )
        trainer.fit(fine_tune_model, target_data_module)

        # Save model
        save_path = self.model_dir / f"tsmixer_finetuned_{run}.pt"
        torch.save(fine_tune_model.state_dict(), save_path)

        return fine_tune_model

    def create_trainer(self, stage, run, max_epochs=None):
        """Create a PyTorch Lightning trainer with appropriate callbacks."""
        if max_epochs is None:
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
            data_module, horizons=list(range(1, self.config.OUTPUT_LENGTH + 1))
        )

        results_df, overall_metrics, basin_metrics = evaluator.evaluate(
            model.test_results
        )

        # Save results
        results_df.to_csv(self.results_dir / f"detailed_results_{run}.csv", index=True)

        overall_summary = evaluator.summarize_metrics(overall_metrics)
        overall_summary.to_csv(
            self.results_dir / f"overall_metrics_{run}.csv", index=True
        )

        basin_summary = evaluator.summarize_metrics(basin_metrics, per_basin=True)
        basin_summary.to_csv(self.results_dir / f"basin_metrics_{run}.csv", index=True)

        return {
            "overall_metrics": overall_metrics,
            "basin_metrics": basin_metrics,
            "results_df": results_df,
        }

    def visualize_domain_adaptation(
        self, model, source_dm, target_dm, run, phase="final"
    ):
        """Create and save visualizations of domain adaptation."""
        print(f"CREATING DOMAIN ADAPTATION VISUALIZATION FOR {phase}")

        # Need to convert the model to domain adaptation model for visualization if it's not already
        if not isinstance(model, LitTSMixerDomainAdaptation):
            viz_model = LitTSMixerDomainAdaptation(
                self.config.get_domain_adaptation_config()
            )

            # Load weights from the provided model
            state_dict = model.state_dict()
            new_state_dict = {}

            # Transform the state dict keys if needed
            for key, value in state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            # Load weights, ignoring missing keys (domain discriminator)
            viz_model.load_state_dict(new_state_dict, strict=False)
        else:
            viz_model = model

        # Create dataloaders for visualization
        source_dl = source_dm.train_dataloader()
        target_dl = target_dm.train_dataloader()

        # Generate visualization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        viz_model = viz_model.to(device)

        fig = viz_model.visualize_domain_adaptation(
            source_dl,
            target_dl,
            max_samples=self.config.VIZ_MAX_SAMPLES,
            device=device,
            perplexity=self.config.VIZ_PERPLEXITY,
            figsize=(10, 8),
            title=f"Domain Adaptation Visualization - {phase.title()} (Run {run})",
        )

        # Save the figure
        if fig:
            save_path = self.viz_dir / f"domain_adaptation_{phase}_run{run}.png"
            fig.savefig(save_path, dpi=self.config.VIZ_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved domain adaptation visualization to {save_path}")
        else:
            print(f"Warning: No visualization generated for {phase}")

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
            summary_stats.to_csv(self.results_dir / "aggregate_metrics.csv")

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
    runner = AdversarialFinetuneRunner(config)
    runner.load_data()
    runner.run_experiment()
