"""Hyperparameter tuning implementation using Optuna."""

import sys
from pathlib import Path
import gc
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.HyperparameterTune.hyperparameter_space import get_hyperparameter_space
from src.data_models.datamodule import HydroDataModule
from src.models.model_factory import create_model
from experiments.HyperparameterTune.utils import save_study_results


class HyperparameterTuner:
    """Hyperparameter tuning framework using Optuna for hydrological models."""

    def __init__(self, config, model_type, country, study_name=None, dirs=None):
        """Initialize the hyperparameter tuner.
        
        Args:
            config: Configuration object
            model_type: Type of model to tune ('tide', 'tsmixer', 'ealstm', 'tft')
            country: Country to use for tuning ('Tajikistan', 'Kyrgyzstan', 'Combined')
            study_name: Optional name for the Optuna study
            dirs: Directory structure for saving outputs (if None, will use config.get_*_dir)
        """
        self.config = config
        self.model_type = model_type.lower()
        self.country = country
        self.study_name = study_name or f"{self.model_type}_{country.lower()}_optimization"
        self.dirs = dirs
        
        # Get hyperparameter search space
        self.search_space = get_hyperparameter_space(model_type)
        
        # Configure Optuna study
        self.sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="minimize", study_name=self.study_name, sampler=self.sampler
        )
        
        # Flag to track if data has been loaded
        self._data_loaded = False
        self.data = None
    
    def load_data(self):
        """Load data for the specified country."""
        from experiments.HyperparameterTune.data_loader import load_data
        
        print(f"Loading data for country: {self.country}")
        self.data = load_data(self.config, country=self.country)
        self._data_loaded = True
        
        # Store basin count for reporting
        self.basin_count = self.data["basin_count"]
        print(f"Loaded {self.basin_count} basins for {self.country}")
    
    def sample_hyperparameters(self, trial):
        """Sample hyperparameters based on model type and search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        hyperparameters = {}
        
        # Sample common hyperparameters
        for param_name, param_config in self.search_space["common"].items():
            param_type = param_config["type"]
            
            if param_type == "int":
                step = param_config.get("step", 1)
                hyperparameters[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], step=step
                )
            elif param_type == "float":
                log = param_config.get("log", False)
                hyperparameters[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=log
                )
            elif param_type == "categorical":
                hyperparameters[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
        
        # Sample model-specific hyperparameters
        if "model_specific" in self.search_space:
            for param_name, param_config in self.search_space["model_specific"].items():
                param_type = param_config["type"]
                
                if param_type == "int":
                    step = param_config.get("step", 1)
                    hyperparameters[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"], step=step
                    )
                elif param_type == "float":
                    log = param_config.get("log", False)
                    hyperparameters[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], log=log
                    )
                elif param_type == "categorical":
                    hyperparameters[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
        
        return hyperparameters
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Best validation loss achieved
        """
        # Make sure data is loaded
        if not self._data_loaded:
            self.load_data()
        
        # Sample hyperparameters
        hyperparameters = self.sample_hyperparameters(trial)
        
        # Store trial information
        trial.set_user_attr("basin_count", self.basin_count)
        trial.set_user_attr("country", self.country)
        trial.set_user_attr("model_type", self.model_type)
        
        # Get data
        time_series_data = self.data["time_series"]
        static_data = self.data["static"]
        
        # Extract hyperparameters for data module
        input_length = hyperparameters.get("input_length", self.config.input_length)
        batch_size = self.config.batch_size
        
        # Get preprocessing configs
        preprocessing_configs = self.config.get_preprocessing_config()
        
        # Create data module
        data_module = HydroDataModule(
            time_series_df=time_series_data,
            static_df=static_data,
            group_identifier=self.config.group_identifier,
            preprocessing_config=preprocessing_configs,
            input_length=input_length,
            output_length=self.config.output_length,
            batch_size=batch_size,
            num_workers=min(self.config.max_workers, multiprocessing.cpu_count()),
            features=self.config.forcing_features + [self.config.target],
            static_features=[f for f in self.config.static_features if f != "country"],
            target=self.config.target,
            use_proportional_split=self.config.use_proportional_split,
            train_prop=self.config.train_prop,
            val_prop=self.config.val_prop,
            test_prop=self.config.test_prop,
            min_train_years=self.config.ca_min_train_years,
        )
        
        # Prepare data
        data_module.prepare_data()
        data_module.setup()
        
        # Log dataset sizes
        train_size = (
            len(data_module.train_dataset)
            if hasattr(data_module, "train_dataset")
            else 0
        )
        val_size = (
            len(data_module.val_dataset) 
            if hasattr(data_module, "val_dataset") 
            else 0
        )
        test_size = (
            len(data_module.test_dataset) 
            if hasattr(data_module, "test_dataset") 
            else 0
        )
        
        print(
            f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}"
        )
        
        # Store for later logging
        trial.set_user_attr("train_size", train_size)
        trial.set_user_attr("val_size", val_size)
        trial.set_user_attr("test_size", test_size)
        
        # Create temporary YAML file with hyperparameters for model creation
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name
            yaml.dump(hyperparameters, f)
        
        try:
            # Create model
            model, _ = create_model(self.model_type, yaml_path)
            
            # Clean up temporary file
            import os
            os.unlink(yaml_path)
            
            # Set up directories
            if self.dirs:
                # Use provided directory structure
                checkpoint_dir = self.dirs[self.country.lower()]["checkpoints"][self.model_type] / f"trial_{trial.number}"
                logs_dir = self.dirs[self.country.lower()]["logs"][self.model_type]
            else:
                # Use config methods to get directories
                checkpoint_dir = self.config.get_checkpoint_dir(self.country, self.model_type) / f"trial_{trial.number}"
                logs_dir = self.config.get_logs_dir(self.country, self.model_type)
            
            # Create checkpoint directory
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up TensorBoard logger
            tb_logger = TensorBoardLogger(
                save_dir=str(logs_dir),
                name=f"{self.country}_{self.model_type}",
                version=f"trial_{trial.number}",
            )
            
            # Configure callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=self.config.early_stopping_patience, 
                    mode="min",
                    min_delta=self.config.early_stopping_min_delta
                ),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename=f"{self.country.lower()}_{self.model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=self.config.save_top_k,
                    save_last=self.config.save_last,
                ),
            ]
            
            # Configure trainer
            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                accelerator=self.config.accelerator,
                devices=1,
                logger=tb_logger,
                callbacks=callbacks,
                enable_progress_bar=True,
            )
            
            # Train and get best validation loss
            try:
                trainer.fit(model, data_module)
                
                # Get the best validation loss
                best_val_loss = trainer.callback_metrics["val_loss"].item()
                
                # Log additional information
                trial.set_user_attr("best_epoch", trainer.current_epoch)
                trial.set_user_attr("checkpoint_path", str(checkpoint_dir / "last.ckpt"))
                
                return best_val_loss
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                # Cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
                
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            raise optuna.exceptions.TrialPruned()
    
    def run_optimization(self):
        """Run the hyperparameter optimization study.
        
        Returns:
            Completed Optuna study object
        """
        print(f"\n===== Starting hyperparameter optimization for {self.model_type} on {self.country} data =====")
        print(f"Running {self.config.n_trials} trials...")
        
        # Run optimization
        try:
            self.study.optimize(self.objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
            
            # Save results
            if self.dirs:
                results_dir = self.dirs[self.country.lower()]["results"][self.model_type]
            else:
                results_dir = self.config.get_results_dir(self.country, self.model_type)
                
            save_study_results(self.study, self.model_type, self.country, results_dir)
            
            print("\nBest trial:")
            print(f"  Value: {self.study.best_trial.value:.5f}")
            print("  Params: ")
            for key, value in self.study.best_trial.params.items():
                print(f"    {key}: {value}")
            
            return self.study
            
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            
            # Still save partial results if available
            if self.study.trials:
                if self.dirs:
                    results_dir = self.dirs[self.country.lower()]["results"][self.model_type]
                else:
                    results_dir = self.config.get_results_dir(self.country, self.model_type)
                    
                save_study_results(self.study, self.model_type, self.country, results_dir)
            
            return self.study
    
    def cleanup(self):
        """Clean up resources to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear data references
        self.data = None
        
        # Run garbage collection
        gc.collect()
