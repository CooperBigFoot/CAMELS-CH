import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Union
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.data_models.camels_ch import CamelsCHConfig
from src.data_models.preprocessing import (
    apply_log_transform,
    scale_time_series,
    scale_static_attributes,
    inverse_scale_time_series,
    reverse_log_transform,
    check_data_quality,
    ScalingParameters,
)
from src.data_models.dataset import HydroDataset


class HydroDataModule(pl.LightningDataModule):
    def __init__(
        self,
        time_series_df: pd.DataFrame,
        static_df: pd.DataFrame,
        group_identifier: str,
        preprocessing_config: Dict[str, Dict[str, Union[str, List[str], bool]]],
        batch_size: int = 32,
        input_length: int = 365,
        output_length: int = 1,
        num_workers: int = 4,
        features: List[str] = None,
        static_features: List[str] = None,
        target: str = "discharge_spec",
        min_train_years: int = 10,
        val_years: int = 1,
        test_years: int = 2,
        max_missing_pct: float = 0.1,
        max_gap_length: int = 30,
    ):
        """
        Initialize the HydroDataModule.

        Args:
            time_series_df: DataFrame containing preprocessed time series data
            static_df: DataFrame containing preprocessed static features
            preprocessing_config: Dictionary specifying preprocessing options:
                {
                    "features": {
                        "scale_method": "global" or "per_basin",
                        "log_transform": List of features to log transform
                    },
                    "target": {
                        "scale_method": "global" or "per_basin",
                        "log_transform": bool
                    },
                    "static_features": {
                        "scale_method": "global" or "per_basin"
                    }
                }
            batch_size: Batch size for dataloaders
            input_length: Number of timesteps to use as input
            output_length: Number of timesteps to predict
            num_workers: Number of workers for dataloaders
            features: List of features to use as input
            static_features: List of static features to use
            target: Target variable to predict
            min_train_years: Minimum number of years required for training
            val_years: Number of years to use for validation
            test_years: Number of years to use for testing
            max_missing_pct: Maximum percentage of missing values allowed
            max_gap_length: Maximum gap length allowed in data
        """
        super().__init__()

        self.time_series_df = time_series_df
        self.static_df = static_df
        self.group_identifier = group_identifier
        self.preprocessing_config = preprocessing_config
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.num_workers = num_workers
        self.features = features if features else []
        self.static_features = static_features if static_features else []
        self.target = target
        self.min_train_years = min_train_years
        self.val_years = val_years
        self.test_years = test_years
        self.max_missing_pct = max_missing_pct
        self.max_gap_length = max_gap_length

        # Initialize attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.scalers = {}
        self.quality_report = None
        self.processed_static = None
        self.processed_time_series = None

        # Validate features
        self._validate_features()

    def _validate_features(self):
        """Validate that all specified features exist in the dataframes"""
        # Check time series features
        missing_features = [
            f for f in self.features if f not in self.time_series_df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Features {missing_features} not found in time series data"
            )

        # Check static features
        if self.static_features:

            static_columns = list(self.static_df.columns)

            if self.group_identifier not in static_columns:
                raise ValueError(
                    f"Group identifier {self.group_identifier} not found in static data. Make sure static data includes the [{self.group_identifier}] column."
                )

            missing_static = [
                f
                for f in self.static_features
                if f not in static_columns and f != self.group_identifier
            ]
            if missing_static:
                raise ValueError(
                    f"Static features {missing_static} not found in static data"
                )

        # Check target
        if self.target not in self.time_series_df.columns:
            raise ValueError(f"Target {self.target} not found in time series data")

    def prepare_data(self):
        """Prepare data with fixed validation and test periods."""
        required_columns = list(set(self.features + [self.target]))
        filtered_df, quality_report = check_data_quality(
            self.time_series_df,
            required_columns=required_columns,
            max_missing_pct=self.max_missing_pct,
            max_gap_length=self.max_gap_length,
            min_train_years=self.min_train_years,
            val_years=self.val_years,
            test_years=self.test_years,
            group_identifier=self.group_identifier,
        )

        if filtered_df.empty:
            raise ValueError("No basins passed quality checks.")

        # Store quality report and processed time series
        self.quality_report = quality_report
        self.processed_time_series = filtered_df

        # Process static data if provided
        self.processed_static = None
        if self.static_df is not None and self.static_features:
            retained_basins = filtered_df[self.group_identifier].unique()
            static_filtered = self.static_df[
                self.static_df[self.group_identifier].isin(retained_basins)
            ]
            if not static_filtered.empty:
                self.processed_static, self.scalers["static"] = scale_static_attributes(
                    static_filtered, self.static_features, self.group_identifier
                )

        # Split data first
        train_data, val_data, test_data = [], [], []
        for gauge_id, basin_data in self.processed_time_series.groupby(
            self.group_identifier
        ):
            periods = self.quality_report["valid_periods"][gauge_id]
            valid_end = min(
                period["end"]
                for period in periods.values()
                if period["end"] is not None
            )

            test_start = valid_end - pd.Timedelta(days=int(self.test_years * 365.25))
            val_start = test_start - pd.Timedelta(days=int(self.val_years * 365.25))

            basin_data = basin_data.sort_values("date")
            test_mask = basin_data["date"] >= test_start
            val_mask = (basin_data["date"] >= val_start) & (
                basin_data["date"] < test_start
            )
            train_mask = basin_data["date"] < val_start

            train_data.append(basin_data[train_mask])
            val_data.append(basin_data[val_mask])
            test_data.append(basin_data[test_mask])

        train_df = (
            pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        )

        # Apply log transforms if configured
        if self.preprocessing_config["target"]["log_transform"]:
            self.processed_time_series = apply_log_transform(
                self.processed_time_series,
                transform_cols=[self.target],
                group_identifier=self.group_identifier,
            )

        if "log_transform" in self.preprocessing_config["features"]:
            log_features = self.preprocessing_config["features"]["log_transform"]
            if log_features:
                self.processed_time_series = apply_log_transform(
                    self.processed_time_series,
                    transform_cols=log_features,
                    group_identifier=self.group_identifier,
                )

        # Scale features using only training data
        if self.features:
            self.processed_time_series, self.scalers["features"] = scale_time_series(
                df_full=self.processed_time_series,
                df_train=train_df,
                features=self.features,
                by_basin=self.preprocessing_config["features"]["scale_method"]
                == "per_basin",
                group_identifier=self.group_identifier,
            )

        # Scale target separately if not in features
        if self.target not in self.features:
            self.processed_time_series, self.scalers["target"] = scale_time_series(
                df_full=self.processed_time_series,
                df_train=train_df,
                features=[self.target],
                by_basin=self.preprocessing_config["target"]["scale_method"]
                == "per_basin",
                group_identifier=self.group_identifier,
            )
        else:
            # Use feature scaler for target
            full_scaler = self.scalers["features"]
            self.scalers["target"] = ScalingParameters(
                scalers={self.target: full_scaler.scalers[self.target]},
                feature_names=[self.target],
                gauge_ids=full_scaler.gauge_ids,
            )

    def setup(self, stage: Optional[str] = None):
        """
        Split data into train/val/test sets using valid periods from quality checks.
        Creates HydroDataset instances for each split.

        Args:
            stage: Optional string specifying stage ('fit' or 'test')
        """
        if not hasattr(self, "quality_report"):
            raise RuntimeError("Quality report not found. Did you run prepare_data()?")

        train_data, val_data, test_data = [], [], []

        # Calculate split dates and create datasets for each basin
        for gauge_id, basin_data in self.processed_time_series.groupby(
            self.group_identifier
        ):
            # Get valid period from quality report
            periods = self.quality_report["valid_periods"][gauge_id]
            valid_end = min(
                period["end"]
                for period in periods.values()
                if period["end"] is not None
            )

            # Calculate split dates working backwards
            test_start = valid_end - pd.Timedelta(days=int(self.test_years * 365.25))
            val_start = test_start - pd.Timedelta(days=int(self.val_years * 365.25))

            # Split the data
            basin_data = basin_data.sort_values("date")
            test_mask = basin_data["date"] >= test_start
            val_mask = (basin_data["date"] >= val_start) & (
                basin_data["date"] < test_start
            )
            train_mask = basin_data["date"] < val_start

            train_data.append(basin_data[train_mask])
            val_data.append(basin_data[val_mask])
            test_data.append(basin_data[test_mask])

        # Combine splits
        train_df = (
            pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        )
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = (
            pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        )

        # Handle static data if provided
        static_data = None
        if self.processed_static is not None and not self.processed_static.empty:
            if self.group_identifier not in self.processed_static.columns:
                raise ValueError(
                    f"'{self.group_identifier}' must be a column in processed_static"
                )
            remaining_basins = set(train_df[self.group_identifier].unique())
            static_data = self.processed_static[
                self.processed_static[self.group_identifier].isin(remaining_basins)
            ]

        # Create datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = HydroDataset(
                time_series_df=train_df,
                static_df=static_data,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
            )

            self.val_dataset = HydroDataset(
                time_series_df=val_df,
                static_df=static_data,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
            )

        if stage == "test" or stage is None:
            self.test_dataset = HydroDataset(
                time_series_df=test_df,
                static_df=static_data,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
            )

        # Print split info
        print("\nData split summary:")
        print(
            f"Training: {len(train_df)} samples from {len(train_df[self.group_identifier].unique())} basins"
        )
        print(
            f"Validation: {len(val_df)} samples from {len(val_df[self.group_identifier].unique())} basins"
        )
        print(
            f"Testing: {len(test_df)} samples from {len(test_df[self.group_identifier].unique())} basins"
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training data loader.

        Returns:
            DataLoader configured for training data
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not created. Did you run setup()?")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.

        Returns:
            DataLoader configured for validation data
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not created. Did you run setup()?")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test data loader.

        Returns:
            DataLoader configured for test data
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not created. Did you run setup()?")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        # Build dataframe with target and gauge_id columns
        df_pred = pd.DataFrame(
            {self.target: predictions.flatten(), self.group_identifier: basin_ids}
        )
        # Get target scaler parameters
        target_scalers = self.scalers["target"]

        # Inverse scale the target column
        inv_scaled = inverse_scale_time_series(
            df=df_pred,
            scaling_params=target_scalers,
            group_identifier=self.group_identifier,
        )

        # Reverse log transform if needed
        if self.preprocessing_config["target"]["log_transform"]:
            inv_scaled = reverse_log_transform(
                df=inv_scaled,
                transform_cols=[self.target],
                group_identifier=self.group_identifier,
            )

        return inv_scaled[self.target].values

    def get_scalers(self):
        """Return the scalers used for preprocessing"""
        return self.scalers

    def get_preprocessing_config(self):
        """Return the preprocessing configuration"""
        return self.preprocessing_config

    def get_training_periods(self, df: pd.DataFrame, train_years: int) -> pd.DataFrame:
        """Get training periods for all basins based on first train_years."""
        train_data = []

        for gauge_id, basin_data in df.groupby(self.group_identifier):
            basin_data = basin_data.sort_values("date")
            start_date = basin_data["date"].min()
            train_end = start_date + pd.DateOffset(years=train_years)
            train_data.append(basin_data[basin_data["date"] < train_end])

        return pd.concat(train_data)
