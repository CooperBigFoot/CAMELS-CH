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
        preprocessing_config: Dict[str, Dict[str, Union[str, List[str], bool]]],
        batch_size: int = 32,
        input_length: int = 365,
        output_length: int = 1,
        num_workers: int = 4,
        features: List[str] = None,
        static_features: List[str] = None,
        target: str = "discharge_spec",
        train_years: int = 15,
        val_years: int = 1,
        min_test_years: int = 1,
        max_missing_pct: float = 0.1,
        max_gap_length: int = 30,
        imputation_gap_size: int = 2,
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
            train_years: Number of years to use for training
            val_years: Number of years to use for validation
            min_test_years: Minimum number of years to use for testing
            max_missing_pct: Maximum percentage of missing values allowed
            max_gap_length: Maximum gap length allowed in data
            imputation_gap_size: Number of consecutive missing values to fill
        """
        super().__init__()

        # Store the input data
        self.time_series_df = time_series_df
        self.static_df = static_df

        # Configuration
        self.preprocessing_config = preprocessing_config
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.num_workers = num_workers

        # Features and target
        self.features = features if features else []
        self.static_features = static_features if static_features else []
        self.target = target

        # Validate features exist in dataframes
        self._validate_features()

        # Will be set up in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Will store preprocessing parameters
        self.scalers = {}

        # The splitting configuration
        self.train_years = train_years
        self.val_years = val_years
        self.min_test_years = min_test_years

        # Data quality parameters
        self.max_missing_pct = max_missing_pct
        self.max_gap_length = max_gap_length
        self.imputation_gap_size = imputation_gap_size
        self.total_years = self.train_years + self.val_years + self.min_test_years
        self.quality_report = None

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
            missing_static = [
                f for f in self.static_features if f not in self.static_df.columns
            ]
            if missing_static:
                raise ValueError(
                    f"Static features {missing_static} not found in static data"
                )

        # Check target
        if self.target not in self.time_series_df.columns:
            raise ValueError(f"Target {self.target} not found in time series data")

    def prepare_data(self):
        """Prepare data with proper temporal splitting and scaling."""
        # Quality check
        required_columns = list(set(self.features + [self.target]))
        filtered_df, quality_report = check_data_quality(
            self.time_series_df,
            required_columns=required_columns,
            max_missing_pct=self.max_missing_pct,
            max_gap_length=self.max_gap_length,
            imputation_gap_size=self.imputation_gap_size,
            total_years=self.total_years,
        )

        if filtered_df.empty:
            raise ValueError(
                "No basins passed quality checks. Check quality_report for details."
            )

        print("\nQuality Check Summary:")
        print(f"Original basins: {quality_report['original_basins']}")
        print(f"Retained basins: {quality_report['retained_basins']}")
        print(f"Excluded basins: {len(quality_report['excluded_basins'])}")

        # Store quality report
        self.quality_report = quality_report

        # Update time series and static data with filtered results
        self.processed_time_series = filtered_df
        self.processed_static = None
        if self.static_df is not None:
            self.processed_static = self.static_df.copy()
            self.processed_static = self.processed_static[
                self.processed_static.index.isin(filtered_df["gauge_id"].unique())
            ]

        # Get training periods from filtered data
        train_data = HydroDataModule.get_training_periods(
            self.processed_time_series, self.train_years
        )

        # Log transforms - apply to full dataset
        if self.preprocessing_config["target"]["log_transform"]:
            self.processed_time_series = apply_log_transform(
                self.processed_time_series, transform_cols=[self.target]
            )
            train_data = apply_log_transform(train_data, transform_cols=[self.target])

        if "log_transform" in self.preprocessing_config["features"]:
            log_features = self.preprocessing_config["features"]["log_transform"]
            if log_features:
                self.processed_time_series = apply_log_transform(
                    self.processed_time_series, transform_cols=log_features
                )
                train_data = apply_log_transform(
                    train_data, transform_cols=log_features
                )

        # Scale features using training data
        if self.features:
            self.processed_time_series, self.scalers["features"] = scale_time_series(
                df_full=self.processed_time_series,
                df_train=train_data,
                features=self.features,
                by_basin=self.preprocessing_config["features"]["scale_method"]
                == "per_basin",
            )

        # Scale static features if any
        if self.static_features:
            self.processed_static, self.scalers["static"] = scale_static_attributes(
                static_df=self.processed_static, attributes=self.static_features
            )

        if self.target in self.features:
            # This happens when running the models autoregressively
            # Reuse the feature scaler for the target, but keep only the target column
            full_scaler = self.scalers["features"]
            self.scalers["target"] = ScalingParameters(
                scalers={self.target: full_scaler.scalers[self.target]},
                feature_names=[self.target],
                gauge_ids=full_scaler.gauge_ids,
            )
        else:
            self.processed_time_series, self.scalers["target"] = scale_time_series(
                df_full=self.processed_time_series,
                df_train=train_data,
                features=[self.target],
                by_basin=self.preprocessing_config["target"]["scale_method"]
                == "per_basin",
            )

        print("Data preprocessing completed with temporal splitting:")
        print(
            f"- Using first {self.train_years} years of each basin for fitting scalers"
        )
        print(
            f"- Features scaled using {self.preprocessing_config['features']['scale_method']} method"
        )
        print(
            f"- Target scaled using {self.preprocessing_config['target']['scale_method']} method"
        )
        if self.static_features:
            print(f"- {len(self.static_features)} static features scaled")
        print(
            f"- Log transforms applied to: {self.preprocessing_config['features'].get('log_transform', [])} and target: {self.preprocessing_config['target']['log_transform']}"
        )

    def setup(self, stage: Optional[str] = None):
        """
        Split data into train/val/test sets and create HydroDataset instances.
        Splits are done chronologically by basin, ensuring each basin has enough years of data.
        """
        # Group data by gauge_id and get time ranges
        basin_groups = self.processed_time_series.groupby("gauge_id")

        # Initialize containers for each split
        train_data = []
        val_data = []
        test_data = []
        excluded_basins = []

        for gauge_id, basin_data in basin_groups:
            # Calculate total years for this basin
            basin_data = basin_data.sort_values("date")
            total_years = (
                basin_data["date"].max() - basin_data["date"].min()
            ).days / 365.25

            # Check if basin has enough data
            required_years = self.train_years + self.val_years + self.min_test_years
            if total_years < required_years:
                excluded_basins.append(gauge_id)
                continue

            # Calculate split dates
            start_date = basin_data["date"].min()
            train_end = start_date + pd.DateOffset(years=self.train_years)
            val_end = train_end + pd.DateOffset(years=self.val_years)

            # Split the data
            train_mask = basin_data["date"] < train_end
            val_mask = (basin_data["date"] >= train_end) & (
                basin_data["date"] < val_end
            )
            test_mask = basin_data["date"] >= val_end

            train_data.append(basin_data[train_mask])
            val_data.append(basin_data[val_mask])
            test_data.append(basin_data[test_mask])

        # Print excluded basins
        if excluded_basins:
            print(f"Excluded {len(excluded_basins)} basins due to insufficient data:")
            print(f"Basin IDs: {', '.join(excluded_basins)}")

        # Combine data for each split
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
            remaining_basins = set(train_df["gauge_id"].unique())
            static_data = self.processed_static[
                self.processed_static.index.isin(remaining_basins)
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
            )

            self.val_dataset = HydroDataset(
                time_series_df=val_df,
                static_df=static_data,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
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
            )

        # Print split info
        print("\nData split summary:")
        print(
            f"Training: {len(train_df)} samples from {len(train_df['gauge_id'].unique())} basins"
        )
        print(
            f"Validation: {len(val_df)} samples from {len(val_df['gauge_id'].unique())} basins"
        )
        print(
            f"Testing: {len(test_df)} samples from {len(test_df['gauge_id'].unique())} basins"
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
            {self.target: predictions.flatten(), "gauge_id": basin_ids}
        )
        # Get target scaler parameters
        target_scalers = self.scalers["target"]

        # Inverse scale the target column
        inv_scaled = inverse_scale_time_series(
            df=df_pred, scaling_params=target_scalers
        )

        # Reverse log transform if needed
        if self.preprocessing_config["target"]["log_transform"]:
            inv_scaled = reverse_log_transform(
                df=inv_scaled, transform_cols=[self.target]
            )

        return inv_scaled[self.target].values

    def get_scalers(self):
        """Return the scalers used for preprocessing"""
        return self.scalers

    def get_preprocessing_config(self):
        """Return the preprocessing configuration"""
        return self.preprocessing_config

    @staticmethod
    def get_training_periods(df: pd.DataFrame, train_years: int) -> pd.DataFrame:
        """Get training periods for all basins based on first train_years."""
        train_data = []

        for gauge_id, basin_data in df.groupby("gauge_id"):
            basin_data = basin_data.sort_values("date")
            start_date = basin_data["date"].min()
            train_end = start_date + pd.DateOffset(years=train_years)
            train_data.append(basin_data[basin_data["date"] < train_end])

        return pd.concat(train_data)
