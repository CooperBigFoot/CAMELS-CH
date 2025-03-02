from typing import Optional, Dict, Union, List, Tuple
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from src.data_models.dataset import HydroDataset
from src.data_models.preprocessing import check_data_quality
from src.preprocessing.grouped import GroupedTransformer


class HydroDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for hydrological data processing and loading.

    This module handles data preprocessing, including feature scaling and transformations,
    while maintaining proper separation between training, validation, and test data.
    It supports both grouped and non-grouped preprocessing pipelines.
    """

    def __init__(
        self,
        time_series_df: Union[pd.DataFrame, List[pd.DataFrame]],
        static_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        group_identifier: str = "gauge_id",
        preprocessing_config: Dict[
            str, Dict[str, Union[Pipeline, GroupedTransformer]]
        ] = None,
        batch_size: int = 32,
        input_length: int = 365,
        output_length: int = 1,
        num_workers: int = 4,
        features: List[str] = None,
        static_features: List[str] = None,
        target: str = "streamflow",
        domain_id: Union[str, int] = "training",
        min_train_years: int = 10,
        val_years: int = 1,
        test_years: int = 2,
        max_missing_pct: float = 10,
        max_gap_length: int = 30,
    ):
        """Initialize the HydroDataModule with data and configuration parameters."""
        super().__init__()

        # Store configuration parameters
        self.group_identifier = group_identifier
        self.preprocessing_config = preprocessing_config or {}
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.num_workers = num_workers
        self.features = features if features else []
        self.static_features = static_features if static_features else []
        self.target = target
        self.domain_id = domain_id
        self.min_train_years = min_train_years
        self.val_years = val_years
        self.test_years = test_years
        self.max_missing_pct = max_missing_pct
        self.max_gap_length = max_gap_length

        # Process and validate input data
        self.time_series_df, self.static_df = self._process_input_data(
            time_series_df, static_df
        )

        # Initialize storage attributes
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.quality_report = None
        self.processed_static = None
        self.processed_time_series = None
        self.fitted_pipelines = {}

        # Validate configuration
        self._validate_preprocessing_config(preprocessing_config)
        self._validate_features()

    def _validate_preprocessing_config(self, config: Dict) -> None:
        """
        Validate the preprocessing configuration structure and pipeline types.

        This ensures all pipelines are properly configured and compatible with
        the data module's requirements.
        """
        required_keys = ["pipeline"]

        for data_type, cfg in config.items():
            # Check for required configuration keys
            missing = [k for k in required_keys if k not in cfg]
            if missing:
                raise ValueError(
                    f"Missing required keys {missing} in {data_type} config"
                )

            pipeline = cfg["pipeline"]

            # Validate pipeline type
            if not isinstance(pipeline, (Pipeline, GroupedTransformer)):
                raise TypeError(
                    f"Pipeline for {data_type} must be either sklearn.pipeline.Pipeline "
                    f"or GroupedTransformer, got {type(pipeline)}"
                )

            # For grouped transformers, validate group identifier
            if isinstance(pipeline, GroupedTransformer):
                if pipeline.group_identifier != self.group_identifier:
                    raise ValueError(
                        f"GroupedTransformer for {data_type} uses group_identifier "
                        f"'{pipeline.group_identifier}' but data module uses "
                        f"'{self.group_identifier}'"
                    )

            # Validate transformer compatibility
            self._validate_pipeline_compatibility(pipeline)

    def _validate_pipeline_compatibility(
        self, pipeline: Union[Pipeline, GroupedTransformer]
    ) -> None:
        """
        Verify that all transformers in the pipeline implement required methods.

        This ensures each transformer has fit, transform, and inverse_transform methods.
        """
        if isinstance(pipeline, GroupedTransformer):
            pipeline = pipeline.pipeline

        for _, transformer in pipeline.steps:
            required_methods = ["fit", "transform", "inverse_transform"]
            missing = [m for m in required_methods if not hasattr(transformer, m)]
            if missing:
                raise ValueError(
                    f"Transformer {transformer.__class__.__name__} "
                    f"missing required methods: {missing}"
                )

    def _validate_features(self) -> None:
        """
        Validate that all specified features exist in the input data.

        This prevents errors from missing columns during preprocessing.
        """
        missing_features = [
            f for f in self.features if f not in self.time_series_df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Features {missing_features} not found in time series data"
            )

        if self.static_features:
            if self.group_identifier not in self.static_df.columns:
                raise ValueError(
                    f"Group identifier {self.group_identifier} not found in static data"
                )

            missing_static = [
                f
                for f in self.static_features
                if f not in self.static_df.columns and f != self.group_identifier
            ]
            if missing_static:
                raise ValueError(
                    f"Static features {missing_static} not found in static data"
                )

        if self.target not in self.time_series_df.columns:
            raise ValueError(f"Target {self.target} not found in time series data")

    def _process_input_data(
        self,
        time_series_df: Union[pd.DataFrame, List[pd.DataFrame]],
        static_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process and validate input data formats.

        Args:
            time_series_df: Time series DataFrame or list of DataFrames
            static_df: Static attributes DataFrame or list of DataFrames

        Returns:
            Tuple of (processed_time_series_df, processed_static_df)

        Raises:
            ValueError: If input formats are inconsistent
            TypeError: If input types are invalid
        """
        # Check for mismatched input types
        is_time_series_list = isinstance(time_series_df, list)
        is_static_list = isinstance(static_df, list) if static_df is not None else False

        if is_time_series_list != is_static_list and static_df is not None:
            raise ValueError(
                "Input type mismatch: If time_series_df is a list, static_df must also be a list (or None), and vice versa."
            )

        # Process time series DataFrames
        processed_time_series = self._process_dataframe_input(
            time_series_df, "time_series", is_time_series_list
        )

        # Process static DataFrames
        if static_df is not None:
            # Additional check for list length match
            if (
                is_time_series_list
                and is_static_list
                and len(time_series_df) != len(static_df)
            ):
                raise ValueError(
                    f"Length mismatch: static_df list has {len(static_df)} items, "
                    f"but time_series_df list has {len(time_series_df)} items"
                )

            processed_static = self._process_dataframe_input(
                static_df, "static", is_static_list
            )
        else:
            processed_static = None

        return processed_time_series, processed_static

    def _process_dataframe_input(
        self,
        df_input: Union[pd.DataFrame, List[pd.DataFrame]],
        df_type: str,
        is_list: bool,
    ) -> pd.DataFrame:
        """
        Process a DataFrame input which may be a single DataFrame or a list.

        Args:
            df_input: DataFrame or list of DataFrames
            df_type: Type of data ('time_series' or 'static')
            is_list: Whether the input is expected to be a list

        Returns:
            Processed DataFrame
        """
        if is_list:
            # Validate all items are DataFrames
            if not all(isinstance(df, pd.DataFrame) for df in df_input):
                raise TypeError(
                    f"All elements in {df_type} list must be pandas DataFrames"
                )

            # Set attribute for tracking data origin
            for i, df in enumerate(df_input):
                if not hasattr(df, "attrs"):
                    df.attrs = {}
                df.attrs["df_type"] = df_type
                df.attrs["domain_id"] = f"domain_{i}"

            return self._merge_dfs(df_input)
        else:
            if not isinstance(df_input, pd.DataFrame):
                raise TypeError(
                    f"{df_type} must be a pandas DataFrame or a list of DataFrames"
                )
            return df_input

    def _merge_dfs(self, dfs_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames into a single DataFrame.

        Args:
            dfs_list: List of DataFrames to merge

        Returns:
            Merged DataFrame
        """
        # Validate input
        if not dfs_list:
            raise ValueError("Empty list of DataFrames provided")

        # Check if all DataFrames have the necessary columns
        required_cols = [self.group_identifier]
        if "time_series" in dfs_list[0].attrs.get("df_type", ""):
            required_cols.append("date")

        for i, df in enumerate(dfs_list):
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"DataFrame at index {i} missing required columns: {missing_cols}"
                )

        # Merge all DataFrames
        merged_df = pd.concat(dfs_list, ignore_index=True)

        # Handle potential duplicate gauge_ids by adding prefix if needed
        if "time_series" in dfs_list[0].attrs.get("df_type", ""):
            # For time series data, duplicates are fine as long as (gauge_id, date) is unique
            if merged_df.duplicated(subset=[self.group_identifier, "date"]).any():
                # Add domain identifier to gauge_id if provided in DataFrame attributes
                start_idx = 0
                for i, df in enumerate(dfs_list):
                    domain_id = df.attrs.get("domain_id", f"domain_{i}")
                    prefix = f"{domain_id}_"

                    # Only modify gauge_ids in the merged DataFrame that came from this source
                    df_size = len(df)
                    end_idx = start_idx + df_size

                    # Check if these gauge_ids need prefixing (have duplicates)
                    gauge_ids = merged_df.iloc[start_idx:end_idx][
                        self.group_identifier
                    ].unique()

                    # Check for duplicates with other datasets
                    has_duplicates = False
                    for gid in gauge_ids:
                        count = merged_df[
                            merged_df[self.group_identifier] == gid
                        ].shape[0]
                        if count > df[df[self.group_identifier] == gid].shape[0]:
                            has_duplicates = True
                            break

                    if has_duplicates:
                        # Add prefix to these gauge_ids
                        merged_df.iloc[
                            start_idx:end_idx,
                            merged_df.columns.get_loc(self.group_identifier),
                        ] = (
                            prefix
                            + merged_df.iloc[start_idx:end_idx][self.group_identifier]
                        )

                    start_idx = end_idx
        else:
            # For static data, gauge_id should be unique
            if merged_df.duplicated(subset=[self.group_identifier]).any():
                # Add domain identifier to gauge_id if provided in DataFrame attributes
                start_idx = 0
                for i, df in enumerate(dfs_list):
                    domain_id = df.attrs.get("domain_id", f"domain_{i}")
                    prefix = f"{domain_id}_"

                    # Only modify gauge_ids in the merged DataFrame that came from this source
                    df_size = len(df)
                    end_idx = start_idx + df_size

                    # Add prefix to these gauge_ids
                    merged_df.iloc[
                        start_idx:end_idx,
                        merged_df.columns.get_loc(self.group_identifier),
                    ] = (
                        prefix
                        + merged_df.iloc[start_idx:end_idx][self.group_identifier]
                    )

                    start_idx = end_idx

                # Check if there are still duplicates after prefixing
                if merged_df.duplicated(subset=[self.group_identifier]).any():
                    raise ValueError(
                        f"Duplicate {self.group_identifier} found in static DataFrames even after adding prefixes"
                    )

        return merged_df

    def prepare_data(self) -> None:
        """
        Prepare data by performing quality checks and preprocessing.

        This method handles data quality validation, splitting, and transformation
        using the configured preprocessing pipelines.
        """
        # Check data quality
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
            raise ValueError("No basins passed quality checks")

        self.quality_report = quality_report

        print(f"Original basins: {self.quality_report['original_basins']}")
        print(f"Retained basins: {self.quality_report['retained_basins']}")

        self.processed_time_series = filtered_df.copy()

        # Process static data if provided
        if self.static_df is not None and self.static_features:
            retained_basins = filtered_df[self.group_identifier].unique()
            self.processed_static = self.static_df[
                self.static_df[self.group_identifier].isin(retained_basins)
            ]

        # Split data and apply preprocessing
        train_df, val_df, test_df = self._split_data()
        self._apply_preprocessing(train_df)

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.

        Returns:
            Tuple of DataFrames for train, validation, and test sets
        """
        train_data, val_data, test_data = [], [], []

        for gauge_id, basin_data in self.processed_time_series.groupby(
            self.group_identifier
        ):
            periods = self.quality_report["valid_periods"][gauge_id]
            valid_end = min(
                period["end"] for period in periods.values() if period["end"]
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

        return (
            pd.concat(train_data, ignore_index=True),
            pd.concat(val_data, ignore_index=True),
            pd.concat(test_data, ignore_index=True),
        )

    def _apply_preprocessing(self, train_df: pd.DataFrame) -> None:
        """
        Apply preprocessing pipelines to features, target, and static data.

        The pipelines are fitted on training data only to prevent data leakage.
        Fitted pipelines are stored for later use in inverse transformations.
        """
        # Process features
        if "features" in self.preprocessing_config and self.features:
            # Remove target from features list if present
            features_to_process = [f for f in self.features if f != self.target]

            config = self.preprocessing_config["features"]
            pipeline = clone(config["pipeline"])

            # Only pass numeric features to the pipeline, not the group identifier
            # Remove group_identifier
            train_features = train_df[features_to_process]
            all_features = self.processed_time_series[
                features_to_process
            ]  # Remove group_identifier

            # Fit and transform
            pipeline.fit(train_features)
            transformed = pipeline.transform(all_features)

            # Update only the feature columns in processed_time_series
            if isinstance(transformed, np.ndarray):
                for i, col in enumerate(features_to_process):
                    self.processed_time_series[col] = transformed[:, i]
            else:
                for col in features_to_process:
                    self.processed_time_series[col] = transformed[col]

            self.fitted_pipelines["features"] = pipeline

        # Process target - similar change needed
        if "target" in self.preprocessing_config:
            config = self.preprocessing_config["target"]
            pipeline = clone(config["pipeline"])

            if isinstance(pipeline, GroupedTransformer):
                # GroupedTransformer needs the group_identifier
                train_target = train_df[[self.target, self.group_identifier]]
                all_target = self.processed_time_series[
                    [self.target, self.group_identifier]
                ]
            else:
                # Regular pipeline only gets the target column
                train_target = train_df[[self.target]]
                all_target = self.processed_time_series[[self.target]]

            pipeline.fit(train_target)
            transformed = pipeline.transform(all_target)

            # Update target column in processed_time_series
            if isinstance(transformed, np.ndarray):
                self.processed_time_series[self.target] = transformed[:, 0]
            else:
                self.processed_time_series[self.target] = transformed[self.target]

            self.fitted_pipelines["target"] = pipeline

        # Process static features
        if (
            "static_features" in self.preprocessing_config
            and self.processed_static is not None
            and self.static_features
        ):
            config = self.preprocessing_config["static_features"]
            pipeline = clone(config["pipeline"])

            # Filter out group identifier from features to process
            features_to_process = [
                f for f in self.static_features if f != self.group_identifier
            ]

            if isinstance(pipeline, GroupedTransformer):
                static_data = self.processed_static[
                    features_to_process + [self.group_identifier]
                ]
                pipeline.fit(static_data)
                transformed = pipeline.transform(static_data)

                # Update static features columns
                for col in features_to_process:
                    self.processed_static.loc[:, col] = transformed[col]
            else:
                static_data = self.processed_static[features_to_process]
                pipeline.fit(static_data)
                transformed = pipeline.transform(static_data)

                # Handle case where transform returns numpy array
                if isinstance(transformed, np.ndarray):
                    for i, col in enumerate(features_to_process):
                        self.processed_static.loc[:, col] = transformed[:, i]
                else:
                    # Handle DataFrame/Series case
                    for col in features_to_process:
                        self.processed_static.loc[:, col] = transformed[col]

            self.fitted_pipelines["static"] = pipeline

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets for training, validation, and testing.

        This method is called by PyTorch Lightning to set up datasets
        for each stage of training.
        """
        if not hasattr(self, "quality_report"):
            raise RuntimeError("Quality report not found. Did you run prepare_data()?")

        # Split data
        train_data, val_data, test_data = self._split_data()

        # Create datasets based on stage
        if stage == "fit" or stage is None:
            self._train_dataset = HydroDataset(
                time_series_df=train_data,
                static_df=self.processed_static,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
                domain_id=self.domain_id,
            )

            self._val_dataset = HydroDataset(
                time_series_df=val_data,
                static_df=self.processed_static,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
                domain_id=self.domain_id,
            )

        if stage == "test" or stage is None:
            self._test_dataset = HydroDataset(
                time_series_df=test_data,
                static_df=self.processed_static,
                input_length=self.input_length,
                output_length=self.output_length,
                features=self.features,
                target=self.target,
                static_features=self.static_features,
                group_identifier=self.group_identifier,
                domain_id=self.domain_id,
            )

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        basin_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.

        Args:
            predictions: Model predictions to inverse transform
            basin_ids: Associated basin IDs for the predictions

        Returns:
            Array of inverse-transformed predictions
        """
        if "target" not in self.fitted_pipelines:
            raise ValueError("Target pipeline not found - did you run prepare_data()?")

        # Ensure predictions and basin_ids are numpy arrays
        predictions = np.asarray(predictions)
        basin_ids = np.asarray(basin_ids)

        # Create DataFrame with target and group identifier columns
        df_pred = pd.DataFrame(
            {self.group_identifier: basin_ids, self.target: predictions.flatten()}
        )

        # Get unique basin IDs to validate
        unique_basins = np.unique(basin_ids)
        missing_basins = []
        for basin in unique_basins:
            if basin not in self.fitted_pipelines["target"].fitted_pipelines:
                missing_basins.append(basin)

        if missing_basins:
            raise ValueError(f"No fitted pipeline found for basins: {missing_basins}")

        # Use fitted pipeline for inverse transform
        target_pipeline = self.fitted_pipelines["target"]
        df_inverse = target_pipeline.inverse_transform(df_pred)

        return df_inverse[self.target].values

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def save_preprocessing_report(self) -> None:
        """Save data quality report and preprocessing configuration."""
        self.preprocessing_state = {
            "quality_report": self.quality_report,
            "config": self.preprocessing_config,
        }

    # Protected properties to access datasets. I do not want to expose them (shitty stuff happens if I do)
    @property
    def train_dataset(self) -> HydroDataset:
        """Get the training dataset."""
        return self._train_dataset

    @property
    def val_dataset(self) -> HydroDataset:
        """Get the validation dataset."""
        return self._val_dataset

    @property
    def test_dataset(self) -> HydroDataset:
        """Get the test dataset."""
        return self._test_dataset


class HydroTransferDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for transfer learning in hydrological modeling.
    This module combines source and target data modules for training, validation,
    and testing.
    """

    def __init__(
        self,
        source_datamodule: HydroDataModule,
        target_datamodule: HydroDataModule,
        num_workers: int = 4,
        mode: str = "min_size",
    ):
        super().__init__()
        self.source_dm = source_datamodule
        self.target_dm = target_datamodule
        self.num_workers = num_workers
        self.mode = mode

        # Validate datasets
        self._validate_datasets()

    def _validate_datasets(self):
        """Ensure datasets have matching features and domains"""
        if self.source_dm.features != self.target_dm.features:
            raise ValueError("Source and target features must match")

        if self.source_dm.static_features != self.target_dm.static_features:
            raise ValueError("Static features must match between domains")

    def _create_combined_loader(self, stage: str) -> CombinedLoader:
        """Create combined loader for either train/val/test"""
        loaders = {
            "source": getattr(self.source_dm, f"{stage}_dataloader")(),
            "target": getattr(self.target_dm, f"{stage}_dataloader")(),
        }
        return CombinedLoader(loaders, mode=self.mode)

    def train_dataloader(self) -> CombinedLoader:
        return self._create_combined_loader("train")

    def val_dataloader(self) -> CombinedLoader:
        return self._create_combined_loader("val")

    def test_dataloader(self) -> CombinedLoader:
        return self._create_combined_loader("test")
