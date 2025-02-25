from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import copy


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """Applies transformations by group (e.g., by catchment).

    This transformer fits and applies a separate pipeline for each unique value in
    the group_identifier column. Useful for hydrological data where different
    catchments might require different preprocessing.

    Attributes:
        pipeline: sklearn Pipeline to apply to each group
        columns: Columns to transform
        group_identifier: Column name to group by
        fitted_pipelines: Dictionary mapping group values to fitted pipelines
    """

    def __init__(
        self,
        pipeline: Pipeline,
        columns: List[str],
        group_identifier: str
    ):
        """Initialize GroupedTransformer.

        Args:
            pipeline: sklearn Pipeline to apply
            columns: Columns to transform
            group_identifier: Column name to group by
        """
        self.pipeline = pipeline
        self.columns = columns
        self.group_identifier = group_identifier
        self.fitted_pipelines: Dict[Union[str, int], Pipeline] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "GroupedTransformer":
        """Fit a separate pipeline for each group.

        Args:
            X: Input data with group_identifier column
            y: Target variable (optional, passed to pipeline fit method)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If columns or group_identifier not found in data
        """
        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        # Store all unique groups to ensure we can handle new/unseen groups
        # during transform/inverse_transform
        self.all_groups = X[self.group_identifier].unique().tolist()

        # Fit a separate pipeline for each group
        for group in self.all_groups:
            group_mask = X[self.group_identifier] == group
            if not group_mask.any():
                continue

            group_data = X.loc[group_mask, self.columns].copy()
            if y is not None:
                group_y = y.loc[group_mask] if hasattr(
                    y, 'loc') else y[group_mask]
            else:
                group_y = None

            # Create a fresh copy of the pipeline for this group
            group_pipeline = copy.deepcopy(self.pipeline)
            group_pipeline.fit(group_data, group_y)
            self.fitted_pipelines[group] = group_pipeline

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform each group with its fitted pipeline.

        Args:
            X: Input data with group_identifier column

        Returns:
            Transformed data

        Notes:
            Groups not seen during fit will be passed through unchanged
        """
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        X_transformed = X.copy()

        # Process each group separately
        for group in X[self.group_identifier].unique():
            group_mask = X[self.group_identifier] == group
            if not group_mask.any():
                continue

            if group not in self.fitted_pipelines:
                print(
                    f"Warning: Group {group} not seen during fit, passing through unchanged")
                continue

            group_data = X.loc[group_mask, self.columns].copy()

            # Transform this group's data
            transformed_data = self.fitted_pipelines[group].transform(
                group_data)

            # Handle the case where pipeline returns ndarray instead of DataFrame
            if isinstance(transformed_data, np.ndarray):
                for i, col in enumerate(self.columns):
                    X_transformed.loc[group_mask, col] = transformed_data[:, i]
            else:
                X_transformed.loc[group_mask, self.columns] = transformed_data

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform each group with its fitted pipeline.

        Args:
            X: Input data with group_identifier column

        Returns:
            Inverse transformed data

        Notes:
            Groups not seen during fit will be passed through unchanged
        """
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        X_inverse = X.copy()

        # Process each group separately
        for group in X[self.group_identifier].unique():
            group_mask = X[self.group_identifier] == group
            if not group_mask.any():
                continue

            if group not in self.fitted_pipelines:
                print(
                    f"Warning: Group {group} not seen during fit, passing through unchanged")
                continue

            group_data = X.loc[group_mask, self.columns].copy()

            # Check if pipeline has inverse_transform method
            if not hasattr(self.fitted_pipelines[group], 'inverse_transform'):
                print(
                    f"Warning: Pipeline for group {group} does not support inverse_transform")
                continue

            # Inverse transform this group's data
            inverse_data = self.fitted_pipelines[group].inverse_transform(
                group_data)

            # Handle the case where pipeline returns ndarray instead of DataFrame
            if isinstance(inverse_data, np.ndarray):
                for i, col in enumerate(self.columns):
                    X_inverse.loc[group_mask, col] = inverse_data[:, i]
            else:
                X_inverse.loc[group_mask, self.columns] = inverse_data

        return X_inverse

    def get_feature_names_out(self) -> List[str]:
        """Return feature names after transformation.

        Returns:
            List of output feature names
        """
        if not self.fitted_pipelines:
            raise ValueError(
                "Transformer must be fitted before getting feature names")

        # Return the input columns as output feature names, assuming the pipeline
        # doesn't change the feature names
        return self.columns.copy()
