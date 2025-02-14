from typing import Dict, List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """Wrapper to apply any sklearn Pipeline by group.

    Args:
        pipeline: sklearn Pipeline to apply
        columns: Columns to transform
        group_identifier: Column name to group by
    """

    def __init__(self, pipeline: Pipeline, columns: List[str], group_identifier: str):
        self.pipeline = pipeline
        self.columns = columns
        self.group_identifier = group_identifier
        self.fitted_pipelines: Dict[str, Pipeline] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "GroupedTransformer":
        """Fit a separate pipeline for each group."""
        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        for group in X[self.group_identifier].unique():
            group_mask = X[self.group_identifier] == group
            group_data = X.loc[group_mask, self.columns].copy()

            # Create a fresh copy of the pipeline for this group
            group_pipeline = Pipeline(self.pipeline.steps)
            group_pipeline.fit(group_data)
            self.fitted_pipelines[group] = group_pipeline

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform each group with its fitted pipeline."""
        X_transformed = X.copy()

        for group in X[self.group_identifier].unique():
            if group not in self.fitted_pipelines:
                continue

            group_mask = X[self.group_identifier] == group
            group_data = X.loc[group_mask, self.columns].copy()

            # Transform this group's data
            X_transformed.loc[group_mask, self.columns] = self.fitted_pipelines[
                group
            ].transform(group_data)

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform each group with its fitted pipeline."""
        X_inverse = X.copy()

        for group in X[self.group_identifier].unique():
            if group not in self.fitted_pipelines:
                continue

            group_mask = X[self.group_identifier] == group
            group_data = X.loc[group_mask, self.columns].copy()

            # Inverse transform this group's data
            X_inverse.loc[group_mask, self.columns] = self.fitted_pipelines[
                group
            ].inverse_transform(group_data)

        return X_inverse

    def get_feature_names_out(self) -> List[str]:
        """Return feature names after transformation."""
        return self.columns
