from typing import Optional, Dict, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HydroTransformer(BaseEstimator, TransformerMixin):
    """Base class for all hydrological transformers."""

    def __init__(self, columns: List[str], **kwargs):
        self.columns = columns
        self.kwargs = kwargs
        self._fitted_state = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HydroTransformer":
        """Fit transformer to specified columns."""
        # Validate columns exist in data
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specified columns."""
        # Validate columns exist in data
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform specified columns."""
        # Validate columns exist in data
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        return X

    def get_feature_names_out(self) -> List[str]:
        """Return feature names after transformation."""
        return self.columns
