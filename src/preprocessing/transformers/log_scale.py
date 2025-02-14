from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from ..base import HydroTransformer


class LogTransformer(HydroTransformer):
    """Applies log1p transform to specified columns, handling negative values.

    For each column, computes y = log1p(x + offset), where offset is computed
    to ensure all values are positive before transformation.

    Args:
        columns: List of column names to transform
        epsilon: Small constant added for numerical stability

    Attributes:
        _fitted_state: Stores offset values per column needed for inverse transform
    """

    def __init__(self, columns: List[str], epsilon: float = 1e-8):
        super().__init__(columns)
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LogTransformer":
        """Compute offset for each column based on minimum values.

        Args:
            X: Input features
            y: Ignored, included for sklearn compatibility

        Returns:
            self

        Raises:
            ValueError: If any specified columns are missing from X
        """
        super().fit(X)

        # Calculate offsets per column to handle negative values
        self._fitted_state["offsets"] = {}
        for col in self.columns:
            min_val = X[col].min()
            # Only add offset if we have negative values
            if min_val < 0:
                self._fitted_state["offsets"][col] = abs(min_val) + self.epsilon
            else:
                self._fitted_state["offsets"][col] = 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transform with computed offsets.

        Args:
            X: Input features

        Returns:
            DataFrame with transformed values

        Raises:
            ValueError: If transform is called before fit or columns are missing
        """
        if not self._fitted_state:
            raise ValueError("Transformer must be fitted before calling transform")

        super().transform(X)
        X_transformed = X.copy()

        for col in self.columns:
            offset = self._fitted_state["offsets"][col]
            X_transformed[col] = np.log1p(X[col] + offset)

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the log transform using stored offsets.

        Args:
            X: Input features

        Returns:
            DataFrame with original scale values

        Raises:
            ValueError: If inverse_transform is called before fit or columns are missing
        """
        if not self._fitted_state:
            raise ValueError(
                "Transformer must be fitted before calling inverse_transform"
            )

        super().inverse_transform(X)
        X_inverse = X.copy()

        for col in self.columns:
            offset = self._fitted_state["offsets"][col]
            X_inverse[col] = np.expm1(X[col]) - offset

        return X_inverse
