from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from ..base import HydroTransformer


class LogTransformer(HydroTransformer):
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon
        self._fitted_state = {}
        self._feature_names = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None
    ) -> "LogTransformer":
        self._fitted_state["offsets"] = {}

        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            for col_idx, col in enumerate(self._feature_names):
                min_val = X[col].min()
                self._fitted_state["offsets"][col_idx] = (
                    abs(min_val) + self.epsilon if min_val < 0 else 0
                )
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            for col_idx in range(X.shape[1]):
                min_val = X[:, col_idx].min()
                self._fitted_state["offsets"][col_idx] = (
                    abs(min_val) + self.epsilon if min_val < 0 else 0
                )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if not self._fitted_state:
            raise ValueError("Transformer must be fitted before calling transform")

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col_idx, col in enumerate(X.columns):
                offset = self._fitted_state["offsets"][col_idx]
                X_transformed[col] = np.log1p(X[col] + offset)
            return X_transformed
        else:
            X_transformed = X.copy()
            for col_idx in range(X.shape[1]):
                offset = self._fitted_state["offsets"][col_idx]
                X_transformed[:, col_idx] = np.log1p(X[:, col_idx] + offset)
            return X_transformed

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if not self._fitted_state:
            raise ValueError(
                "Transformer must be fitted before calling inverse_transform"
            )

        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for col_idx, col in enumerate(X.columns):
                offset = self._fitted_state["offsets"][col_idx]
                X_inverse[col] = np.expm1(X[col]) - offset
            return X_inverse
        else:
            X_inverse = X.copy()
            for col_idx in range(X.shape[1]):
                offset = self._fitted_state["offsets"][col_idx]
                X_inverse[:, col_idx] = np.expm1(X[:, col_idx]) - offset
            return X_inverse

    def get_feature_names_out(self) -> List[str]:
        if self._feature_names is None:
            raise ValueError("Transformer must be fitted before getting feature names")
        return self._feature_names
