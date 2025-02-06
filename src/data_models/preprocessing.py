from typing import Union, List, Tuple, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class StaticScalingParameters:
    """Store scalers for static attributes"""

    scaler: StandardScaler
    attribute_names: List[str]


def scale_static_attributes(
    static_df: pd.DataFrame,
    attributes: List[str],
) -> Tuple[pd.DataFrame, StaticScalingParameters]:
    # Sort attributes alphabetically
    attributes = sorted(attributes)

    # Stack values from sorted columns
    static_stacked = static_df[attributes].values.reshape(-1, 1)

    # Scale
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(static_stacked)

    # Reshape and create DataFrame with sorted columns
    df_static_scaled = pd.DataFrame(
        scaled_values.reshape(static_df.shape[0], -1), columns=attributes
    )

    return df_static_scaled, StaticScalingParameters(
        scaler=scaler, attribute_names=attributes
    )


def inverse_scale_static_attributes(
    static_scaled: pd.DataFrame,
    scaling_params: StaticScalingParameters,
) -> pd.DataFrame:
    # Ensure columns are in same order as during scaling
    static_stacked = static_scaled[scaling_params.attribute_names].values.reshape(-1, 1)

    inverse_transformed = scaling_params.scaler.inverse_transform(static_stacked)

    return pd.DataFrame(
        inverse_transformed.reshape(static_scaled.shape[0], -1),
        columns=scaling_params.attribute_names,
    )


@dataclass
class ScalingParameters:
    """Store scalers for each feature and basin combination (or global scaling)"""

    scalers: Dict[str, Dict[str, StandardScaler]]
    feature_names: List[str]
    gauge_ids: List[str]


def scale_time_series(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    by_basin: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, ScalingParameters]:
    """
    Scale features using z-score standardization. If by_basin is True, scale each feature
    separately for each basin (gauge_id); otherwise, scale globally.

    Args:
        df_train: Training data
        df_test: Test data
        features: List of features to scale
        by_basin: Whether to scale features by basin

    Returns:
        Tuple of scaled training and test dataframes, and ScalingParameters object
    """
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    scalers = {feat: {} for feat in features}

    if by_basin:
        gauge_ids = df_train["gauge_id"].unique().tolist()
        for gauge_id in gauge_ids:
            train_mask = df_train["gauge_id"] == gauge_id
            test_mask = df_test["gauge_id"] == gauge_id
            for feat in features:
                sc = StandardScaler()
                df_train_scaled.loc[train_mask, feat] = sc.fit_transform(
                    df_train.loc[train_mask, [feat]]
                )
                if test_mask.any():
                    df_test_scaled.loc[test_mask, feat] = sc.transform(
                        df_test.loc[test_mask, [feat]]
                    )
                scalers[feat][gauge_id] = sc
    else:
        gauge_ids = ["global"]
        for feat in features:
            sc = StandardScaler()
            df_train_scaled[feat] = sc.fit_transform(df_train[[feat]])
            df_test_scaled[feat] = sc.transform(df_test[[feat]])
            scalers[feat]["global"] = sc

    scaling_params = ScalingParameters(
        scalers=scalers, feature_names=features, gauge_ids=gauge_ids
    )
    return df_train_scaled, df_test_scaled, scaling_params


def inverse_scale_time_series(
    df: pd.DataFrame, scaling_params: ScalingParameters
) -> pd.DataFrame:
    """
    Inverse transform scaled features, handling basin-specific or global scaling.

    Args:
        df: DataFrame with scaled features
        scaling_params: ScalingParameters object with scalers

    Returns:
        DataFrame with inverse-transformed features
    """
    df_inverse = df.copy()
    if scaling_params.gauge_ids == ["global"]:
        for feat in scaling_params.feature_names:
            scaler = scaling_params.scalers[feat]["global"]
            df_inverse[feat] = scaler.inverse_transform(df[[feat]])
    else:
        for gauge_id in scaling_params.gauge_ids:
            mask = df["gauge_id"] == gauge_id
            if not mask.any():
                continue
            for feat in scaling_params.feature_names:
                scaler = scaling_params.scalers[feat][gauge_id]
                df_inverse.loc[mask, feat] = scaler.inverse_transform(
                    df.loc[mask, [feat]]
                )
    return df_inverse


def apply_log_transform(
    df: pd.DataFrame, transform_cols: List[str], epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Apply log1p transform to specified columns in a dataframe.

    Args:
        df: Input dataframe containing data for multiple basins
        transform_cols: Column(s) to transform
        epsilon: Small constant to add before log transform to handle zeros

    Returns:
        DataFrame with transformed values
    """
    df_transformed = df.copy()

    # Apply log1p transform to specified columns
    for col in transform_cols:
        if col not in df_transformed.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df_transformed[col] = np.log1p(df_transformed[col] + epsilon)

    return df_transformed


def reverse_log_transform(
    df: pd.DataFrame, transform_cols: List[str], epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Reverse log1p transform on specified columns in a dataframe.

    Args:
        df: Input dataframe containing log-transformed data
        transform_cols: Column(s) to reverse transform
        epsilon: Small constant that was added before log transform

    Returns:
        DataFrame with original scale values
    """
    df_reversed = df.copy()

    # Apply reverse transform to specified columns
    for col in transform_cols:
        if col not in df_reversed.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df_reversed[col] = np.expm1(df_reversed[col]) - epsilon

    return df_reversed


def train_validate_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into training and validation sets.

    Args:
        df: Input dataframe
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of training and validation dataframes
    """
    train_size = int(len(df) * train_ratio)
    df_train = df[:train_size]
    df_val = df[train_size:]
    return df_train, df_val
