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
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    features: List[str],
    by_basin: bool = True,
) -> Tuple[pd.DataFrame, ScalingParameters]:
    """
    Scale features using z-score standardization. If by_basin is True, scale each feature
    separately for each basin (gauge_id); otherwise, scale globally.

    Args:
        df_full: DataFrame with time series data for multiple basins
        df_train: DataFrame with training data
        features: List of features to scale
        by_basin: If True, scale features separately for each

    Returns:
        Tuple of scaled training and test dataframes, and ScalingParameters object
    """
    df_scaled = df_full.copy()
    scalers = {feat: {} for feat in features}

    if by_basin:
        for gauge_id in df_full["gauge_id"].unique():
            for feat in features:
                mask = df_full["gauge_id"] == gauge_id
                train_mask = df_train["gauge_id"] == gauge_id

                sc = StandardScaler()
                sc.fit(df_train.loc[train_mask, [feat]])
                df_scaled.loc[mask, feat] = sc.transform(df_full.loc[mask, [feat]])
                scalers[feat][gauge_id] = sc
    else:
        for feat in features:
            sc = StandardScaler()
            sc.fit(df_train[[feat]])
            df_scaled[feat] = sc.transform(df_full[[feat]])
            scalers[feat]["global"] = sc

    return df_scaled, ScalingParameters(
        scalers=scalers,
        feature_names=features,
        gauge_ids=["global"] if not by_basin else df_full["gauge_id"].unique().tolist(),
    )


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


from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def validate_input(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate input DataFrame has required columns."""
    if "gauge_id" not in df.columns or "date" not in df.columns:
        raise ValueError("DataFrame must contain 'gauge_id' and 'date' columns")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")


def initialize_quality_report(df: pd.DataFrame) -> Dict:
    """Initialize quality report structure."""
    return {
        "original_basins": len(df["gauge_id"].unique()),
        "excluded_basins": {},
        "imputable_gaps": {},
        "date_gaps_filled": {},
        "retained_basins": 0,
    }


def ensure_complete_date_range(
    basin_data: pd.DataFrame, gauge_id: str, quality_report: Dict
) -> pd.DataFrame:
    """Ensure basin data has complete daily date range."""
    min_date = basin_data["date"].min()
    max_date = basin_data["date"].max()
    complete_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    complete_df = pd.DataFrame({"date": complete_dates})
    complete_df["gauge_id"] = gauge_id

    basin_data = pd.merge(complete_df, basin_data, on=["date", "gauge_id"], how="left")

    n_added_dates = len(complete_dates) - len(basin_data)
    if n_added_dates > 0:
        quality_report["date_gaps_filled"][gauge_id] = n_added_dates

    return basin_data


def check_years_of_data(
    basin_data: pd.DataFrame, gauge_id: str, total_years: int, quality_report: Dict
) -> bool:
    """Check if basin has required years of data."""
    total_days = (basin_data["date"].max() - basin_data["date"].min()).days
    if total_days < total_years * 365.25:
        quality_report["excluded_basins"][gauge_id] = "insufficient_years"
        return False
    return True


def check_missing_percentage(
    basin_data: pd.DataFrame,
    gauge_id: str,
    required_columns: List[str],
    max_missing_pct: float,
    quality_report: Dict,
) -> bool:
    """Check if missing data percentage exceeds threshold."""
    missing_pcts = basin_data[required_columns].isna().mean()
    if any(missing_pcts > max_missing_pct):
        failed_cols = missing_pcts[missing_pcts > max_missing_pct].index.tolist()
        quality_report["excluded_basins"][gauge_id] = f"high_missing_pct:{failed_cols}"
        return False
    return True


def check_missing_gaps(
    basin_data: pd.DataFrame,
    gauge_id: str,
    required_columns: List[str],
    max_gap_length: int,
    imputation_gap_size: int,
    quality_report: Dict,
) -> bool:
    """Check consecutive missing value gaps."""
    for col in required_columns:
        gaps = basin_data[col].isna()
        if not gaps.any():
            continue

        gap_starts = np.where(gaps.values[1:] & ~gaps.values[:-1])[0] + 1
        gap_ends = np.where(~gaps.values[1:] & gaps.values[:-1])[0] + 1

        if len(gap_starts) > 0:
            if gap_starts[0] == 0:
                gap_starts = np.concatenate([[0], gap_starts])
            if gap_ends[-1] == len(gaps):
                gap_ends = np.concatenate([gap_ends, [len(gaps)]])

            gap_lengths = gap_ends - gap_starts

            imputable_gaps = sum(gap_lengths <= imputation_gap_size)
            if imputable_gaps > 0:
                if gauge_id not in quality_report["imputable_gaps"]:
                    quality_report["imputable_gaps"][gauge_id] = {}
                quality_report["imputable_gaps"][gauge_id][col] = imputable_gaps

            if any(gap_lengths > max_gap_length):
                quality_report["excluded_basins"][gauge_id] = f"long_gaps:{col}"
                return False
    return True


def check_basin_data(
    basin_data: pd.DataFrame,
    gauge_id: str,
    required_columns: List[str],
    max_missing_pct: float,
    max_gap_length: int,
    imputation_gap_size: int,
    total_years: int,
    quality_report: Dict,
) -> Optional[pd.DataFrame]:
    """Process single basin data through all quality checks."""
    basin_data = ensure_complete_date_range(basin_data, gauge_id, quality_report)

    if not check_years_of_data(basin_data, gauge_id, total_years, quality_report):
        return None

    if not check_missing_percentage(
        basin_data, gauge_id, required_columns, max_missing_pct, quality_report
    ):
        return None

    if not check_missing_gaps(
        basin_data,
        gauge_id,
        required_columns,
        max_gap_length,
        imputation_gap_size,
        quality_report,
    ):
        return None

    return basin_data


def check_data_quality(
    df: pd.DataFrame,
    required_columns: List[str],
    max_missing_pct: float = 0.1,
    max_gap_length: int = 30,
    imputation_gap_size: int = 2,
    total_years: int = 30,
) -> Tuple[pd.DataFrame, Dict]:
    """Main function to check data quality for all basins."""
    validate_input(df, required_columns)
    quality_report = initialize_quality_report(df)

    filtered_basins = []
    for gauge_id, basin_data in df.groupby("gauge_id"):
        processed_data = check_basin_data(
            basin_data,
            gauge_id,
            required_columns,
            max_missing_pct,
            max_gap_length,
            imputation_gap_size,
            total_years,
            quality_report,
        )
        if processed_data is not None:
            filtered_basins.append(processed_data)

    if not filtered_basins:
        return pd.DataFrame(), quality_report

    filtered_df = pd.concat(filtered_basins, ignore_index=True)
    quality_report["retained_basins"] = len(filtered_df["gauge_id"].unique())

    return filtered_df, quality_report
