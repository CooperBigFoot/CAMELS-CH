from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

# TODO: add group identifier to scaling function to avoid hardcoding "gauge_id"
# Pandas FutureWarning:
pd.set_option("future.no_silent_downcasting", True)


# Function to create a modified pd.concat that uses pyarrow when appropriate
def concat_with_pyarrow(*args, **kwargs) -> pd.DataFrame:
    """Wrapper for pandas concat that uses pyarrow engine when possible."""
    result = pd.concat(*args, **kwargs)
    return result


# Function to create a modified pd.read_csv that uses pyarrow engine
def read_csv_with_pyarrow(path, **kwargs) -> pd.DataFrame:
    """Wrapper for pandas read_csv that always uses pyarrow engine."""
    if 'engine' not in kwargs:
        kwargs['engine'] = 'pyarrow'
    return pd.read_csv(path, **kwargs)


@dataclass
class StaticScalingParameters:
    """Store scalers for static attributes"""

    scaler: StandardScaler
    attribute_names: List[str]


def scale_static_attributes(
    static_df: pd.DataFrame,
    attributes: List[str],
    group_identifier: str = "gauge_id",
) -> Tuple[pd.DataFrame, StaticScalingParameters]:
    """
    Scale static attributes while preserving group identifier relationships.

    Args:
        static_df: DataFrame containing static attributes with a group identifier column
        attributes: List of attribute names to scale
        group_identifier: Column name identifying the grouping variable (e.g., gauge_id, basin_id)

    Returns:
        Tuple of (scaled DataFrame, scaling parameters)

    Raises:
        ValueError: If group_identifier is missing from static_df or if specified attributes are not found
    """
    # Remove group identifier from attributes if present
    scaling_attributes = [attr for attr in attributes if attr != group_identifier]

    # Sort attributes for consistency
    scaling_attributes = sorted(scaling_attributes)

    # Verify group identifier exists
    if group_identifier not in static_df.columns:
        raise ValueError(f"static_df must contain a '{group_identifier}' column")

    # Set group identifier as index to preserve relationships
    df_indexed = static_df.set_index(group_identifier)

    # Verify all attributes exist
    missing_attrs = [
        attr for attr in scaling_attributes if attr not in df_indexed.columns
    ]
    if missing_attrs:
        raise ValueError(f"Attributes {missing_attrs} not found in static_df")

    # Stack values from specified columns
    static_stacked = df_indexed[scaling_attributes].values.reshape(-1, 1)

    # Scale
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(static_stacked)

    # Reshape and create DataFrame with scaled values
    df_static_scaled = pd.DataFrame(
        scaled_values.reshape(df_indexed.shape[0], -1),
        columns=scaling_attributes,
        index=df_indexed.index,
    )

    # Reset index to get group identifier as first column
    df_static_scaled = df_static_scaled.reset_index()

    return df_static_scaled, StaticScalingParameters(
        scaler=scaler, attribute_names=scaling_attributes
    )


def inverse_scale_static_attributes(
    static_scaled: pd.DataFrame,
    scaling_params: StaticScalingParameters,
    group_identifier: str = "gauge_id",
) -> pd.DataFrame:
    """
    Inverse transform scaled static attributes back to their original scale.

    Args:
        static_scaled: DataFrame with scaled static attributes
        scaling_params: ScalingParameters object containing the scaler and attribute names
        group_identifier: Column name identifying the grouping variable (e.g., gauge_id, basin_id)

    Returns:
        DataFrame with inverse-transformed attributes

    Raises:
        ValueError: If required attributes are missing from the input DataFrame
    """
    # Preserve group identifier column
    group_ids = static_scaled[group_identifier].copy()

    # Ensure columns are in same order as during scaling
    static_stacked = static_scaled[scaling_params.attribute_names].values.reshape(-1, 1)

    # Apply inverse transform
    inverse_transformed = scaling_params.scaler.inverse_transform(static_stacked)

    # Create DataFrame with inverse-transformed values
    df_inverse = pd.DataFrame(
        inverse_transformed.reshape(static_scaled.shape[0], -1),
        columns=scaling_params.attribute_names,
    )

    # Add back group identifier
    df_inverse.insert(0, group_identifier, group_ids)

    return df_inverse


@dataclass
class ScalingParameters:
    """Store scalers for each feature and group combination (or global scaling)"""

    scalers: Dict[str, Dict[str, StandardScaler]]
    feature_names: List[str]
    group_ids: List[str]


def scale_time_series(
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    features: List[str],
    by_group: bool = True,
    group_identifier: str = "gauge_id",
) -> Tuple[pd.DataFrame, ScalingParameters]:
    """
    Scale features using only training data for fitting.

    Args:
        df_full: Full DataFrame to scale
        df_train: Training data DataFrame used for fitting scalers
        features: List of feature names to scale
        by_group: Whether to scale separately for each group
        group_identifier: Column name identifying the grouping variable

    Returns:
        Tuple of (scaled DataFrame, scaling parameters)

    Raises:
        ValueError: If zero variance found in features or if group_identifier missing
    """
    if group_identifier not in df_full.columns:
        raise ValueError(f"DataFrame must contain '{group_identifier}' column")

    df_scaled = df_full.copy()
    scalers = {feat: {} for feat in features}

    if by_group:
        for group_id in df_full[group_identifier].unique():
            for feat in features:
                mask = df_full[group_identifier] == group_id
                train_mask = df_train[group_identifier] == group_id

                train_data = df_train.loc[train_mask, feat].astype(float)

                if np.isclose(train_data.var(), 0):
                    raise ValueError(
                        f"Zero variance in {feat} for {group_identifier} {group_id}"
                    )

                sc = StandardScaler()
                sc.fit(train_data.values.reshape(-1, 1))

                scaled_values = sc.transform(
                    df_full.loc[mask, feat].values.reshape(-1, 1)
                )
                df_scaled.loc[mask, feat] = np.round(scaled_values, decimals=3)

                scalers[feat][group_id] = sc
    else:
        for feat in features:
            train_data = df_train[feat].astype(float)

            if np.isclose(train_data.var(), 0):
                raise ValueError(f"Zero variance in feature {feat}")

            sc = StandardScaler()
            sc.fit(train_data.values.reshape(-1, 1))

            scaled_values = sc.transform(df_full[feat].values.reshape(-1, 1))
            df_scaled[feat] = np.round(scaled_values, decimals=3)

            scalers[feat]["global"] = sc

    return df_scaled, ScalingParameters(
        scalers=scalers,
        feature_names=features,
        group_ids=(
            ["global"] if not by_group else df_full[group_identifier].unique().tolist()
        ),
    )


def inverse_scale_time_series(
    df: pd.DataFrame,
    scaling_params: ScalingParameters,
    group_identifier: str = "gauge_id",
) -> pd.DataFrame:
    """
    Inverse transform scaled features, handling group-specific or global scaling.

    Args:
        df: DataFrame with scaled features
        scaling_params: ScalingParameters object with scalers
        group_identifier: Column name identifying the grouping variable

    Returns:
        DataFrame with inverse-transformed features

    Raises:
        ValueError: If group_identifier column is missing or if features cannot be inverse transformed
    """
    if group_identifier not in df.columns:
        raise ValueError(f"DataFrame must contain '{group_identifier}' column")

    df_inverse = df.copy()

    if scaling_params.group_ids == ["global"]:
        for feat in scaling_params.feature_names:
            scaler = scaling_params.scalers[feat]["global"]
            df_inverse[feat] = scaler.inverse_transform(df[[feat]])
    else:
        for group_id in scaling_params.group_ids:
            mask = df[group_identifier] == group_id
            if not mask.any():
                continue
            for feat in scaling_params.feature_names:
                scaler = scaling_params.scalers[feat][group_id]
                df_inverse.loc[mask, feat] = scaler.inverse_transform(
                    df.loc[mask, [feat]]
                )

    return df_inverse


def apply_log_transform(
    df: pd.DataFrame,
    transform_cols: List[str],
    epsilon: float = 1e-8,
    group_identifier: str = "gauge_id",
) -> pd.DataFrame:
    """
    Apply log1p transform to specified columns in a dataframe.

    Args:
        df: Input dataframe containing grouped data
        transform_cols: Column(s) to transform
        epsilon: Small constant to add before log transform
        group_identifier: Column name identifying the grouping variable

    Returns:
        DataFrame with transformed values

    Raises:
        ValueError: If columns not found in dataframe
    """
    if group_identifier not in df.columns:
        raise ValueError(f"DataFrame must contain '{group_identifier}' column")

    df_transformed = df.copy()

    for col in transform_cols:
        if col not in df_transformed.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df_transformed[col] = np.log1p(df_transformed[col] + epsilon)

    return df_transformed


def reverse_log_transform(
    df: pd.DataFrame,
    transform_cols: List[str],
    epsilon: float = 1e-8,
    group_identifier: str = "gauge_id",
) -> pd.DataFrame:
    """
    Reverse log1p transform on specified columns in a dataframe.

    Args:
        df: Input dataframe containing log-transformed data
        transform_cols: Column(s) to reverse transform
        epsilon: Small constant added before log transform
        group_identifier: Column name identifying the grouping variable

    Returns:
        DataFrame with original scale values

    Raises:
        ValueError: If columns not found in dataframe
    """
    if group_identifier not in df.columns:
        raise ValueError(f"DataFrame must contain '{group_identifier}' column")

    df_reversed = df.copy()

    for col in transform_cols:
        if col not in df_reversed.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df_reversed[col] = np.expm1(df_reversed[col]) - epsilon

    return df_reversed


def validate_input(
    df: pd.DataFrame, required_columns: List[str], group_identifier: str = "gauge_id"
) -> None:
    """
    Validate input DataFrame has required columns.

    Args:
        df: Input DataFrame to validate
        required_columns: List of required column names
        group_identifier: Column name identifying the grouping variable

    Raises:
        ValueError: If required columns are missing
    """
    if group_identifier not in df.columns or "date" not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{group_identifier}' and 'date' columns"
        )

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")


def find_valid_data_period(
    series: pd.Series, dates: pd.Series
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the first and last valid (non-NaN) data points in a series. Removes leading and trailing NaNs.

    Args:
        series: The data series to check for valid periods
        dates: The corresponding dates for each value in the series

    Returns:
        Tuple of (start_date, end_date) representing the valid data period.
        If no valid data is found, returns (None, None).

    Raises:
        ValueError: If series and dates have different lengths or dates is not sorted
    """
    # Input validation
    if len(series) != len(dates):
        raise ValueError("Series and dates must have the same length")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order")

    # Find non-NaN values
    valid_mask = ~series.isna()
    valid_indices = np.where(valid_mask)[0]

    # If no valid data found, return None for both dates
    if len(valid_indices) == 0:
        return None, None

    # Get first and last valid indices
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    # Get corresponding dates - use iloc for integer-based indexing
    start_date = dates.iloc[first_valid_idx]
    end_date = dates.iloc[last_valid_idx]

    return start_date, end_date


def check_data_period(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    min_train_years: float = 5,  # Default minimum training period
    val_years: float = None,
    test_years: float = None,
    train_prop: float = None,
    val_prop: float = None,
    test_prop: float = None,
    use_proportional_split: bool = False,
    min_valid_days: int = 365,  # Minimum valid period length in days
) -> Tuple[bool, Optional[str]]:
    """
    Check if period has sufficient data, enforcing minimum training period regardless of splitting method.

    Args:
        start_date: Start date of available data period
        end_date: End date of available data period
        min_train_years: Minimum required years for training (enforced in both methods)
        val_years: Fixed validation period in years (for fixed-year method)
        test_years: Fixed test period in years (for fixed-year method)
        train_prop: Proportion of data for training (for proportional method)
        val_prop: Proportion of data for validation (for proportional method)
        test_prop: Proportion of data for testing (for proportional method)
        use_proportional_split: Whether to use proportional splitting
        min_valid_days: Minimum required length of valid period in days

    Returns:
        Tuple of (meets_requirement, reason)
    """
    if start_date is None or end_date is None:
        return False, "Missing start or end date"

    if not isinstance(start_date, pd.Timestamp) or not isinstance(end_date, pd.Timestamp):
        return False, "Invalid date format"

    if start_date > end_date:
        return False, "Start date is after end date"

    # Calculate total available days
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    if total_days < min_valid_days:
        return False, f"Insufficient data period ({total_days} days, minimum {min_valid_days} required)"
    
    if use_proportional_split:
        # For proportional splits, calculate segment sizes
        train_days = int(total_days * train_prop)
        val_days = int(total_days * val_prop)
        test_days = total_days - (train_days + val_days)
        
        # Convert training days to years for validation against minimum requirement
        train_years = train_days / 365.25
        
        if train_years < min_train_years:
            required_total_years = min_train_years / train_prop
            return False, (f"Insufficient training period ({train_years:.2f} years, minimum {min_train_years} required). "
                           f"Need {required_total_years:.2f} total years with current proportions.")
        
        MIN_SEGMENT_DAYS = 30
        if val_days < MIN_SEGMENT_DAYS:
            return False, f"Validation segment too small ({val_days} days, minimum {MIN_SEGMENT_DAYS} required)"
        if test_days < MIN_SEGMENT_DAYS:
            return False, f"Test segment too small ({test_days} days, minimum {MIN_SEGMENT_DAYS} required)"
    else:
        required_val_test_days = int((val_years + test_years) * 365.25)
        available_train_days = total_days - required_val_test_days
        available_train_years = available_train_days / 365.25

        if available_train_days < 0:
            return False, f"Insufficient data for validation and test periods ({total_days} days available, {required_val_test_days} required)"
        if available_train_years < min_train_years:
            return False, f"Insufficient training data ({available_train_years:.2f} years available, {min_train_years} required)"

    return True, None


def initialize_quality_report(
    df: pd.DataFrame, group_identifier: str = "gauge_id"
) -> Dict:
    """
    Initialize quality report structure.

    Args:
        df: Input DataFrame
        group_identifier: Column name identifying the grouping variable

    Returns:
        Dict with initialized quality report structure
    """
    return {
        "original_basins": len(df[group_identifier].unique()),
        "excluded_basins": {},
        "imputable_gaps": {},
        "date_gaps_filled": {},
        "retained_basins": 0,
    }


def ensure_complete_date_range(
    basin_data: pd.DataFrame,
    group_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    quality_report: Dict,
    group_identifier: str = "gauge_id",
) -> pd.DataFrame:
    """
    Ensure basin data has complete daily date range between valid dates.

    Args:
        basin_data: DataFrame with basin data
        group_id: Group identifier value
        start_date: Start of valid data period
        end_date: End of valid data period
        quality_report: Dictionary to store quality information
        group_identifier: Column name identifying the grouping variable

    Returns:
        DataFrame with complete date range
    """
    if "date_gaps" not in quality_report:
        quality_report["date_gaps"] = {}
    if group_id not in quality_report["date_gaps"]:
        quality_report["date_gaps"][group_id] = {
            "valid_start": start_date,
            "valid_end": end_date,
            "original_dates": len(basin_data),
            "missing_dates": 0,
            "gap_locations": [],
        }

    complete_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    complete_df = pd.DataFrame({"date": complete_dates})
    complete_df[group_identifier] = group_id

    filled_data = pd.merge(
        complete_df, basin_data, on=["date", group_identifier], how="left"
    )

    missing_dates = complete_df.shape[0] - basin_data.shape[0]
    if missing_dates > 0:
        quality_report["date_gaps"][group_id]["missing_dates"] = missing_dates

        existing_dates = set(basin_data["date"])
        missing_dates = sorted(
            [date for date in complete_dates if date not in existing_dates]
        )

        gaps = []
        if missing_dates:
            gap_start = missing_dates[0]
            prev_date = missing_dates[0]

            for date in missing_dates[1:]:
                if (date - prev_date).days > 1:
                    gaps.append(
                        (gap_start.strftime("%Y-%m-%d"), prev_date.strftime("%Y-%m-%d"))
                    )
                    gap_start = date
                prev_date = date

            gaps.append(
                (gap_start.strftime("%Y-%m-%d"), prev_date.strftime("%Y-%m-%d"))
            )

        quality_report["date_gaps"][group_id]["gap_locations"] = gaps

    return filled_data


def check_years_of_data(
    basin_data: pd.DataFrame, gauge_id: str, total_years: int, quality_report: Dict
) -> bool:
    """Check if basin has required years of data.

    Args:
        basin_data: DataFrame with basin data
        gauge_id: Basin identifier
        total_years: Required years of data
        quality_report: Dictionary to store quality information

    Returns:
        bool: True if basin has required years of data, False otherwise
    """
    total_days = (basin_data["date"].max() - basin_data["date"].min()).days
    if total_days < total_years * 365.25:
        quality_report["excluded_basins"][gauge_id] = "insufficient_years"
        return False
    return True


def check_missing_percentage(
    basin_data: pd.DataFrame,
    group_id: str,
    required_columns: List[str],
    max_missing_pct: float,
    quality_report: Dict,
    group_identifier: str = "gauge_id",
) -> bool:
    """
    Check if missing data percentage exceeds threshold.

    Args:
        basin_data: DataFrame with basin data
        group_id: Group identifier value
        required_columns: List of columns to check
        max_missing_pct: Maximum allowed percentage of missing values
        quality_report: Dictionary to store quality information
        group_identifier: Column name identifying the grouping variable

    Returns:
        bool: True if missing percentage checks pass
    """
    if "missing_data" not in quality_report:
        quality_report["missing_data"] = {}
    if group_id not in quality_report["missing_data"]:
        quality_report["missing_data"][group_id] = {
            "columns": {},
            "failure_reason": None,
        }

    failed_columns = []
    for column in required_columns:
        missing_count = basin_data[column].isna().sum()
        total_count = len(basin_data)
        missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0

        quality_report["missing_data"][group_id]["columns"][column] = {
            "missing_count": int(missing_count),
            "total_count": int(total_count),
            "missing_percentage": round(missing_pct, 2),
        }

        if missing_pct > max_missing_pct:
            failed_columns.append({"column": column, "missing_percentage": missing_pct})

    if failed_columns:
        failure_details = [
            f"{fc['column']} ({fc['missing_percentage']:.2f}%)" for fc in failed_columns
        ]
        quality_report["missing_data"][group_id]["failure_reason"] = (
            f"Exceeded maximum missing percentage ({max_missing_pct}%) "
            f"in columns: {', '.join(failure_details)}"
        )
        return False

    return True


def find_gaps(series: pd.Series, max_gap_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Find start and end indices of gaps in time series data.

    Args:
        series: Input time series
        max_gap_length: Maximum allowed gap length

    Returns:
        Tuple of arrays containing gap start and end indices
    """
    # Find missing value runs
    is_missing = series.isna()

    # Edge case: no gaps
    if not is_missing.any():
        return np.array([]), np.array([])

    # Find runs of missing values
    missing_runs = (is_missing != is_missing.shift()).cumsum()[is_missing]

    if len(missing_runs) == 0:
        return np.array([]), np.array([])

    # Get run lengths and start indices
    run_starts = missing_runs.index[missing_runs != missing_runs.shift()]
    run_ends = missing_runs.index[missing_runs != missing_runs.shift(-1)]

    if len(run_starts) == 0:
        return np.array([]), np.array([])

    # Handle case where gap continues to end of series
    if len(run_ends) < len(run_starts):
        run_ends = np.append(run_ends, len(series))

    return run_starts.values, run_ends.values


def check_missing_gaps(
    basin_data: pd.DataFrame,
    group_id: str,
    required_columns: List[str],
    max_gap_length: int,
    quality_report: Dict,
    group_identifier: str = "gauge_id",
) -> bool:
    """
    Check for gaps in data that exceed maximum allowed length.

    Args:
        basin_data: DataFrame with basin data
        group_id: Group identifier value
        required_columns: List of columns to check
        max_gap_length: Maximum allowed gap length in days
        quality_report: Dictionary to store quality information
        group_identifier: Column name identifying the grouping variable

    Returns:
        bool: True if gap checks pass

    Raises:
        ValueError: If date column is missing or invalid
    """
    if "date" not in basin_data.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    if not pd.api.types.is_datetime64_any_dtype(basin_data["date"]):
        raise ValueError("'date' column must be datetime type")
    if not basin_data["date"].is_monotonic_increasing:
        raise ValueError("'date' column must be sorted in ascending order")

    if "gaps" not in quality_report:
        quality_report["gaps"] = {}
    if group_id not in quality_report["gaps"]:
        quality_report["gaps"][group_id] = {"columns": {}, "failure_reason": None}

    failed_columns = []
    for column in required_columns:
        is_missing = basin_data[column].isna()
        gap_starts = is_missing[is_missing & ~is_missing.shift(1).fillna(False)].index
        gap_ends = is_missing[is_missing & ~is_missing.shift(-1).fillna(False)].index

        if len(gap_starts) > len(gap_ends):
            gap_ends = gap_ends.append(pd.Index([is_missing.index[-1]]))
        elif len(gap_ends) > len(gap_starts):
            gap_starts = gap_starts.insert(0, is_missing.index[0])

        if len(gap_starts) > 0 and len(gap_ends) > 0:
            gaps = []
            max_gap = 0
            for start, end in zip(gap_starts, gap_ends):
                try:
                    gap_length = (
                        basin_data.loc[end, "date"] - basin_data.loc[start, "date"]
                    ).days + 1
                    max_gap = max(max_gap, gap_length)

                    if gap_length > max_gap_length:
                        gaps.append(
                            {
                                "start_date": basin_data.loc[start, "date"].strftime(
                                    "%Y-%m-%d"
                                ),
                                "end_date": basin_data.loc[end, "date"].strftime(
                                    "%Y-%m-%d"
                                ),
                                "length": gap_length,
                            }
                        )
                except KeyError:
                    continue

            quality_report["gaps"][group_id]["columns"][column] = {
                "max_gap_length": int(max_gap),
                "number_of_gaps": len(gap_starts),
                "gaps_exceeding_max": gaps,
            }

            if max_gap > max_gap_length:
                failed_columns.append(
                    {"column": column, "max_gap": max_gap, "gaps": gaps}
                )
        else:
            quality_report["gaps"][group_id]["columns"][column] = {
                "max_gap_length": 0,
                "number_of_gaps": 0,
                "gaps_exceeding_max": [],
            }

    if failed_columns:
        failure_details = [
            f"{fc['column']} (max gap: {fc['max_gap']} days)" for fc in failed_columns
        ]
        quality_report["gaps"][group_id]["failure_reason"] = (
            f"Found gaps exceeding maximum length ({max_gap_length} days) "
            f"in columns: {', '.join(failure_details)}"
        )
        return False

    return True


def check_basin_data(
    basin_data: pd.DataFrame,
    gauge_id: str,
    required_columns: List[str],
    max_missing_pct: float,
    max_gap_length: int,
    total_years: int,
    quality_report: Dict,
) -> Optional[pd.DataFrame]:
    """Check data quality for a single basin.

    Args:
        basin_data: DataFrame with basin data
        gauge_id: Basin identifier
        required_columns: List of columns to check for gaps
        max_missing_pct: Maximum allowed missing data percentage
        max_gap_length: Maximum allowed gap length
        total_years: Required years of data
        quality_report: Dictionary to store quality report

    Returns:
        Optional DataFrame with processed basin data, or None if quality checks fail"""
    basin_data = ensure_complete_date_range(basin_data, gauge_id, quality_report)

    if not check_years_of_data(basin_data, gauge_id, total_years, quality_report):
        return None
    if not check_missing_percentage(
        basin_data, gauge_id, required_columns, max_missing_pct, quality_report
    ):
        return None
    if not check_missing_gaps(
        basin_data, gauge_id, required_columns, max_gap_length, quality_report
    ):
        return None
    return basin_data


def check_data_quality(
    df: pd.DataFrame,
    required_columns: List[str],
    max_missing_pct: float,
    max_gap_length: int,
    min_train_years: float,
    val_years: float,
    test_years: float,
    max_imputation_gap_size: int = 5,  # Parameter for imputation threshold
    group_identifier: str = "gauge_id",
    train_prop: float = None,  # Added parameter for train proportion
    val_prop: float = None,    # Added parameter for validation proportion
    test_prop: float = None,   # Added parameter for test proportion
    use_proportional_split: bool = False,  # Added flag for split method
) -> Tuple[pd.DataFrame, Dict]:
    """
    Check data quality, trim leading/trailing NaNs, and impute short gaps.
    
    Args:
        df: DataFrame with data
        required_columns: List of columns to check
        max_missing_pct: Maximum allowed percentage of missing values
        max_gap_length: Maximum allowed gap length in days (for quality reporting only)
        min_train_years: Minimum required years for training
        val_years: Fixed validation period in years
        test_years: Fixed test period in years
        max_imputation_gap_size: Maximum gap length to impute with linear interpolation
        group_identifier: Column name identifying the grouping variable
        train_prop: Proportion of data for training when using proportional splitting
        val_prop: Proportion of data for validation when using proportional splitting
        test_prop: Proportion of data for testing when using proportional splitting
        use_proportional_split: Whether to use proportional splitting method
    
    Returns:
        Tuple of (processed_df, quality_report)
    """
    validate_input(df, required_columns, group_identifier)
    
    quality_report = {
        "original_basins": len(df[group_identifier].unique()),
        "retained_basins": 0,
        "excluded_basins": {},
        "valid_periods": {},
        "imputation_report": {},
        "processing_steps": {},
        "split_method": "proportional" if use_proportional_split else "fixed_years",
    }
    
    # 1. Trim leading and trailing NaNs by finding valid periods for each column in each group
    valid_periods = trim_leading_trailing_nans(df, required_columns, group_identifier)
    quality_report["valid_periods"] = valid_periods
    
    processed_basins = []
    
    for group_id, basin_data in df.groupby(group_identifier):
        basin_data = basin_data.sort_values("date").reset_index(drop=True)
        quality_report["processing_steps"][group_id] = []
        
        # Find overall valid period (overlap of all required columns)
        try:
            overall_start = max(
                period["start"]
                for period in valid_periods[group_id].values()
                if period["start"] is not None
            )
            overall_end = min(
                period["end"]
                for period in valid_periods[group_id].values()
                if period["end"] is not None
            )
        except ValueError:
            quality_report["excluded_basins"][group_id] = "No valid data period found"
            quality_report["processing_steps"][group_id].append(
                "Failed: No valid data period found"
            )
            continue
        
        # Check if period meets minimum length requirements
        meets_requirement, reason = check_data_period(
            overall_start, 
            overall_end, 
            min_train_years=min_train_years, 
            val_years=val_years, 
            test_years=test_years,
            train_prop=train_prop,
            val_prop=val_prop, 
            test_prop=test_prop,
            use_proportional_split=use_proportional_split
        )
        if not meets_requirement:
            quality_report["excluded_basins"][group_id] = reason
            quality_report["processing_steps"][group_id].append(f"Failed: {reason}")
            continue
        
        # 2. Trim basin data to valid period
        trimmed_data = basin_data[(basin_data["date"] >= overall_start) & 
                                 (basin_data["date"] <= overall_end)]
        quality_report["processing_steps"][group_id].append(
            f"Trimmed data from {len(basin_data)} to {len(trimmed_data)} rows"
        )
        
        # 3. Ensure complete date range (fill missing dates)
        complete_data = ensure_complete_date_range(
            trimmed_data,
            group_id,
            overall_start,
            overall_end,
            quality_report,
            group_identifier,
        )
        quality_report["processing_steps"][group_id].append(
            "Completed date range filling"
        )
        
        # 4. Check missing percentage (for reporting only, not for filtering)
        check_missing_percentage(
            complete_data,
            group_id,
            required_columns,
            max_missing_pct,
            quality_report,
            group_identifier,
        )
        quality_report["processing_steps"][group_id].append(
            "Calculated missing percentage"
        )
        
        # 5. Record gap information (for reporting purposes only)
        check_missing_gaps(
            complete_data,
            group_id,
            required_columns,
            max_gap_length,
            quality_report,
            group_identifier,
        )
        quality_report["processing_steps"][group_id].append(
            "Recorded gap information"
        )
        
        # Add to processed basins - we keep all basins now
        processed_basins.append(complete_data)
        quality_report["processing_steps"][group_id].append("Basin processed successfully")
    
    if processed_basins:
        # Combine all processed basin data
        processed_df = pd.concat(processed_basins, ignore_index=True)
        
        # 6. Perform imputation of short gaps
        processed_df, imputation_report = impute_short_gaps(
            processed_df, 
            required_columns,
            max_imputation_gap_size,
            group_identifier
        )
        quality_report["imputation_report"] = imputation_report
        quality_report["retained_basins"] = len(processed_df[group_identifier].unique())
    else:
        processed_df = pd.DataFrame()
        quality_report["retained_basins"] = 0
    
    return processed_df, quality_report


def trim_leading_trailing_nans(
    df: pd.DataFrame, 
    columns: List[str], 
    group_identifier: str = "gauge_id"
) -> Dict[str, Dict[str, Dict[str, Optional[pd.Timestamp]]]]:
    """
    Trim leading and trailing NaNs from each time series by identifying valid data periods.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to check for NaNs
        group_identifier: Column name identifying the grouping variable
    
    Returns:
        Dictionary with valid periods for each group and column
        {group_id: {column: {"start": start_date, "end": end_date}}}
    """
    # Verify input
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    
    valid_periods = {}
    
    # Process each group separately
    for group_id, group_data in df.groupby(group_identifier):
        valid_periods[group_id] = {}
        
        # Sort by date to ensure correct order
        group_data = group_data.sort_values("date")
        dates = group_data["date"]
        
        for column in columns:
            # Find first and last non-NaN values
            start_date, end_date = find_valid_data_period(group_data[column], dates)
            valid_periods[group_id][column] = {"start": start_date, "end": end_date}
    
    return valid_periods


def impute_short_gaps(
    df: pd.DataFrame, 
    columns: List[str], 
    max_imputation_gap_size: int,
    group_identifier: str = "gauge_id"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Linearly impute short gaps (<=max_imputation_gap_size) in time series data.
    Longer gaps remain as NaNs.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to impute
        max_imputation_gap_size: Maximum gap length (in days) to impute
        group_identifier: Column name identifying the grouping variable
    
    Returns:
        Tuple of (DataFrame with imputed values for short gaps, imputation report)
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    
    # Create a copy to avoid modifying the input
    imputed_df = df.copy()
    imputation_report = {}
    
    # Process each group separately
    for group_id, group_data in imputed_df.groupby(group_identifier):
        imputation_report[group_id] = {}
        
        # Sort by date to ensure correct time-based imputation
        group_idx = group_data.index
        
        for column in columns:
            # Current column data
            series = imputed_df.loc[group_idx, column]
            
            # Find gaps (runs of NaNs)
            is_nan = series.isna()
            if not is_nan.any():  # Ensure parentheses are used for method call
                # No NaNs to impute
                imputation_report[group_id][column] = {
                    "short_gaps_count": 0,
                    "long_gaps_count": 0,
                    "imputed_values_count": 0
                }
                continue
                
            # Track consecutive NaN runs
            run_starts = []
            run_lengths = []
            
            in_run = False
            run_start = None
            run_length = 0
            
            # Find all runs of NaNs
            for i, (idx, is_missing) in enumerate(is_nan.items()):
                if is_missing and not in_run:
                    # Start of a new run
                    in_run = True
                    run_start = idx
                    run_length = 1
                elif is_missing and in_run:
                    # Continue current run
                    run_length += 1
                elif not is_missing and in_run:
                    # End of a run
                    run_starts.append(run_start)
                    run_lengths.append(run_length)
                    in_run = False
                    run_start = None
                    run_length = 0
            
            # Handle case where series ends with NaNs
            if in_run:
                run_starts.append(run_start)
                run_lengths.append(run_length)
            
            # Separate short and long gaps
            short_gaps = [(start, length) for start, length in zip(run_starts, run_lengths) 
                          if length <= max_imputation_gap_size]
            long_gaps = [(start, length) for start, length in zip(run_starts, run_lengths) 
                         if length > max_imputation_gap_size]
            
            # Apply linear interpolation to the entire column for the group
            # This will only fill the NaN values where interpolation is possible
            series_imputed = imputed_df.loc[group_idx, column].interpolate(method='linear')
            imputed_df.loc[group_idx, column] = series_imputed
            
            # Record imputation statistics
            imputation_report[group_id][column] = {
                "short_gaps_count": len(short_gaps),
                "long_gaps_count": len(long_gaps),
                "imputed_values_count": sum(length for _, length in short_gaps)
            }
    
    return imputed_df, imputation_report
