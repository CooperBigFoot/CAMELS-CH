# Proportional Splitting in Hydrological Time Series Experiments

## Overview

This document explains the proportional data splitting approach implemented in the hydrological modeling experiments. The system now uses a fixed proportion for dividing available data between training, validation, and test sets instead of using fixed time periods.

## Implementation Details

### Proportional Split Parameters

The proportions used for data splitting are:

- **Training Set**: 50% of available data
- **Validation Set**: 25% of available data
- **Test Set**: 25% of available data

These proportions are configurable in the `ExperimentConfig` class using the following parameters:

```python
USE_PROPORTIONAL_SPLIT: bool = True  # Enable proportional splitting
TRAIN_PROP: float = 0.5              # 50% of data for training
VAL_PROP: float = 0.25               # 25% of data for validation
TEST_PROP: float = 0.25              # 25% of data for testing
```

### Benefits of Proportional Splitting

1. **Efficient data utilization**: Makes better use of available data across different catchments
2. **Consistency across regions**: Maintains the same relative data allocation regardless of data availability
3. **Better comparability**: Results from different basins are more comparable when using the same proportional split
4. **Adaptability**: Automatically adapts to basins with different record lengths

### Legacy Parameters

The system maintains backward compatibility with the following parameters (used only if `USE_PROPORTIONAL_SPLIT=False`):

- `MIN_TRAIN_YEARS`: Minimum number of years required for training
- `VAL_YEARS`: Number of years allocated for validation
- `TEST_YEARS`: Number of years allocated for testing

## Usage

When instantiating a `HydroDataModule`, the appropriate splitting method is selected based on the `use_proportional_split` parameter:

```python
data_module = HydroDataModule(
    # Other parameters...
    use_proportional_split=config.USE_PROPORTIONAL_SPLIT,
    train_prop=config.TRAIN_PROP,
    val_prop=config.VAL_PROP,
    test_prop=config.TEST_PROP,
    # Legacy parameters
    min_train_years=config.CA_CONFIG["MIN_TRAIN_YEARS"],
    val_years=config.CA_CONFIG["VAL_YEARS"],
    test_years=config.CA_CONFIG["TEST_YEARS"],
)
```

## Monitoring and Verification

The system logs dataset sizes for verification:

- Training set size
- Validation set size
- Test set size

These statistics are stored as trial attributes in Optuna experiments and included in the final results CSV.

## Best Practices

1. Verify that the total proportion equals 1.0 (100% of valid data)
2. Monitor the absolute sizes of splits to ensure adequate data in each segment
3. Consider minimum sequence requirements when working with basins having very short periods of record
