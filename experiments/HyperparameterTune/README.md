# Hydrological Model Hyperparameter Tuning Experiment

This experiment implements a comprehensive hyperparameter tuning framework for hydrological forecasting models, supporting country-specific optimizations for Central Asian basins.

## Experiment Overview

The experiment uses Optuna to efficiently search hyperparameter spaces for four deep learning models (TiDE, TSMixer, EA-LSTM, and TFT), focusing on:

1. **Country-specific optimization**: Tune hyperparameters separately for Tajikistan, Kyrgyzstan, or Combined datasets
2. **Model comparison**: Evaluate multiple model architectures on the same data
3. **Results reporting**: Generate comprehensive reports of optimization results

## Supported Models

The experiment supports four deep learning model architectures:

1. **TiDE**: Time-series Dense Encoder, a densely connected neural network for time series
2. **TSMixer**: Time Series Mixer, an MLP-based architecture inspired by MLP-Mixer
3. **EA-LSTM**: Entity-Aware LSTM, which separates static and dynamic inputs
4. **TFT**: Temporal Fusion Transformer, combining attention mechanisms and RNN components

Each model has a customized hyperparameter search space defined in the `hyperparameter_space` directory.

## Data Description

The experiment uses CARAVAN data for Central Asian catchments, specifically:

- Filtering: Only catchments with low/medium human influence are used
- Features: Meteorological forcings (temperature, precipitation, etc.) and static catchment attributes
- Target: Daily streamflow
- Country-specific datasets can be selected for Tajikistan, Kyrgyzstan, or Combined (both countries)

## Usage

To run the experiment with default settings:

```bash
python experiment.py
```

### Command Line Options

- `--models`: Model types to tune (`tide`, `tsmixer`, `ealstm`, `tft`)
- `--countries`: Countries to tune for (`Tajikistan`, `Kyrgyzstan`, `Combined`)
- `--output-dir`: Output directory for results
- `--n-trials`: Number of optimization trials per model/country combination
- `--seed`: Base random seed for reproducibility
- `--batch-size`: Batch size for training
- `--max-epochs`: Maximum training epochs
- `--timeout`: Timeout for optimization in seconds (None for no timeout)

### Example Commands

Tune TiDE model for Tajikistan with 30 trials:

```bash
python experiment.py --models tide --countries Tajikistan --n-trials 30
```

Tune all models on combined dataset:

```bash
python experiment.py --countries Combined --n-trials 20
```

Tune TSMixer and EA-LSTM for both individual countries:

```bash
python experiment.py --models tsmixer ealstm --countries Tajikistan Kyrgyzstan
```

## Output Structure

The experiment creates the following directory structure:
