# Central Asian Data Sharing Experiment

This experiment evaluates the impact of data sharing between Tajikistan and Kyrgyzstan on hydrological model performance. The core hypothesis is that combining data from both countries will lead to improved model performance compared to training on individual country data.

## Experiment Overview

The experiment trains four deep learning models (TiDE, TSMixer, EA-LSTM, and TFT) on three different data scenarios:

1. Tajikistan data only
2. Kyrgyzstan data only
3. Combined data from both countries

By comparing model performance across these scenarios, we can quantify the benefits of data sharing for hydrological modeling in data-sparse regions.

## Data Description

The experiment uses CARAVAN data for Central Asian catchments, specifically:

- Total catchments: Varies based on available data
- Data filtering: Only catchments with low/medium human influence are used
- Features: Meteorological forcings (temperature, precipitation, etc.) and static catchment attributes
- Target: Daily streamflow

### Data Filtering

The data is filtered in several ways:

1. Human influence filtering (only low/medium impact catchments)
2. Country-specific filtering based on the experiment scenario
3. Quality control to ensure sufficient data for training/validation/testing

## Supported Models

The experiment supports four deep learning model architectures:

1. **TiDE**: Time-series Dense Encoder, a dense neural network architecture for time series
2. **TSMixer**: Time Series Mixer, an MLP-based architecture inspired by MLP-Mixer
3. **EA-LSTM**: Entity-Aware LSTM, which separates static and dynamic inputs
4. **TFT**: Temporal Fusion Transformer, which combines attention and RNN components

Each model is configured via a YAML file in the `yaml_files` directory.

## Usage

To run the experiment with default settings:

```bash
python experiment.py
```

### Command Line Options

- `--models`: Model types to evaluate (`tide`, `tsmixer`, `ealstm`, `tft`)
- `--yaml-dir`: Directory containing model hyperparameter YAML files
- `--output-dir`: Output directory for results
- `--num-runs`: Number of runs for each model/country combination
- `--countries`: Countries to include (`Tajikistan`, `Kyrgyzstan`, `Combined`)
- `--seed`: Base random seed
- `--batch-size`: Batch size for training
- `--max-epochs`: Maximum training epochs

Example with custom settings:

```bash
python experiment.py --models tide tsmixer --countries Tajikistan Combined --num-runs 3 --batch-size 1024
```

## Output Structure

The experiment creates the following directory structure:
