# Quantile Mapping Experiment

This experiment evaluates whether using quantile-mapped meteorological forcing data improves hydrological model performance compared to original forcing data. The experiment trains and compares four deep learning models using the same reduced feature set (temperature and precipitation) for both data sources.

## Experiment Overview

Quantile mapping is a bias correction technique that adjusts the distribution of model-simulated variables to match observed distributions. This experiment tests the hypothesis that using quantile-mapped meteorological forcing data will lead to improved hydrological predictions compared to the original forcing data.

The experiment trains models on two different data sources:

1. **Original Data**: Raw meteorological forcing data with a reduced feature set
2. **Quantile-Mapped Data**: Bias-corrected meteorological forcing data using quantile mapping

By comparing model performance across these data sources, we can quantify the benefits of quantile mapping for hydrological modeling.

## Data Description

The experiment uses a reduced feature set from CARAVAN data for catchments:

- **Features**: Only `temperature_2m_mean` and `total_precipitation_sum`
- **Target**: Daily streamflow
- **Filtering**: Only catchments with low/medium human influence are used

### Data Sources

1. **Original Data**: Loaded directly from the CARAVAN dataset with filtering
2. **Quantile-Mapped Data**: Pre-processed CSV files with quantile mapping already applied

## Supported Models

The experiment supports four deep learning model architectures:

1. **TiDE**: Time-series Dense Encoder
2. **TSMixer**: Time Series Mixer
3. **EA-LSTM**: Entity-Aware LSTM
4. **TFT**: Temporal Fusion Transformer

Each model is configured via a YAML file in the `yaml_files` directory.

## Usage

To run the experiment with default settings:

```bash
python experiment.py --quantile-mapped-folder /path/to/quantile/mapped/data
```

### Command Line Options

- `--models`: Model types to evaluate (`tide`, `tsmixer`, `ealstm`, `tft`)
- `--data-sources`: Data sources to use (`original`, `quantile_mapped`)
- `--quantile-mapped-folder`: Path to folder with quantile-mapped data
- `--yaml-dir`: Directory containing model hyperparameter YAML files
- `--output-dir`: Output directory for results
- `--num-runs`: Number of runs for each model
- `--seed`: Base random seed
- `--batch-size`: Batch size for training
- `--max-epochs`: Maximum training epochs
- `--features`: Override default forcing features

Example with custom settings:

```bash
python experiment.py \
  --models tide tsmixer \
  --data-sources original quantile_mapped \
  --quantile-mapped-folder /path/to/quantile/mapped/data \
  --num-runs 3 \
  --batch-size 1024
```

## Output Structure

The experiment creates the following directory structure:

```
output_dir/
├── checkpoints/
│   ├── original/
│   │   ├── tide/
│   │   │   ├── run_0/
│   │   │   └── run_1/
│   │   └── ...
│   └── quantile_mapped/
│       └── ...
├── logs/
│   ├── original/
│   │   ├── tide/
│   │   │   ├── run_0/
│   │   │   └── run_1/
│   │   └── ...
│   └── quantile_mapped/
│       └── ...
└── results/
    ├── original/
    │   ├── all_results.csv
    │   ├── all_results.json
    │   ├── best_model_results.csv
    │   ├── tide_results.csv
    │   └── ...
    └── quantile_mapped/
        └── ...
```

## Results Analysis

The experiment produces several CSV files in the `results` directory that can be used for analysis:

- `all_results.csv`: All training runs for all models
- `best_model_results.csv`: Best run for each model type
- `{model_type}_results.csv`: Results for specific model types

To compare performance between original and quantile-mapped data, analyze the validation losses across data sources.
