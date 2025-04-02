# Global Hydrological Pretraining Experiment

This experiment focuses on pretraining hydrological forecasting models on global data from regions with low to medium human influence. The pretrained models can later be fine-tuned for specific regions like Central Asia.

## Experiment Overview

The experiment trains multiple deep learning models (TSMixer, EA-LSTM, TiDE, TFT) on data from three regions:

- Switzerland (CH)
- Chile (CL)
- USA

Importantly, Central Asian (CA) data is **excluded** from the pretraining to enable later evaluation of transfer learning performance when the models are fine-tuned on CA data.

## Data Description

The experiment uses CARAVAN data for global catchments, specifically:

- **Regions**: Switzerland, Chile, USA
- **Human influence**: Only catchments with low/medium human influence are used
- **Features**: Meteorological forcings (temperature, precipitation, etc.) and static catchment attributes
- **Target**: Daily streamflow

### Data Filtering Criteria

The data is filtered as follows:

1. Only catchments from the specified regions (CH, CL, USA) are included
2. Only catchments with low or medium human influence are used
3. Quality control ensures sufficient data for training/validation/testing

## Supported Models

The experiment supports four deep learning model architectures:

1. **TSMixer**: Time Series Mixer, an MLP-based architecture inspired by MLP-Mixer
2. **EA-LSTM**: Entity-Aware LSTM, which separates static and dynamic inputs
3. **TiDE**: Time-series Dense Encoder, a dense neural network architecture for time series
4. **TFT**: Temporal Fusion Transformer, which combines attention and RNN components

Each model is configured via a YAML file in the `yaml_files` directory.

## Usage

To run the experiment with default settings:

```bash
python experiment.py
```

### Command Line Options

- `--models`: Model types to evaluate (`tsmixer`, `ealstm`, `tide`, `tft`)
- `--yaml-dir`: Directory containing model hyperparameter YAML files
- `--output-dir`: Output directory for results
- `--num-runs`: Number of runs for each model
- `--seed`: Base random seed
- `--batch-size`: Batch size for training
- `--max-epochs`: Maximum training epochs

Example with custom settings:

```bash
python experiment.py --models tsmixer ealstm --num-runs 3 --batch-size 1024
```

## Output Structure

The experiment creates the following directory structure:

```
output_dir/
├── checkpoints/
│   ├── tsmixer/
│   │   ├── run_0/
│   │   └── ...
│   ├── ealstm/
│   │   └── ...
│   └── ...
├── logs/
│   ├── tsmixer/
│   ├── ealstm/
│   └── ...
├── models/
│   ├── tsmixer/
│   ├── ealstm/
│   └── ...
└── results/
    ├── tsmixer/
    │   └── results.csv
    ├── ealstm/
    │   └── results.csv
    └── ...
```

### Key Output Files

- `checkpoints/`: Contains model checkpoints for each run
- `logs/`: TensorBoard logs for training progress
- `models/`: Saved models ready for fine-tuning
- `results/summary.csv`: Summary of all runs
- `results/average_performance.csv`: Average performance metrics per model

## Model Training

Each model is trained using the following process:

1. Data from all regions is combined and preprocessed consistently
2. Proportional time-based splitting into train/validation/test sets
3. Models are trained with early stopping based on validation loss
4. Best checkpoints are saved for later use in fine-tuning experiments
5. Performance metrics are recorded and saved to CSV files

## Fine-tuning Guidelines

The pretrained models saved in the `models/` directory can be used for fine-tuning on Central Asian data. The recommended approach is:

1. Load the pretrained model using the appropriate checkpoint
2. Reduce the learning rate by a factor of 10-100
3. Train on Central Asian data with early stopping

Example code for loading a pretrained model:

```python
from src.models.model_factory import load_pretrained_model

# Load pretrained model with reduced learning rate
model, model_hp = load_pretrained_model(
    model_type="tsmixer",
    yaml_path="path/to/yaml",
    checkpoint_path="path/to/checkpoint",
    lr_factor=10.0,  # Reduce learning rate by factor of 10
)
```
