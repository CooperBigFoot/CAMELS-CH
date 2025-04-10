# Fine-Tuning Experiment

This experiment focuses on fine-tuning pre-trained hydrological forecasting models on specific data subsets (e.g., country-specific data). The primary use case is to adapt models trained on data-rich regions to data-sparse regions by fine-tuning with a reduced learning rate.

## Overview

Fine-tuning is a transfer learning technique where a model pre-trained on one dataset is adapted to a new dataset by continuing training with a reduced learning rate. This experiment loads pre-trained model checkpoints and adapts them to Central Asian catchment data, allowing knowledge transfer between regions.

## Usage

The experiment is run using the main `experiment.py` script, which accepts several command-line arguments:

```bash
python experiments/FineTuning/experiment.py \
  --model tide \
  --checkpoint-path /path/to/pretrained/checkpoint.ckpt \
  --yaml-path /path/to/model/config.yaml \
  --country Tajikistan \
  --lr-factor 10.0 \
  --epochs 100 \
  --batch-size 2048 \
  --output-dir experiments/FineTuning/results
```

### Required Arguments

- `--model`: Type of model to fine-tune (tide, tsmixer, ealstm, or tft)
- `--checkpoint-path`: Path to the pre-trained model checkpoint
- `--yaml-path`: Path to the model hyperparameter YAML file

### Optional Arguments

- `--country`: Target country to fine-tune on (Tajikistan, Kyrgyzstan, or Combined)
- `--output-dir`: Directory to save fine-tuned checkpoints (default: experiments/FineTuning/results)
- `--lr-factor`: Factor to reduce learning rate by for fine-tuning (default: 10.0)
- `--epochs`: Maximum fine-tuning epochs (default: 100)
- `--batch-size`: Batch size for fine-tuning (default: 2048)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num-runs`: Number of fine-tuning runs to perform (default: 1)

## Multiple Runs

The experiment supports running multiple fine-tuning instances with different random seeds to assess model stability. Use the `--num-runs` parameter to specify the number of runs:

```bash
python experiments/FineTuning/experiment.py \
  --model tide \
  --checkpoint-path /path/to/pretrained/checkpoint.ckpt \
  --yaml-path /path/to/model/config.yaml \
  --country Tajikistan \
  --num-runs 5
```

Each run uses a different seed (base_seed + run_index) but the same pre-trained checkpoint. Results from all runs are aggregated, and the best performing model (based on validation loss) is highlighted in the results.

## Output Structure

Fine-tuning results are organized in the following directory structure:

```
output_dir/
├── checkpoints/
│   ├── [country]/
│   │   └── [model_type]/
│   │       ├── run_0/
│   │       │   └── [model]_run0_[epoch]_[val_loss].ckpt
│   │       ├── run_1/
│   │       │   └── [model]_run1_[epoch]_[val_loss].ckpt
├── logs/
│   ├── [country]/
│   │   └── [model_type]/
│   │       ├── run_0/
│   │       ├── run_1/
└── results/
    ├── [country]/
    │   ├── [model_type]_results.csv
    │   └── [model_type]_best_result.csv
```

The results CSV files contain information about the fine-tuning process, including best validation loss, training epochs, and learning rates for each run.

## Components

- `experiment.py`: Main script for running the experiment
- `config.py`: Configuration for the fine-tuning experiment
- `data_loader.py`: Data loading utilities
- `utils.py`: Helper functions for fine-tuning models

## Example

Fine-tune a TiDE model on Tajikistan data with 3 runs:

```bash
python experiments/FineTuning/experiment.py \
  --model tide \
  --checkpoint-path experiments/DataSharing/checkpoints/combined/tide/run_0/Combined_tide_epoch=15_val_loss=0.2345.ckpt \
  --yaml-path experiments/DataSharing/yaml_files/tide.yaml \
  --country Tajikistan \
  --lr-factor 10.0 \
  --num-runs 3
```

## Dependencies

This experiment relies on the core utilities and model implementations in the `src` directory, particularly:

- `src/models/model_factory.py`: For loading pre-trained models
- `src/data_models/caravanify.py`: For loading Central Asian data
- `src/data_models/datamodule.py`: For creating PyTorch data modules
