# Similarity-Based Transfer Learning Experiment

This experiment evaluates the effectiveness of knowledge transfer from data-rich regions (Switzerland, USA, Chile) to data-sparse Central Asian catchments using hydrological similarity groups. The core hypothesis is that models trained on similar catchments from data-rich regions will perform better on Central Asian catchments than models trained on randomly selected catchments.

## Experiment Overview

The experiment organizes catchments into similarity groups based on prior clustering results. For each group:

1. Central Asian (CA) catchments serve as the target domain
2. Similar catchments from source domains (CH, USA, CL) are identified
3. Models are trained on combined data from both domains
4. Performance is evaluated to assess knowledge transfer effectiveness

By comparing model performance across different groups, we can analyze how hydrological similarity affects transfer learning success.

## Data Description

The experiment uses several data sources:

- **Central Asia (CA)**: Target catchments in data-sparse regions
- **Switzerland (CH)**: Data-rich European Alpine catchments
- **United States (USA)**: Data-rich catchments with diverse climates
- **Chile (CL)**: Data-rich South American catchments

### Similarity Grouping

Catchments are grouped based on clustering results from previous analyses:

- **Group 1 [13, 14]**: Primarily high-elevation mountainous catchments
- **Group 2 [4, 8]**: Primarily semi-arid plains catchments
- **Group 3 [0, 5, 9]**: Primarily forested mid-elevation catchments

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
python experiment.py
```

### Command Line Options

- `--models`: Model types to evaluate (`tide`, `tsmixer`, `ealstm`, `tft`)
- `--groups`: Similarity groups to evaluate (`group1`, `group2`, `group3`)
- `--yaml-dir`: Directory containing model hyperparameter YAML files
- `--output-dir`: Output directory for results
- `--num-runs`: Number of runs for each model/group combination
- `--seed`: Base random seed
- `--max-epochs`: Maximum training epochs

Example with custom settings:

```bash
python experiment.py \
  --models tide tsmixer \
  --groups group1 group3 \
  --num-runs 3 \
  --train-prop 0.7 --val-prop 0.15 --test-prop 0.15 \
  --batch-size 1024
```

## Output Structure

The experiment creates the following directory structure:
