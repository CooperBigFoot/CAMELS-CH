import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from src.data_models.caravanify import Caravanify, CaravanifyConfig
from src.data_models.datamodule import HydroDataModule
from src.models.TSMixer import LitTSMixer
from src.models.evaluators import TSForecastEvaluator

import multiprocessing

from experiments.Exp2 import config2 as config

if config.ACCELERATOR == "cuda":
    torch.set_float32_matmul_precision("medium")

# ----------------- LOAD DATASETS -----------------
ts_columns = config.FORCING_FEATURES + [config.TARGET]
static_columns = config.STATIC_FEATURES

# Load CH data
print("Configuring CH dataset")
ch_config = CaravanifyConfig(
    attributes_dir=config.CH_ATTRIBUTE_DIR,
    timeseries_dir=config.CH_TIMESERIES_DIR,
    gauge_id_prefix=config.CH_GAUGE_ID_PREFIX,
    use_hydroatlas_attributes=True,
    use_caravan_attributes=True,
    use_other_attributes=True,
)
ch_caravan = Caravanify(ch_config)
ch_basins = ch_caravan.get_all_gauge_ids()
ch_caravan.load_stations(ch_basins)
ch_ts_data = ch_caravan.get_time_series()[
    ts_columns + ["date"] + [config.GROUP_IDENTIFIER]
]
ch_static_data = ch_caravan.get_static_attributes()[static_columns]

# Load CA data
print("Configuring CA dataset")
ca_config = CaravanifyConfig(
    attributes_dir=config.CA_ATTRIBUTE_DIR,
    timeseries_dir=config.CA_TIMESERIES_DIR,
    gauge_id_prefix=config.CA_GAUGE_ID_PREFIX,
    use_hydroatlas_attributes=True,
    use_caravan_attributes=True,
    use_other_attributes=True,
)
ca_caravan = Caravanify(ca_config)
ca_basins = ca_caravan.get_all_gauge_ids()
ca_caravan.load_stations(ca_basins)
ca_ts_data = ca_caravan.get_time_series()[
    ts_columns + ["date"] + [config.GROUP_IDENTIFIER]
]
ca_static_data = ca_caravan.get_static_attributes()[static_columns]

# ----------------- MERGE DATASETS -----------------
print("Merging CA and CH datasets")
bundle_ts_data = pd.concat([ch_ts_data, ca_ts_data], ignore_index=True)
bundle_static_data = pd.concat([ch_static_data, ca_static_data], ignore_index=True)

# ----------------- PREPROCESSING & DATA MODULE -----------------
preprocessing_config = {
    "features": {"scale_method": "per_basin", "log_transform": []},
    "target": {"scale_method": "per_basin", "log_transform": False},
    "static_features": {"scale_method": "global"},
}

num_workers = multiprocessing.cpu_count()
print(f"Using {num_workers} CPU cores")

bundle_data_module = HydroDataModule(
    time_series_df=bundle_ts_data,
    static_df=bundle_static_data,
    group_identifier=config.GROUP_IDENTIFIER,
    preprocessing_config=preprocessing_config,
    batch_size=config.BATCH_SIZE,
    input_length=config.INPUT_LENGTH,
    output_length=config.OUTPUT_LENGTH,
    num_workers=num_workers,
    features=ts_columns,
    static_features=static_columns,
    target=config.TARGET,
    min_train_years=config.BUNDLE_MIN_TRAIN_YEARS,
    val_years=config.BUNDLE_VAL_YEARS,
    test_years=config.BUNDLE_TEST_YEARS,
    max_missing_pct=config.BUNDLE_MAX_MISSING_PCT,
)

# ----------------- MODEL SETUP -----------------
print("Setting up model")
model = LitTSMixer(
    input_len=config.INPUT_LENGTH,
    output_len=config.OUTPUT_LENGTH,
    input_size=len(ts_columns),
    static_size=len(static_columns) - 1,
    hidden_size=config.HIDDEN_SIZE,
)

# ----------------- TRAINING -----------------
trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator=config.ACCELERATOR,
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/bundle",
            filename="bundle-checkpoint",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)
print("Starting training on bundled dataset")
trainer.fit(model, bundle_data_module)

# Save trained model
model_save_path = Path("experiments/Exp1/saved_models/tsmixer_challenger2.pt")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print("Training complete and model saved.")

# ----------------- EVALUATION -----------------
print("Starting evaluation")
trainer.test(model, bundle_data_module)
raw_results = model.test_results

evaluator = TSForecastEvaluator(
    bundle_data_module, horizons=list(range(1, model.config.pred_len + 1))
)
results_df, overall_metrics, basin_metrics = evaluator.evaluate(raw_results)

# Save evaluation results
results_dir = Path("experiments/Exp2/results")
results_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(results_dir / "challenger2_detailed_results.csv", index=True)
overall_metrics_df = evaluator.summarize_metrics(overall_metrics)
overall_metrics_df.to_csv(results_dir / "challenger2_overall_metrics.csv", index=True)
basin_metrics_df = evaluator.summarize_metrics(basin_metrics, per_basin=True)
basin_metrics_df.to_csv(results_dir / "challenger2_basin_metrics.csv", index=True)

print("Evaluation complete.")
