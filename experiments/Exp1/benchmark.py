# ----- IMPORTS AND SUCH (mostly imports) -----

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

import multiprocessing

from experiments.Exp1 import config

if config.ACCELERATOR == "cuda":
    torch.set_float32_matmul_precision("medium")


# ----- CONFIGURE DATASET -----
print("CONFIGURING DATASET")
benchmark_config = CaravanifyConfig(
    attributes_dir=config.CA_ATTRIBUTE_DIR,
    timeseries_dir=config.CA_TIMESERIES_DIR,
    gauge_id_prefix=config.CA_GAUGE_ID_PREFIX,
    use_hydroatlas_attributes=True,
    use_caravan_attributes=True,
    use_other_attributes=True,
)

benchmark_caravan = Caravanify(benchmark_config)

all_basins = benchmark_caravan.get_all_gauge_ids()

benchmark_caravan.load_stations(all_basins)

ts_columns = config.FORCING_FEATURES + [config.TARGET]

ts_data = benchmark_caravan.get_time_series()[
    ts_columns + ["date"] + [config.GROUP_IDENTIFIER]
]

static_columns = config.STATIC_FEATURES

static_data = benchmark_caravan.get_static_attributes()[static_columns]

# ----- CONFIGURE PREPROCESSING (hardcoded for now) -----
print("CONFIGURING PREPROCESSING (hardcoded for now)")

preprocessing_config = {
    "features": {"scale_method": "per_basin", "log_transform": []},
    "target": {"scale_method": "per_basin", "log_transform": False},
    "static_features": {"scale_method": "global"},
}

# ----- CREATE DATA_MODULE -----
print("CREATING DATA_MODULE")

num_workers = multiprocessing.cpu_count()
print(f"Available CPU cores: {num_workers}")

data_module = HydroDataModule(
    time_series_df=ts_data,
    static_df=static_data,
    group_identifier=config.GROUP_IDENTIFIER,
    preprocessing_config=preprocessing_config,
    batch_size=config.BATCH_SIZE,
    input_length=config.INPUT_LENGTH,
    output_length=config.OUTPUT_LENGTH,
    num_workers=num_workers,
    features=ts_columns,
    static_features=static_columns,
    target=config.TARGET,
    min_train_years=config.CA_MIN_TRAIN_YEARS,
    val_years=config.CA_VAL_YEARS,
    test_years=config.CA_TEST_YEARS,
    max_missing_pct=config.CA_MAX_MISSING_PCT,
)

# ----- MODEL SET UP AND TRAINING START -----
print("SETTING UP MODEL")

model = LitTSMixer(
    input_len=config.INPUT_LENGTH,
    output_len=config.OUTPUT_LENGTH,
    input_size=len(ts_columns),
    static_size=len(static_columns) - 1,
    hidden_size=config.HIDDEN_SIZE,
)

trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator=config.ACCELERATOR,
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)

print("STARTING THE TRAINING")

trainer.fit(model, data_module)

# ----- SAVE FINAL MODEL -----
print("SAVING FINAL MODEL")
model_save_path = Path("experiments/Exp1/saved_models/benchmark.pt")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print("TRAINING COMPLETE")
