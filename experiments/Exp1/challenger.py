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

# ----- CONFIGURE CH DATASET FOR PRETRAINING -----
print("CONFIGURING CH DATASET FOR PRETRAINING")
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
print(f"Loading {len(ch_basins)} CH basins")
ch_caravan.load_stations(ch_basins)

ts_columns = config.FORCING_FEATURES + [config.TARGET]
static_columns = config.STATIC_FEATURES

ch_ts_data = ch_caravan.get_time_series()[ts_columns]
ch_static_data = ch_caravan.get_static_attributes()[static_columns]

# ----- CONFIGURE PREPROCESSING -----
print("CONFIGURING PREPROCESSING")
preprocessing_config = {
    "features": {"scale_method": "per_basin", "log_transform": []},
    "target": {"scale_method": "per_basin", "log_transform": False},
    "static_features": {"scale_method": "global"},
}

# ----- CREATE CH DATA MODULE -----
print("CREATING CH DATA MODULE")
num_workers = multiprocessing.cpu_count()
print(f"Available CPU cores: {num_workers}")

ch_data_module = HydroDataModule(
    time_series_df=ch_ts_data,
    static_df=ch_static_data,
    group_identifier=config.GROUP_IDENTIFIER,
    preprocessing_config=preprocessing_config,
    batch_size=config.BATCH_SIZE,
    input_length=config.INPUT_LENGTH,
    output_length=config.OUTPUT_LENGTH,
    num_workers=num_workers,
    features=ts_columns,
    static_features=static_columns,
    target=config.TARGET,
    min_train_years=config.MIN_TRAIN_YEARS,
    val_years=config.VAL_YEARS,
    test_years=config.TEST_YEARS,
    max_missing_pct=config.MAX_MISSING_PCT,
)

# ----- MODEL SETUP AND PRETRAINING START -----
print("SETTING UP MODEL FOR PRETRAINING")
model = LitTSMixer(
    input_len=config.INPUT_LENGTH,
    output_len=config.OUTPUT_LENGTH,
    input_size=len(ts_columns),
    static_size=len(static_columns) - 1,
    hidden_size=config.HIDDEN_SIZE,
)

pretrain_trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/pretrain",
            filename="ch-pretrain-checkpoint",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)

print("STARTING CH PRETRAINING")
pretrain_trainer.fit(model, ch_data_module)

# ----- CONFIGURE CA DATASET FOR FINE-TUNING -----
print("CONFIGURING CA DATASET FOR FINE-TUNING")
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
print(f"Loading {len(ca_basins)} CA basins")
ca_caravan.load_stations(ca_basins)

ca_ts_data = ca_caravan.get_time_series()[ts_columns]
ca_static_data = ca_caravan.get_static_attributes()[static_columns]

# ----- CREATE CA DATA MODULE -----
print("CREATING CA DATA MODULE")
ca_data_module = HydroDataModule(
    time_series_df=ca_ts_data,
    static_df=ca_static_data,
    group_identifier=config.GROUP_IDENTIFIER,
    preprocessing_config=preprocessing_config,
    batch_size=config.BATCH_SIZE,
    input_length=config.INPUT_LENGTH,
    output_length=config.OUTPUT_LENGTH,
    num_workers=num_workers,
    features=ts_columns,
    static_features=static_columns,
    target=config.TARGET,
    min_train_years=config.MIN_TRAIN_YEARS,
    val_years=config.VAL_YEARS,
    test_years=config.TEST_YEARS,
    max_missing_pct=config.MAX_MISSING_PCT,
)

# ----- FINE-TUNING START -----
print("STARTING CA FINE-TUNING")
finetune_trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator="gpu",
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/finetune",
            filename="ca-finetune-checkpoint",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)

finetune_trainer.fit(model, ca_data_module)

# ----- SAVE FINAL MODEL -----
print("SAVING FINAL MODEL")
model_save_path = Path("experiments/Exp1/saved_models/tsmixer_challenger.pt")
model_save_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print("TRAINING COMPLETE")
