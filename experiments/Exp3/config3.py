# General configs
GROUP_IDENTIFIER = "gauge_id"
BATCH_SIZE = 128
INPUT_LENGTH = 64
OUTPUT_LENGTH = 10
MAX_EPOCHS = 15
ACCELERATOR = "cuda"

# Learning rate configs
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-4  # 10x smaller

# Model specific configs
HIDDEN_SIZE = 32

TARGET = "streamflow"
STATIC_FEATURES = [
    "gauge_id",
    "p_mean",
    "area",
    "ele_mt_sav",
    "high_prec_dur",
    "frac_snow",
    "high_prec_freq",
    "slp_dg_sav",
    "cly_pc_sav",
    "aridity_ERA5_LAND",
    "aridity_FAO_PM",
]

FORCING_FEATURES = [
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "surface_net_thermal_radiation_mean",
    "potential_evaporation_sum_ERA5_LAND",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "total_precipitation_sum",
]

# Config specific to Central Asia -> these configs result in 44 valid basins
CA_ATTRIBUTE_DIR = "/workspace/CARAVANIFY/CA/post_processed/attributes"
CA_TIMESERIES_DIR = "/workspace/CARAVANIFY/CA/post_processed/timeseries/csv"
CA_GAUGE_ID_PREFIX = "CA"
CA_MIN_TRAIN_YEARS = 8
CA_VAL_YEARS = 2
CA_TEST_YEARS = 3
CA_MAX_MISSING_PCT = 10

# Config specific to Switzeerland -> these configs result in 132 valid basins
CH_ATTRIBUTE_DIR = "/workspace/CARAVANIFY/CH/post_processed/attributes"
CH_TIMESERIES_DIR = "/workspace/CARAVANIFY/CH/post_processed/timeseries/csv"
CH_GAUGE_ID_PREFIX = "CH"
CH_MIN_TRAIN_YEARS = 20
CH_VAL_YEARS = 10
CH_TEST_YEARS = 0
CH_MAX_MISSING_PCT = 10
