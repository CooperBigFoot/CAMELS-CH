# General configs
GROUP_IDENTIFIER = "gauge_id"
BATCH_SIZE = 32
INPUT_LENGTH = 30
OUTPUT_LENGTH = 10
MAX_EPOCHS = 15
ACCELERATOR = "cuda"

ACCELERATOR = "cpu"

# Model specific configs
HIDDEN_SIZE = 64

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
CA_ATTRIBUTE_DIR = (
    "/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY/CA/post_processed/attributes"
)
CA_TIMESERIES_DIR = (
    "/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY/CA/post_processed/timeseries/csv"
)
CA_GAUGE_ID_PREFIX = "CA"

# Config specific to Switzeerland -> these configs result in 132 valid basins
CH_ATTRIBUTE_DIR = (
    "/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY/CH/post_processed/attributes"
)
CH_TIMESERIES_DIR = (
    "/Users/cooper/Desktop/CAMELS-CH/data/CARAVANIFY/CH/post_processed/timeseries/csv"
)
CH_GAUGE_ID_PREFIX = "CH"

# Bundle configs
BUNDLE_MIN_TRAIN_YEARS = 8
BUNDLE_VAL_YEARS = 2
BUNDLE_TEST_YEARS = 3
BUNDLE_MAX_MISSING_PCT = 10
