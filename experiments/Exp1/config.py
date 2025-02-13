# General configs
GROUP_IDENTIFIER = "gauge_id"
BATCH_SIZE = 128
INPUT_LENGTH = 60
OUTPUT_LENGTH = 10
MAX_EPOCHS = 1

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
MIN_TRAIN_YEARS = 8
VAL_YEARS = 2
TEST_YEARS = 3
MAX_MISSING_PCT = 10

# Config specific to Switzeerland -> these configs result in 132 valid basins
CH_ATTRIBUTE_DIR = ""
CH_TIMESERIES_DIR = ""
CH_GAUGE_ID_PREFIX = "CH"
MIN_TRAIN_YEARS = 20
VAL_YEARS = 10
TEST_YEARS = 0
MAX_MISSING_PCT = 10
