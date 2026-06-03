import os
from pathlib import Path


FETCH_DAYS = int(os.getenv("FETCH_DAYS", "730"))
FETCH_RETRIES = int(os.getenv("FETCH_RETRIES", "3"))
FETCH_RETRY_SECONDS = float(os.getenv("FETCH_RETRY_SECONDS", "20"))
ALLOW_STALE_OBS = os.getenv("ALLOW_STALE_OBS", "1").lower() not in {"0", "false", "no"}
BIAS_DECAY_HOURS = float(os.getenv("BIAS_DECAY_HOURS", "36"))
BIAS_TRAINING_DAYS = int(os.getenv("BIAS_TRAINING_DAYS", "90"))
MIN_BIAS_SAMPLES = int(os.getenv("MIN_BIAS_SAMPLES", "8"))
ARCHIVE_RETENTION_DAYS = int(os.getenv("ARCHIVE_RETENTION_DAYS", "180"))
ECMWF_RUN_HOUR = int(os.getenv("ECMWF_RUN_HOUR", "6"))
ECMWF_STEPS = [
    int(step.strip())
    for step in os.getenv("ECMWF_STEPS", "3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48").split(",")
    if step.strip()
]
ECMWF_PARAMS = [
    param.strip()
    for param in os.getenv("ECMWF_PARAMS", "mx2t3,2t,mn2t3").split(",")
    if param.strip()
]
ECMWF_SOURCES = [
    source.strip()
    for source in os.getenv("ECMWF_SOURCES", os.getenv("ECMWF_SOURCE", "azure,google,aws")).split(",")
    if source.strip()
]
if not ECMWF_SOURCES:
    ECMWF_SOURCES = ["azure", "google", "aws"]
ECMWF_SOURCE_RETRY_SECONDS = float(os.getenv("ECMWF_SOURCE_RETRY_SECONDS", "15"))
FETCH_ALL_INAMHI_COLUMNS = os.getenv("FETCH_ALL_INAMHI_COLUMNS", "0").lower() in {"1", "true", "yes"}

STATION_ID = 63777
STATION_NAME = "Inaquito"
STATION_LAT = -0.178300
STATION_LON = -78.487700

RAW_DATA_PATH = Path("merged_data_export.csv")
ECMWF_GRIB_PATH = Path("surface_temp_multi.grib2")
FORECAST_CSV_PATH = Path("forecast_output.csv")
FORECAST_HTML_PATH = Path("forecast_Inaquito.html")
FORECAST_ARCHIVE_PATH = Path("forecast_archive.csv")
FORECAST_VERIFICATION_PATH = Path("forecast_verification.csv")
FORECAST_METRICS_PATH = Path("forecast_metrics.csv")
SYSTEM_STATUS_PATH = Path("SYSTEM_STATUS.md")
METRICS_HTML_PATH = Path("forecast_metrics.html")

FORECAST_TARGETS = {
    "max": {
        "label": "Max",
        "station_column": "TEMPERATURA AIRE MAX",
        "ecmwf_param": "mx2t3",
        "raw_column": "ecmwf_max_c",
        "forecast_column": "forecast_max_c",
        "persistence_column": "persistence_max_c",
        "lower_column": "forecast_max_lower_c",
        "upper_column": "forecast_max_upper_c",
        "bias_column": "bias_correction_max_c",
    },
    "prom": {
        "label": "Prom",
        "station_column": "TEMPERATURA AIRE PROM",
        "ecmwf_param": "2t",
        "raw_column": "ecmwf_prom_c",
        "forecast_column": "forecast_prom_c",
        "persistence_column": "persistence_prom_c",
        "lower_column": "forecast_prom_lower_c",
        "upper_column": "forecast_prom_upper_c",
        "bias_column": "bias_correction_prom_c",
    },
    "min": {
        "label": "Min",
        "station_column": "TEMPERATURA AIRE MIN",
        "ecmwf_param": "mn2t3",
        "raw_column": "ecmwf_min_c",
        "forecast_column": "forecast_min_c",
        "persistence_column": "persistence_min_c",
        "lower_column": "forecast_min_lower_c",
        "upper_column": "forecast_min_upper_c",
        "bias_column": "bias_correction_min_c",
    },
}
DEFAULT_TARGET_KEY = "prom"
ECMWF_VARIABLE_CANDIDATES = {
    "2t": ["t2m", "2t"],
    "mx2t3": ["mx2t3", "mx2t"],
    "mn2t3": ["mn2t3", "mn2t"],
}

TEMPERATURE_TABLE_NAMES = [
    "029030101h", "029030401h", "029030201h",
]

ALL_TABLE_NAMES = [
    "009010101h", "009010201h", "009010401h",
    "017140801h",
    "018070201h", "018070401h", "018070101h",
    "021200201h", "021200101h", "021200401h", "021200801h",
    "022200201h", "022200401h", "022200101h",
    *TEMPERATURE_TABLE_NAMES,
]
TABLE_NAMES = ALL_TABLE_NAMES if FETCH_ALL_INAMHI_COLUMNS else TEMPERATURE_TABLE_NAMES

NAME_MAP = {
    "009010101h": "HUMEDAD RELATIVA DEL AIRE MAX",
    "009010201h": "HUMEDAD RELATIVA DEL AIRE MIN",
    "009010401h": "HUMEDAD RELATIVA DEL AIRE PROM",
    "017140801h": "PRECIPITACION SUM",
    "018070201h": "PRESION ATMOSFERICA MIN",
    "018070401h": "PRESION ATMOSFERICA PROM",
    "018070101h": "PRESION ATMOSFERICA MAX",
    "021200201h": "RADIACION SOLAR GLOBAL MIN",
    "021200101h": "RADIACION SOLAR GLOBAL MAX",
    "021200401h": "RADIACION SOLAR GLOBAL PROM",
    "021200801h": "RADIACION SOLAR GLOBAL SUM",
    "022200201h": "RADIACION SOLAR REFLEJADA MIN",
    "022200401h": "RADIACION SOLAR REFLEJADA PROM",
    "022200101h": "RADIACION SOLAR REFLEJADA MAX",
    "029030101h": "TEMPERATURA AIRE MAX",
    "029030401h": "TEMPERATURA AIRE PROM",
    "029030201h": "TEMPERATURA AIRE MIN",
}
